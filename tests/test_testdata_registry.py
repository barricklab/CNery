"""Tests for the authentic-dataset fetch machinery itself.

These are deliberately NOT marked ``authentic``: they need no external network and must keep
running in the default suite. If the plumbing only ran under ``-m authentic``, a broken
registry or fetch path would go unnoticed for exactly as long as nobody opted in.

The end-to-end fetch is exercised against a localhost HTTP server rather than GitHub, because
pooch has no ``file://`` downloader and stubbing the download would skip the hash check that
is the whole point.
"""

import functools
import hashlib
import http.server
import json
import re
import socketserver
import tarfile
import threading

import pytest

from data import _fetch
from data._fetch import DatasetUnavailable, dataset_url, load_registry

REQUIRED_FIELDS = {"tag", "file", "sha256", "bytes", "description"}


# --------------------------------------------------------------------------- registry shape


def test_registry_is_valid_json_with_datasets_key():
    with open(_fetch.REGISTRY_PATH) as fh:
        raw = json.load(fh)
    assert "datasets" in raw
    assert isinstance(raw["datasets"], dict)


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_every_entry_has_required_fields(field):
    for name, entry in load_registry().items():
        assert field in entry, f"dataset {name!r} is missing {field!r}"


def test_every_sha256_is_well_formed():
    # Guards against a truncated or mispasted digest from dev/add_testdata.py, which would
    # otherwise surface as a confusing download failure much later.
    for name, entry in load_registry().items():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), \
            f"dataset {name!r} has a malformed sha256: {entry['sha256']!r}"


def test_tags_follow_the_per_dataset_scheme():
    for name, entry in load_registry().items():
        assert re.fullmatch(rf"testdata-{re.escape(name)}-v\d+", entry["tag"]), \
            f"dataset {name!r} has tag {entry['tag']!r}, expected testdata-{name}-v<N>"


def test_file_names_are_unique_across_datasets():
    # pooch keys its registry by file name, so a collision would make one dataset shadow another.
    files = [e["file"] for e in load_registry().values()]
    assert len(files) == len(set(files)), f"duplicate asset file names: {files}"


def test_dataset_url_points_at_the_release_asset():
    url = dataset_url({"tag": "testdata-lambda-v1", "file": "lambda.tar.gz"})
    assert url == (
        "https://github.com/barricklab/CNery/releases/download/"
        "testdata-lambda-v1/lambda.tar.gz"
    )


# --------------------------------------------------------------------------- failure paths


def test_unknown_dataset_names_itself_in_the_error():
    with pytest.raises(DatasetUnavailable, match="nonexistent"):
        _fetch.fetch_dataset("nonexistent")


def test_unknown_dataset_becomes_a_skip_not_a_pass():
    # _dataset_or_skip must convert unavailability into a skip carrying the reason.
    # Note pytest.skip raises Skipped, which derives from BaseException -- catching plain
    # Exception here would let the skip escape and mark THIS test skipped rather than passed.
    from conftest import _dataset_or_skip

    with pytest.raises(pytest.skip.Exception) as excinfo:
        _dataset_or_skip("nonexistent")
    assert "nonexistent" in str(excinfo.value)
    assert "registry.json" in str(excinfo.value)


# --------------------------------------------------------------------------- end-to-end fetch


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture
def local_assets(tmp_path_factory):
    """Serve a temp directory over localhost HTTP; yields (root_dir, base_url)."""
    root = tmp_path_factory.mktemp("assets")
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _make_archive(root, name="smoke", members=None):
    """Build a dataset archive shaped like a real one, return (path, sha256).

    ``members`` maps relative path -> contents; defaults to a nested layout.
    """
    if members is None:
        members = {"reference.fasta": ">chr1\nACGT\n", "sub/nested.txt": "nested\n"}

    folder = root / name
    folder.mkdir(parents=True)
    for rel, text in members.items():
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    archive = root / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(folder, arcname=name)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, digest


@pytest.fixture
def served_dataset(local_assets, tmp_path, monkeypatch):
    """Publish a fake dataset over localhost and point _fetch at it."""
    root, base_url = local_assets
    _, digest = _make_archive(root)

    entry = {
        "tag": "testdata-smoke-v1",
        "file": "smoke.tar.gz",
        "sha256": digest,
        "bytes": 0,
        "description": "synthetic smoke dataset",
    }

    monkeypatch.setattr(_fetch, "ASSET_URL", base_url + "/{file}")
    monkeypatch.setenv(_fetch.CACHE_ENV_VAR, str(tmp_path / "cache"))
    return entry


def test_fetch_downloads_verifies_and_extracts(served_dataset, monkeypatch):
    monkeypatch.setattr(_fetch, "load_registry", lambda: {"smoke": served_dataset})

    path = _fetch.fetch_dataset("smoke")

    assert (path / "reference.fasta").read_text() == ">chr1\nACGT\n"
    assert (path / "sub" / "nested.txt").read_text() == "nested\n"


def test_single_file_dataset_still_returns_a_directory(local_assets, tmp_path, monkeypatch):
    # commonpath() of one element returns that element, so a one-file dataset used to hand
    # back the file itself and `root / "reference.fasta"` blew up with NotADirectoryError.
    root, base_url = local_assets
    _, digest = _make_archive(root, "solo", members={"reference.fasta": ">chr1\nACGT\n"})
    entry = {"tag": "testdata-solo-v1", "file": "solo.tar.gz", "sha256": digest,
             "bytes": 0, "description": "single-file dataset"}

    monkeypatch.setattr(_fetch, "ASSET_URL", base_url + "/{file}")
    monkeypatch.setenv(_fetch.CACHE_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.setattr(_fetch, "load_registry", lambda: {"solo": entry})

    path = _fetch.fetch_dataset("solo")

    assert path.is_dir()
    assert (path / "reference.fasta").read_text() == ">chr1\nACGT\n"


def test_datasets_extract_into_separate_directories(local_assets, tmp_path, monkeypatch):
    # Two datasets may legitimately contain identically named members (reference.fasta is
    # the norm). Extraction must keep them apart, or one silently shadows the other.
    root, base_url = local_assets
    datasets = {}
    for name in ("alpha", "beta"):
        _, digest = _make_archive(root, name, members={"reference.fasta": f">{name}\n"})
        datasets[name] = {"tag": f"testdata-{name}-v1", "file": f"{name}.tar.gz",
                          "sha256": digest, "bytes": 0, "description": ""}

    monkeypatch.setattr(_fetch, "ASSET_URL", base_url + "/{file}")
    monkeypatch.setenv(_fetch.CACHE_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.setattr(_fetch, "load_registry", lambda: datasets)

    alpha = _fetch.fetch_dataset("alpha")
    beta = _fetch.fetch_dataset("beta")

    assert alpha != beta
    assert (alpha / "reference.fasta").read_text() == ">alpha\n"
    assert (beta / "reference.fasta").read_text() == ">beta\n"


def test_second_fetch_uses_the_cache(served_dataset, monkeypatch):
    monkeypatch.setattr(_fetch, "load_registry", lambda: {"smoke": served_dataset})

    first = _fetch.fetch_dataset("smoke")
    second = _fetch.fetch_dataset("smoke")

    assert first == second


def test_wrong_hash_fails_loudly(served_dataset, monkeypatch):
    # The registry hash -- not the tag -- is what pins dataset identity. An asset replaced in
    # place must fail, never silently change what the tests run against.
    tampered = dict(served_dataset, sha256="0" * 64)
    monkeypatch.setattr(_fetch, "load_registry", lambda: {"smoke": tampered})

    with pytest.raises(DatasetUnavailable) as excinfo:
        _fetch.fetch_dataset("smoke")
    assert "smoke" in str(excinfo.value)


def test_unreachable_url_reports_the_dataset_and_url(served_dataset, monkeypatch):
    monkeypatch.setattr(_fetch, "ASSET_URL", "http://127.0.0.1:1/{file}")
    monkeypatch.setattr(_fetch, "load_registry", lambda: {"smoke": served_dataset})

    with pytest.raises(DatasetUnavailable) as excinfo:
        _fetch.fetch_dataset("smoke")
    message = str(excinfo.value)
    assert "smoke" in message
    assert "127.0.0.1:1" in message
