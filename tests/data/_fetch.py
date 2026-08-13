"""Fetch and cache the authentic test datasets published as GitHub Release assets.

Datasets are declared in ``registry.json`` and downloaded by pooch, which verifies each
archive against the sha256 pinned there before extracting it. Because the hash is what
establishes identity, a dataset that is replaced in place on the release fails loudly
instead of silently changing what the tests measure.

Each dataset is its own release (``testdata-<name>-v<N>``), so pooch's global ``version=``
is deliberately unused -- per-dataset URLs are supplied through ``urls=`` instead.
"""

import json
import os
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "registry.json"

REPO = "barricklab/CNery"
ASSET_URL = "https://github.com/{repo}/releases/download/{tag}/{file}"

#: Override the cache location. Useful in CI, and for keeping full-size BAMs off a small disk.
CACHE_ENV_VAR = "CNERY_TESTDATA_DIR"


def load_registry():
    """Return the ``datasets`` mapping from registry.json."""
    with open(REGISTRY_PATH) as fh:
        return json.load(fh).get("datasets", {})


def dataset_url(entry):
    return ASSET_URL.format(repo=REPO, tag=entry["tag"], file=entry["file"])


def _make_pooch(datasets):
    import pooch

    # pooch keys everything by file name, so two datasets must not share one.
    registry = {e["file"]: f"sha256:{e['sha256']}" for e in datasets.values()}
    urls = {e["file"]: dataset_url(e) for e in datasets.values()}

    return pooch.create(
        path=os.environ.get(CACHE_ENV_VAR) or pooch.os_cache("cnery"),
        base_url="",  # unused: every file carries its own URL
        registry=registry,
        urls=urls,
        retry_if_failed=2,
    )


def fetch_dataset(name):
    """Download (if needed), verify, and extract dataset ``name``.

    Returns the directory holding the extracted files.

    Raises ``DatasetUnavailable`` when the dataset cannot be obtained -- callers in the
    test suite turn that into a skip carrying the reason, so a failed fetch never
    masquerades as a pass.
    """
    datasets = load_registry()
    if name not in datasets:
        raise DatasetUnavailable(
            f"dataset {name!r} is not in {REGISTRY_PATH}. "
            f"Known datasets: {sorted(datasets) or '(none yet)'}"
        )

    entry = datasets[name]
    url = dataset_url(entry)

    try:
        import pooch
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise DatasetUnavailable(
            f"pooch is required to fetch dataset {name!r} but is not installed "
            f"({exc}). It is in dev-environment.yml."
        ) from exc

    try:
        paths = _make_pooch(datasets).fetch(
            entry["file"], processor=pooch.Untar(extract_dir=name)
        )
    except Exception as exc:
        raise DatasetUnavailable(
            f"could not fetch dataset {name!r} from {url}: {type(exc).__name__}: {exc}"
        ) from exc

    if not paths:
        raise DatasetUnavailable(f"dataset {name!r} extracted to an empty file list ({url})")

    # Untar returns every extracted member; the dataset root is their common parent.
    # commonpath() of a single element returns that element, so a one-file dataset would
    # otherwise hand back the file itself and callers doing `root / "reference.fasta"`
    # would fail with a confusing NotADirectoryError.
    root = Path(os.path.commonpath([str(p) for p in paths]))
    return root.parent if root.is_file() else root


class DatasetUnavailable(Exception):
    """Raised when an authentic dataset cannot be fetched, with the reason attached."""
