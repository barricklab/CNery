#!/usr/bin/env python
"""Package a dataset folder, publish it as a GitHub Release asset, and emit its registry entry.

One release per dataset version (``testdata-<name>-v<N>``) carrying a single ``.tar.gz``, so
adding or revising a dataset never touches the others.

    python tests/data/add_testdata.py <folder> --name lambda              # next version, dry run
    python tests/data/add_testdata.py <folder> --name lambda --publish    # next version, for real
    python tests/data/add_testdata.py <folder> --name lambda --version 3  # pin it explicitly

``--version`` defaults to one past the highest already published for that dataset, taking both the
existing releases and ``registry.json`` into account. Pass it explicitly to override.

This is a maintenance script, not a test module. It sits beside the ``registry.json`` it writes and
the ``_fetch.py`` that reads it. pytest does not collect it -- the default ``python_files`` patterns
are ``test_*.py`` and ``*_test.py``, and this matches neither.

Without ``--publish`` this only builds the archive and prints the entry -- nothing is uploaded
and no release is created, so it is safe to run to inspect what would happen. The collision check
still runs on a dry run, so a version clash surfaces before you do the real one.

The folder should hold what CNery actually reads: one coverage table per sequence, named
``<seq_id>.coverage.tsv`` or ``<seq_id>.coverage.csv`` -- a dataset may hold several, as
cwbi_ssym_ht04 does with a chromosome and two plasmids. No BAM and no reference FASTA -- they are not inputs, and shipping
them only inflates the download. A ``dataset.json`` provenance manifest is generated alongside
them if absent; fill in its blank fields before publishing, because authentic data without
provenance stops being reproducible as soon as whoever generated it moves on.
"""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from datetime import date
from pathlib import Path

REPO = "barricklab/CNery"
MANIFEST_NAME = "dataset.json"
REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


class GitHubUnavailable(Exception):
    """Raised when the release list cannot be read, so versions cannot be determined."""


def published_versions(name):
    """Versions of ``name`` already released on GitHub, as a set of ints.

    Raises GitHubUnavailable if gh cannot answer -- an empty set would otherwise be
    indistinguishable from "never published", and picking v1 for an existing dataset is
    exactly the collision this is meant to prevent.
    """
    try:
        result = subprocess.run(
            ["gh", "release", "list", "-R", REPO, "--limit", "500",
             "--json", "tagName", "-q", ".[].tagName"],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError as exc:
        raise GitHubUnavailable("the gh CLI is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise GitHubUnavailable((exc.stderr or "").strip() or "gh release list failed") from exc

    pattern = re.compile(rf"^testdata-{re.escape(name)}-v(\d+)$")
    return {
        int(m.group(1))
        for m in (pattern.match(line.strip()) for line in result.stdout.splitlines())
        if m
    }


def registry_version(name):
    """Version recorded for ``name`` in registry.json, or None.

    Checked alongside GitHub because the two can disagree: a release cut but not yet pasted
    into the registry, or an entry pointing at a tag that was deleted.
    """
    if not REGISTRY_PATH.exists():
        return None
    with open(REGISTRY_PATH) as fh:
        entry = json.load(fh).get("datasets", {}).get(name)
    if not entry:
        return None
    match = re.fullmatch(rf"testdata-{re.escape(name)}-v(\d+)", entry.get("tag", ""))
    return int(match.group(1)) if match else None


def resolve_version(name, requested):
    """Pick the version to publish and verify it does not collide.

    Returns (version, published_set). With ``requested`` None this is one past the highest
    version seen in either source; otherwise it validates the explicit choice.
    """
    published = published_versions(name)
    in_registry = registry_version(name)
    known = published | ({in_registry} if in_registry else set())

    if requested is None:
        version = max(known) + 1 if known else 1
        if known:
            print(f"note: {name} is at v{max(known)}; using v{version}", file=sys.stderr)
        return version, published

    if requested in published:
        sys.exit(
            f"error: testdata-{name}-v{requested} is already published.\n"
            f"Datasets are immutable once released -- pinned sha256s depend on them. "
            f"Omit --version to use v{max(known) + 1}."
        )
    if in_registry is not None and requested == in_registry:
        sys.exit(
            f"error: registry.json already maps {name!r} to testdata-{name}-v{requested}.\n"
            f"Omit --version to use v{max(known) + 1}."
        )
    return requested, published


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_template(name):
    return {
        "name": name,
        "created": date.today().isoformat(),
        "breseq_version": "",
        "breseq_command": "",
        "reads_accession": "",
        "reference_accession": "",
        "exercises": "",
        "notes": "",
    }


def ensure_manifest(folder, name):
    """Write a provenance stub if the folder has none. Returns True if newly created."""
    manifest = folder / MANIFEST_NAME
    if manifest.exists():
        return False
    manifest.write_text(json.dumps(manifest_template(name), indent=2) + "\n")
    return True


def _reproducible(info):
    """Normalise tar member metadata so identical content hashes identically.

    File mtimes, ownership and permissions vary with how the folder was staged and say
    nothing about the data, so zero them out.
    """
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def build_archive(folder, name, out_dir):
    """Tar the folder under a top-level directory named for the dataset.

    Byte-for-byte reproducible: gzip stamps an mtime into its header and tar records one per
    member, so the default settings give a different sha256 every run for identical content.
    That would make the printed hash unverifiable -- nobody could rebuild the archive and
    confirm it matches what was published. mtime=0 on both layers fixes that. (tarfile.add
    already walks directories in sorted order, so member ordering is stable.)
    """
    archive = out_dir / f"{name}.tar.gz"
    with open(archive, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                tar.add(folder, arcname=name, filter=_reproducible)
    return archive


def publish(tag, archive, name, version):
    """Create the release and upload the asset.

    Collision is already ruled out by resolve_version() before the archive is built, so this
    does not re-check.
    """
    subprocess.run(
        [
            "gh", "release", "create", tag, str(archive),
            "-R", REPO,
            "--title", f"Test data: {name} v{version}",
            "--notes", (
                f"Authentic CNery test dataset `{name}`, version {version}.\n\n"
                f"Consumed by the test suite via `tests/data/registry.json` "
                f"(`pytest -m authentic`). Pinned by sha256; do not replace this asset "
                f"in place -- publish a new version instead."
            ),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder holding the <seq_id>.coverage.tsv tables")
    parser.add_argument("--name", required=True, help="dataset name, e.g. lambda")
    parser.add_argument("--version", type=int, default=None,
                        help="dataset version; defaults to one past the highest already published")
    parser.add_argument("--description", default="", help="one line for the registry entry")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="where to write the archive")
    parser.add_argument("--publish", action="store_true", help="actually create the release and upload")
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"error: {args.folder} is not a directory")

    # Resolve and validate the version BEFORE building, so a clash costs nothing and shows up
    # on a dry run rather than only at publish time.
    try:
        version, _ = resolve_version(args.name, args.version)
    except GitHubUnavailable as exc:
        if args.version is None:
            sys.exit(
                f"error: cannot determine the next version for {args.name!r}: {exc}.\n"
                f"Pass --version explicitly if you know which one you want."
            )
        print(f"warning: could not check existing releases ({exc}); "
              f"proceeding with --version {args.version} unverified.", file=sys.stderr)
        version = args.version

    if ensure_manifest(args.folder, args.name):
        print(f"note: wrote a {MANIFEST_NAME} provenance stub into {args.folder} -- "
              f"fill it in before publishing.", file=sys.stderr)

    tag = f"testdata-{args.name}-v{version}"
    archive = build_archive(args.folder, args.name, args.out_dir)
    digest = sha256_of(archive)
    size = archive.stat().st_size

    print(f"built {archive} ({size:,} bytes)", file=sys.stderr)

    if args.publish:
        publish(tag, archive, args.name, version)
        print(f"published {tag}", file=sys.stderr)
    else:
        print(f"dry run -- re-run with --publish to create {tag}", file=sys.stderr)

    entry = {
        "tag": tag,
        "file": archive.name,
        "sha256": digest,
        "bytes": size,
        "description": args.description,
    }
    print(f"\nAdd to tests/data/registry.json under \"datasets\":\n", file=sys.stderr)
    print(f'    "{args.name}": ' + json.dumps(entry, indent=6).replace("\n}", "\n    }"))


if __name__ == "__main__":
    main()
