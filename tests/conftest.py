import os
import shutil

import pytest
import numpy as np
import pandas as pd

from data._fetch import DatasetUnavailable, fetch_dataset, load_registry


def pytest_collection_modifyitems(config, items):
    """Mark everything not `authentic` as `synthetic`.

    Applied automatically rather than by hand so the two markers stay exhaustive and mutually
    exclusive by construction: a new test cannot end up in neither tier and quietly escape
    both `-m synthetic` and `-m authentic`.
    """
    for item in items:
        if item.get_closest_marker("authentic") is None:
            item.add_marker(pytest.mark.synthetic)


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-goldens",
        action="store_true",
        default=False,
        help=(
            "Overwrite golden files under tests/data/expected/ with the current output "
            "instead of comparing against them. Review the resulting diff before committing."
        ),
    )


@pytest.fixture(scope="session")
def regenerate_goldens(request):
    return request.config.getoption("--regenerate-goldens")


def golden_compare(produced_path, golden_path, regenerate, compare):
    """Compare `produced_path` against `golden_path`, or refresh the golden.

    When regenerating, the test is skipped rather than passed: a run that rewrote its own
    expectations has verified nothing, and reporting it as a pass would hide that.
    """
    if regenerate:
        shutil.copyfile(produced_path, golden_path)
        pytest.skip(f"regenerated golden {os.path.basename(golden_path)} -- review the diff")
    compare(produced_path, golden_path)


def _dataset_or_skip(name):
    """Fetch an authentic dataset, or skip with the reason attached.

    The reason names the dataset, URL, and underlying error so that a run reporting
    "skipped" is visibly different from one reporting "passed" -- a silent skip would
    let real-data coverage lapse without anyone noticing.
    """
    try:
        return fetch_dataset(name)
    except DatasetUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def authentic_registry():
    """The declared datasets, without fetching anything."""
    return load_registry()


@pytest.fixture(scope="session")
def lambda_dataset():
    """breseq lambda output: multiple references, small enough for end-to-end runs."""
    return _dataset_or_skip("lambda")


STRAND_SPLIT_HEADER = [
    "position", "ref_base",
    "unique_top_cov", "unique_bot_cov",
    "redundant_top_cov", "redundant_bot_cov",
]


def write_coverage_table(path, seq, cov=25, delimiter="\t"):
    """Write a minimal strand-split bam2cov table over `seq`, footer included.

    The windowed fixtures below carry no sequence at all, so anything testing
    behaviour that depends on the reference bases -- GC content, GC skew -- has
    to start from a real table. Mirrors the shape tests/test_cli.py builds.
    """
    lines = [delimiter.join(STRAND_SPLIT_HEADER)]
    lines += [
        delimiter.join(str(v) for v in (i + 1, base, cov, cov, 0, 0))
        for i, base in enumerate(seq)
    ]
    lines.append(delimiter.join(("#", "number_of_positions", str(len(seq)))))
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_windowed_df(n=80, del_start=None, del_end=None,
                      amp_start=None, amp_end=None, median_cov=100.0):
    rng = np.random.default_rng(7)
    rc = rng.normal(median_cov, 5.0, n).clip(1).astype(float)
    if del_start is not None:
        rc[del_start:del_end] = 0.0
    if amp_start is not None:
        rc[amp_start:amp_end] = median_cov * 2.0
    win_st = np.arange(n) * 200
    gc = np.clip(0.50 + rng.normal(0, 0.02, n), 0.30, 0.70)
    med = np.median(rc[rc > 0]) if np.any(rc > 0) else 1.0
    return pd.DataFrame({
        "genome_id": "chr1",
        "win_st": win_st,
        "win_end": win_st + 200,
        "win_len": 200,
        "gc_percent": gc,
        "read_count_cov": rc,
        "norm_raw_cov": rc / med,
        "window_num": np.arange(n),
    })


@pytest.fixture
def windowed_flat():
    return _make_windowed_df()


@pytest.fixture
def windowed_with_deletion():
    return _make_windowed_df(del_start=30, del_end=50)


@pytest.fixture
def windowed_with_amplification():
    return _make_windowed_df(amp_start=30, amp_end=45)


@pytest.fixture
def gc_corrected_flat(windowed_flat):
    df = windowed_flat.copy()
    df["gc_corr_norm_cov"] = df["norm_raw_cov"].copy()
    df["gc_corr_fact"] = np.ones(len(df))
    return df


@pytest.fixture
def otr_corrected_flat(gc_corrected_flat):
    df = gc_corrected_flat.copy()
    df["otr_gc_corr_norm_cov"] = df["gc_corr_norm_cov"].copy()
    df["otr_gc_corr_fact"] = np.ones(len(df))
    return df
