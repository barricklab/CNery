"""Consistency tests against authentic breseq coverage tables.

Marked ``authentic``. These run by default; use ``pytest -m synthetic`` for the fast offline
inner loop. On a cold cache they download ~125 MB from GitHub Releases, then cache it (set
CNERY_TESTDATA_DIR to relocate). See DEVELOPER.

Four datasets, chosen to differ where it matters rather than to pile up more of the same:

    ltee_ara_p1_50k_shift   0 read groups,  10 cols,  4-line footer, permuted reference
    ltee_ara_m3_32k_2rg     2 read groups,  26 cols, 10-line footer
    ltee_ara_m3_38k         1 read group,   18 cols,  7-line footer
    ltee_ara_p5_75k_exp     0 read groups,   5 cols,  4-line footer, --total-only, OTR fires

The footer lengths are the point of that column: 4, 10 and 7, so these are real regression
coverage for the prefix-based footer stripping that replaced ``skipfooter=4`` -- which would
have silently deleted 6 and 3 data rows from the 32k and 38k tables.

The last row covers two things nothing else does: the ``--total-only`` schema, where bam2cov
ships the sums it has already taken rather than strand-split counts, and a sample where OTR
correction actually FIRES. It is an exponential-phase culture, so replication forks are active
and the origin-to-terminus gradient is real -- 1.95x peak to trough, against nothing detectable
in the three stationary-phase samples. See TestOriginTerminus.

These assertions record what the code does TODAY. That is deliberate: a tripwire for unintended
change, not a claim the current output is correct.

After an intended behavior change, refresh the goldens and review the diff::

    pytest -m authentic --regenerate-goldens
    git diff tests/data/expected/

Golden tests report as SKIPPED on that run, not passed -- a run that rewrote its own
expectations has verified nothing.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from CNery.core import (
    STRAND_SPLIT_COLUMNS,
    TOTAL_ONLY_COLUMNS,
    _read_coverage_table,
    apply_otr_correction,
    coverage_table_path,
    fit_otr_bias,
    genome_id_from_path,
    preprocess,
    process_multi_genome,
    resolve_coverage_inputs,
    run_HMM,
)
from data._fetch import load_registry

pytestmark = pytest.mark.authentic

GENOME_LEN = 4_629_812

# Goldens are only meaningful at fixed windowing.
WIN, STEP, FRAG = 1000, 500, 150

EXPECTED_DIR = os.path.join(os.path.dirname(__file__), "data", "expected")


@dataclass(frozen=True)
class Spec:
    seq_id: str
    read_groups: int              # 0 = no --per-read-group columns at all
    footer_lines: int             # 4 aggregate + 3 per read group
    data_columns: int             # excluding the position index
    schema: str = "strand_split"  # or "total_only" (bam2cov --total-only)
    otr_detected: bool = False    # does otr_fit find a replication gradient?


DATASETS = {
    "ltee_ara_p1_50k_shift": Spec("REL606_2314906bp_shift", 0, 4, 9, otr_detected=True),
    "ltee_ara_m3_32k_2rg": Spec("REL606", 2, 10, 25),
    "ltee_ara_m3_38k": Spec("REL606", 1, 7, 17),
    "ltee_ara_p5_75k_exp": Spec("REL606", 0, 4, 4,
                                schema="total_only", otr_detected=True),
}

ALL = list(DATASETS)
WITH_READ_GROUPS = [n for n, s in DATASETS.items() if s.read_groups]
WITH_OTR = [n for n, s in DATASETS.items() if s.otr_detected]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def dataset_dir(request):
    """Extracted directory for the dataset named by the parametrized `request.param`."""
    from conftest import _dataset_or_skip

    return request.param, _dataset_or_skip(request.param)


@pytest.fixture(scope="session")
def raw_table(dataset_dir):
    name, path = dataset_dir
    return name, _read_coverage_table(coverage_table_path(str(path), DATASETS[name].seq_id))


@pytest.fixture(scope="session")
def pipeline(dataset_dir, tmp_path_factory):
    """Run the full pipeline once per dataset and share the result."""
    name, path = dataset_dir
    spec = DATASETS[name]
    out = tmp_path_factory.mktemp(f"out_{name}")
    for sub in ("CNV_plt", "CNV_csv", "GC_bias", "OTR_corr"):
        (out / sub).mkdir()

    per_genome = process_multi_genome(
        resolve_coverage_inputs([str(path)]),
        output_prefix=str(out),
        win=WIN, step=STEP, frag=FRAG,
    )
    df_gc = per_genome[spec.seq_id]
    # otr_correction(df, out) was split into fit_otr_bias() + apply_otr_correction().
    # df_gc already carries is_deletion/is_redundant from the mask_coverage_windows()
    # call inside process_multi_genome()'s GC stage, so fit_otr_bias() reuses them.
    df_otr, ori, ter = apply_otr_correction(fit_otr_bias(df_gc, str(out)), str(out))
    df_cnv = run_HMM(df_otr, str(out))
    return {"name": name, "spec": spec, "out": out,
            "gc": df_gc, "otr": df_otr, "cnv": df_cnv}


def _produced(pipeline, subdir, suffix):
    prefix = os.path.basename(str(pipeline["out"]))
    return os.path.join(pipeline["out"], subdir,
                        f"{prefix}{pipeline['spec'].seq_id}{suffix}")


def _coverage_columns(name):
    """The coverage columns this dataset's schema is required to carry.

    bam2cov writes either strand-split counts or, under --total-only, the sums
    it has already taken. `raw_table` reads the file with _read_coverage_table,
    which deliberately does NOT normalize, so the expectation has to follow the
    schema rather than assume one of them.
    """
    if DATASETS[name].schema == "total_only":
        return TOTAL_ONLY_COLUMNS
    return STRAND_SPLIT_COLUMNS


def _golden(name, suffix):
    return os.path.join(EXPECTED_DIR, f"{name}{suffix}")


# --------------------------------------------------------------------------- fetch


@pytest.mark.parametrize("dataset_dir", ALL, indirect=True)
class TestFetch:
    """Proves the published assets are what the tests actually run on.

    Without this a green run only shows that *some* bytes were on disk; a stale or
    hand-placed cache would look identical to a working download.
    """

    def _cached(self, name):
        import pooch

        from data._fetch import CACHE_ENV_VAR

        entry = load_registry()[name]
        cache = os.environ.get(CACHE_ENV_VAR) or str(pooch.os_cache("cnery"))
        return os.path.join(cache, entry["file"]), entry

    def test_archive_landed_in_the_cache(self, dataset_dir):
        path, _ = self._cached(dataset_dir[0])
        assert os.path.isfile(path)

    def test_cached_archive_matches_the_registry_hash(self, dataset_dir):
        # pooch verifies on download; re-checking catches later corruption of a warm cache,
        # which no download-time check can see.
        import hashlib

        path, entry = self._cached(dataset_dir[0])
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        assert digest.hexdigest() == entry["sha256"]

    def test_archive_size_matches_the_registry(self, dataset_dir):
        path, entry = self._cached(dataset_dir[0])
        assert os.path.getsize(path) == entry["bytes"]


# --------------------------------------------------------------------------- input integrity


@pytest.mark.parametrize("dataset_dir", ALL, indirect=True)
class TestInputIntegrity:
    def test_no_bam_is_shipped(self, dataset_dir):
        _, path = dataset_dir
        assert not os.path.exists(os.path.join(str(path), "reference.bam"))

    def test_every_position_survives_parsing(self, raw_table):
        # The direct regression for skipfooter=4: these tables have 4-, 7- and 10-line
        # footers, so any fixed-count skip loses real data from at least two of them.
        name, df = raw_table
        assert len(df) == GENOME_LEN
        assert df.index[0] == 1
        assert df.index[-1] == GENOME_LEN

    def test_position_is_the_index(self, raw_table):
        assert raw_table[1].index.name == "position"

    def test_footer_did_not_leak_into_the_data(self, raw_table):
        name, df = raw_table
        assert not df["ref_base"].astype(str).str.startswith("#").any()
        assert df[_coverage_columns(name)[0]].dtype.kind in "iu"

    def test_columns_cnery_depends_on(self, raw_table):
        name, df = raw_table
        for col in ("ref_base",) + _coverage_columns(name):
            assert col in df.columns

    def test_column_count_matches_spec(self, raw_table):
        name, df = raw_table
        assert df.shape[1] == DATASETS[name].data_columns

    def test_seq_id_is_recoverable_from_the_file_name(self, dataset_dir):
        # The sequence ID now comes from the file name rather than a FASTA header, so
        # the naming rule has to hold against real file names -- including
        # "REL606_2314906bp_shift", which the "strip only the ending" rule must leave
        # intact rather than cutting at a dot.
        name, path = dataset_dir
        table = coverage_table_path(str(path), DATASETS[name].seq_id)
        assert os.path.isfile(table)
        assert genome_id_from_path(table) == DATASETS[name].seq_id

    def test_resolving_the_folder_finds_exactly_this_table(self, dataset_dir):
        name, path = dataset_dir
        assert list(resolve_coverage_inputs([str(path)])) == [DATASETS[name].seq_id]


# --------------------------------------------------------------------------- read groups


@pytest.mark.parametrize("dataset_dir", WITH_READ_GROUPS, indirect=True)
class TestPerReadGroupColumns:
    """--per-read-group appends columns; CNery must read past them untouched."""

    def test_read_group_columns_are_present(self, raw_table):
        name, df = raw_table
        for g in range(DATASETS[name].read_groups):
            assert f"RG-{g}_unique_top_cov" in df.columns

    def test_aggregate_columns_keep_their_names(self, raw_table):
        # breseq appends the repeats after the aggregates precisely so CNery keeps working;
        # if that ever changed to interleaving, this is what would catch it.
        name, df = raw_table
        assert list(df.columns)[:5] == [
            "ref_base", "unique_top_cov", "unique_bot_cov",
            "redundant_top_cov", "redundant_bot_cov",
        ]

    def test_calls_ignore_the_extra_columns(self, raw_table, pipeline):
        # Windowing the aggregate-only subset must give exactly what the full table gives.
        name, df = raw_table
        trimmed = df[["ref_base", "unique_top_cov", "unique_bot_cov",
                      "redundant_top_cov", "redundant_bot_cov"]]
        got = preprocess(trimmed, win=WIN, step=STEP, frag=FRAG)
        assert len(got) == len(pipeline["gc"])
        np.testing.assert_allclose(
            got["read_count_cov"].to_numpy(),
            pipeline["gc"]["read_count_cov"].to_numpy(),
        )


# --------------------------------------------------------------------------- pipeline shape


@pytest.mark.parametrize("dataset_dir", ALL, indirect=True)
class TestPipelineShape:
    def test_row_count_preserved_end_to_end(self, pipeline):
        assert len(pipeline["cnv"]) == len(pipeline["gc"])

    def test_genome_id_survives(self, pipeline):
        assert (pipeline["cnv"]["genome_id"] == pipeline["spec"].seq_id).all()

    def test_no_nan_in_calls(self, pipeline):
        assert not pipeline["cnv"]["prob_copy_number"].isna().any()

    def test_corrected_coverage_is_finite_and_non_negative(self, pipeline):
        values = pipeline["cnv"]["otr_gc_corr_norm_cov"].to_numpy()
        assert np.isfinite(values).all()
        assert (values >= 0).all()

    def test_window_count_is_stable(self, pipeline):
        # preprocess() keeps every repeat-overlapping window (flagging pct_redundant
        # instead of dropping it) and emits only full-width windows, so the count is
        # exactly the number of `win`-wide windows that fit at stride `step`.
        assert len(pipeline["gc"]) == (GENOME_LEN - WIN) // STEP + 1 == 9258

    def test_repeat_windows_are_retained_and_flagged(self, pipeline):
        # All three references have known repeat content; the windows over it must be
        # present in the frame, not dropped, and carry a nonzero pct_redundant.
        df = pipeline["gc"]
        assert "pct_redundant" in df.columns
        assert (df["pct_redundant"] > 0).any()
        assert df["is_redundant"].equals(df["pct_redundant"] > 0)

    def test_gc_correction_does_not_resurrect_deletions(self, pipeline):
        df = pipeline["gc"]
        deleted = df["read_count_cov"] <= df["read_count_cov"].median() * 0.1
        assert deleted.any(), "dataset should contain real deletions"
        assert (df.loc[deleted, "gc_corr_norm_cov"] == 0.0).all()


# --------------------------------------------------------------------------- OTR


@pytest.mark.parametrize("dataset_dir", ALL, indirect=True)
class TestOriginTerminus:
    """Whether OTR correction fires, per dataset, and that it does real work when it does.

    `otr_fit` gates on two things: origin and terminus 35-65% of the genome apart as a
    circular distance, and a fitted anchor ratio above `bias_threshold`. Which datasets
    clear that is a property of the sample, so `Spec.otr_detected` records it rather than
    the suite assuming one answer for all of them.

    ltee_ara_p5_75k_exp is the reason this class is parametrized rather than blanket: it is
    an EXPONENTIAL-phase culture, so replication forks are active and the gradient is real
    -- measured independently of the fit at 1.95x peak-to-trough, 47.4% apart. The two
    stationary-phase Ara-3 samples sit at 31.6% and 14.2% separation and cannot clear the
    gate at all.

    These assertions fail if detection starts OR stops happening on any dataset -- either
    is a real change worth noticing.
    """

    def test_detection_matches_the_spec(self, pipeline):
        # Read what the run PRODUCED, not the golden -- asserting that the golden says what
        # the golden says would be circular and would pass no matter what the code did.
        with open(_produced(pipeline, "OTR_corr", "_otr_results.json")) as fh:
            ratio = json.load(fh)["Origin-to-Termius/Bias Ratio"]
        if pipeline["spec"].otr_detected:
            assert ratio != "Not detected", "expected OTR correction to fire"
            assert float(ratio) > 1.0, f"origin should out-cover terminus, got {ratio}"
        else:
            assert ratio == "Not detected"

    def test_coverage_passes_through_when_no_bias_is_found(self, pipeline):
        # With no bias detected, otr_correction must leave the GC-corrected values alone.
        if pipeline["spec"].otr_detected:
            pytest.skip("OTR fires on this dataset; see test_correction_tightens_coverage")
        df = pipeline["otr"]
        np.testing.assert_allclose(
            df["otr_gc_corr_norm_cov"].to_numpy(),
            df["gc_corr_norm_cov"].to_numpy(),
            rtol=1e-9,
        )

    def test_correction_tightens_coverage(self, pipeline):
        """The correction must pull coverage TOWARD the single-copy level.

        Reporting a ratio is not evidence that anything was corrected. On
        ltee_ara_p5_75k_exp the fraction of windows within 20% of single-copy goes from
        53% to 95%; without that, `test_single_copy_dominates` below could not pass.

        This is also the regression test for the ori/ter label swap: otr_fit's objective is
        symmetric under exchanging the two breakpoints, so a fit that comes back mirrored
        divides by an INVERTED ramp and spreads coverage out instead of tightening it.
        """
        if not pipeline["spec"].otr_detected:
            pytest.skip("no OTR correction applied on this dataset")
        df = pipeline["otr"]
        near = lambda v: float(((v > 0.8) & (v < 1.2)).mean())
        before = near(df["gc_corr_norm_cov"].to_numpy())
        after = near(df["otr_gc_corr_norm_cov"].to_numpy())
        assert after > before, (
            f"OTR correction did not tighten coverage: {before:.1%} -> {after:.1%}. "
            "An inverted ori/ter labelling would do exactly this."
        )
        assert after > 0.85

    def test_otr_json_matches_golden(self, pipeline, regenerate_goldens):
        from conftest import golden_compare

        def compare(got_path, want_path):
            with open(got_path) as fh:
                got = json.load(fh)
            with open(want_path) as fh:
                want = json.load(fh)
            assert got["Origin-to-Termius/Bias Ratio"] == want["Origin-to-Termius/Bias Ratio"]
            assert got["Origin window"] == want["Origin window"]
            assert got["Terminus window"] == want["Terminus window"]

        golden_compare(
            _produced(pipeline, "OTR_corr", "_otr_results.json"),
            _golden(pipeline["name"], "_otr_results.json"),
            regenerate_goldens, compare,
        )


# --------------------------------------------------------------------------- copy number


@pytest.mark.parametrize("dataset_dir", ALL, indirect=True)
class TestCopyNumber:
    def test_segments_match_golden(self, pipeline, regenerate_goldens):
        from conftest import golden_compare

        def compare(got_path, want_path):
            pd.testing.assert_frame_equal(pd.read_csv(got_path), pd.read_csv(want_path))

        golden_compare(
            _produced(pipeline, "CNV_csv", "_break_pts.csv"),
            _golden(pipeline["name"], "_break_pts.csv"),
            regenerate_goldens, compare,
        )

    def test_single_copy_dominates(self, pipeline):
        calls = pipeline["cnv"]["prob_copy_number"]
        assert calls.mode().iloc[0] == 1
        assert (calls == 1).mean() > 0.8

    def test_no_segment_starts_on_a_repeat_window(self, pipeline):
        """Repeat pile-ups must not invent breakpoints.

        Coverage over a repeat reflects how many copies collapsed onto that locus,
        not the sample's copy number there. Before run_HMM() censored `is_redundant`
        windows from the observation sequence, those windows both invented
        high-CN segments of their own (State 5-10 on all three datasets) and split
        genuine deletions in two. Segment starts come from the win_st of windows
        that were actually observed, so a redundant window can no longer be one.
        """
        df = pipeline["cnv"]
        redundant_starts = set(df.loc[df["is_redundant"], "win_st"])
        produced = pd.read_csv(_produced(pipeline, "CNV_csv", "_break_pts.csv"))
        # The first segment starts at 0, which is a coordinate rather than a window.
        offenders = sorted(set(produced["Startpos"][1:]) & redundant_starts)
        assert not offenders, f"segments start on repeat windows: {offenders}"

    def test_repeat_windows_inherit_a_call(self, pipeline):
        # Censored from the fit, but not left uncalled -- every window keeps an
        # integer copy number so the CNV.csv stays complete.
        df = pipeline["cnv"]
        assert df["is_redundant"].any()
        assert not df.loc[df["is_redundant"], "prob_copy_number"].isna().any()

    def test_deletions_and_amplifications_are_called(self, pipeline):
        states = set(pipeline["cnv"]["prob_copy_number"].unique())
        assert 0 in states, "expected called deletions"
        assert max(states) >= 2, "expected a called amplification"
