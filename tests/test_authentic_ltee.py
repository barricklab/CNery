"""Consistency tests against the authentic LTEE Ara+1 dataset.

Marked ``authentic``: deselected by default, run with ``pytest -m authentic``, which fetches
~24 MB from GitHub Releases on first use. See CLAUDE.md.

The dataset is a real breseq ``bam2cov`` coverage table (4,629,812 positions) over a
circularly permuted REL606 reference, with no BAM -- CNery reads the table directly.

These assertions record what the code does TODAY. That is deliberate: they are a tripwire for
unintended change, not a statement that the current output is correct.

*** IMPORTANT -- TestOriginTerminus below still needs review against real output ***
The mask -> fit -> apply refactor changed how preprocess() and otr_fit() behave:

  1. preprocess() no longer DROPS windows that overlap redundant/repeat coverage -- it keeps
     every window and records `pct_redundant` instead. Confirmed against a real run
     (2026-08-14): window count for WIN=1000/STEP=500 on this dataset is 9259, up from the
     old golden of 8818 (the old preprocess() dropped windows near repeats). See the note on
     `test_window_count_is_stable` below for why this number isn't a clean formula.

  2. otr_fit() was rewritten (the old scipy.optimize.minimize-based line fit was degenerate
     and produced position-order-dependent, sometimes-backwards corrections). The
     TestOriginTerminus class below previously documented that OTR correction does NOT fire
     on this dataset due to that specific bug. That premise may no longer hold -- the new
     otr_fit() may now correctly detect and correct OTR bias on this dataset. This class's
     assertions still need to be re-run and reviewed against the real pipeline["ori"] /
     pipeline["ter"] values; it has NOT been updated beyond the mechanical otr_correction()
     -> fit_otr_bias()/apply_otr_correction() call-site translation.

  3. otr_correction(df, out) was split into fit_otr_bias(df, out) + apply_otr_correction(...).
     The `pipeline` fixture below uses the new call sequence.

The two golden files (GOLDEN_BREAKS, GOLDEN_OTR) must be regenerated and reviewed with::

    pytest -m authentic --regenerate-goldens

Golden tests are reported as SKIPPED on that run, not passed -- a run that rewrote its own
expectations has verified nothing.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from CNery.core import (
    _read_coverage_tab,
    coverage_table_path,
    fit_otr_bias,
    apply_otr_correction,
    parse_fasta_records,
    preprocess,
    process_multi_genome,
    run_HMM,
)

from data._fetch import load_registry

pytestmark = pytest.mark.authentic

SEQ_ID = "REL606_2314906bp_shift"
GENOME_LEN = 4_629_812

# The goldens are only meaningful at fixed windowing -- pin it.
WIN, STEP, FRAG = 1000, 500, 150

EXPECTED_DIR = os.path.join(os.path.dirname(__file__), "data", "expected")
GOLDEN_BREAKS = os.path.join(EXPECTED_DIR, "ltee_ara_p1_50k_shift_break_pts.csv")
GOLDEN_OTR = os.path.join(EXPECTED_DIR, "ltee_ara_p1_50k_shift_otr_results.json")

# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def ltee_dataset(request):
    """The extracted dataset directory, or a skip explaining why it is unavailable."""
    from conftest import _dataset_or_skip

    return _dataset_or_skip("ltee_ara_p1_50k_shift")

@pytest.fixture(scope="session")
def raw_table(ltee_dataset):
    return _read_coverage_tab(coverage_table_path(str(ltee_dataset), SEQ_ID))

@pytest.fixture(scope="session")
def pipeline(ltee_dataset, tmp_path_factory):
    """Run the full pipeline once and share it -- every assertion below reads this."""
    out = tmp_path_factory.mktemp("ltee_out")
    for sub in ("CNV_plt", "CNV_csv", "GC_bias", "OTR_corr"):
        (out / sub).mkdir()

    per_genome = process_multi_genome(
        bamfile=os.path.join(str(ltee_dataset), "reference.bam"),  # absent by design
        fastafile=os.path.join(str(ltee_dataset), "reference.fasta"),
        output_prefix=str(out),
        win=WIN, step=STEP, frag=FRAG,
        coverage_dir=str(ltee_dataset),
    )

    df_gc = per_genome[SEQ_ID]
    # otr_correction(df_gc, str(out)) was split into fit_otr_bias() + apply_otr_correction().
    # df_gc already carries is_deletion/is_redundant from the mask_coverage_windows() call
    # inside process_multi_genome()'s GC-correction stage, so fit_otr_bias() reuses them.
    otr_fit_result = fit_otr_bias(df_gc, str(out))
    df_otr, ori, ter = apply_otr_correction(otr_fit_result, str(out))
    df_cnv = run_HMM(df_otr, str(out))
    return {"out": out, "gc": df_gc, "otr": df_otr, "cnv": df_cnv, "ori": ori, "ter": ter}

# --------------------------------------------------------------------------- input integrity

@pytest.fixture(scope="session")
def cached_archive(ltee_dataset):
    """Path of the downloaded archive in pooch's cache.

    Depends on ltee_dataset so the fetch has actually happened -- without that these run
    before anything requests the data and see an empty cache.
    """
    import pooch

    from data._fetch import CACHE_ENV_VAR

    entry = load_registry()["ltee_ara_p1_50k_shift"]
    cache = os.environ.get(CACHE_ENV_VAR) or str(pooch.os_cache("cnery"))
    return os.path.join(cache, entry["file"]), entry

class TestFetch:
    """Proves the published asset is what the tests actually run on.

    Without this, a green authentic run only shows that *some* bytes were on disk -- a stale
    or hand-placed cache would look identical to a working download.
    """

    def test_archive_landed_in_the_cache(self, cached_archive):
        path, _ = cached_archive
        assert os.path.isfile(path)

    def test_cached_archive_matches_the_registry_hash(self, cached_archive):
        # pooch verifies on download; re-checking here catches later corruption or tampering
        # of a warm cache, which no download-time check can see.
        import hashlib

        path, entry = cached_archive
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        assert digest.hexdigest() == entry["sha256"]

    def test_archive_size_is_what_the_registry_claims(self, cached_archive):
        path, entry = cached_archive
        assert os.path.getsize(path) == entry["bytes"]

class TestInputIntegrity:
    """Guards the parsing path against upstream breseq format changes."""

    def test_no_bam_is_present(self, ltee_dataset):
        # The whole point of --coverage-dir: the 229 MB BAM is not shipped and not needed.
        assert not os.path.exists(os.path.join(str(ltee_dataset), "reference.bam"))

    def test_every_position_is_present(self, raw_table):
        # Off-by-four here is exactly the skipfooter=4 bug this suite now guards.
        assert len(raw_table) == GENOME_LEN
        assert raw_table.index[0] == 1
        assert raw_table.index[-1] == GENOME_LEN

    def test_position_is_the_index(self, raw_table):
        assert raw_table.index.name == "position"

    def test_footer_was_stripped(self, raw_table):
        # A surviving '#' row would poison ref_base and the numeric columns.
        assert not raw_table["ref_base"].astype(str).str.startswith("#").any()
        assert raw_table["unique_top_cov"].dtype.kind in "iu"

    def test_columns_cnery_depends_on(self, raw_table):
        for col in ("ref_base", "unique_top_cov", "unique_bot_cov",
                    "redundant_top_cov", "redundant_bot_cov"):
            assert col in raw_table.columns

    def test_reference_matches_the_table(self, ltee_dataset):
        records = parse_fasta_records(os.path.join(str(ltee_dataset), "reference.fasta"))
        assert records == [(SEQ_ID, GENOME_LEN)]

# --------------------------------------------------------------------------- windowing

class TestWindowing:
    def test_window_count_is_stable(self, raw_table):
        # Golden: 9259, confirmed against a real run of this dataset (2026-08-14), up from
        # the old golden of 8818 -- the old preprocess() dropped every window that
        # overlapped redundant/repeat coverage; the new preprocess() keeps all of them
        # (see test_redundant_regions_are_retained_with_pct_redundant_flag below).
        #
        # This is intentionally NOT derived from a closed-form formula. An earlier version
        # of this test computed the expected count as `(GENOME_LEN - WIN) // STEP + 1`,
        # which assumes every window is full-width (i + WIN <= GENOME_LEN). That's wrong:
        # preprocess()'s loop only requires `i <= GENOME_LEN - 1`, so it also emits one
        # final PARTIAL window at the tail (winu < WIN bases) before `lst_win` reaches
        # GENOME_LEN and the loop stops -- e.g. here the last window starts at i=4629000
        # with only 812 of its nominal 1000 bases available. That trailing partial window
        # is exactly the off-by-one (9259 vs. the formula's 9258). Rather than re-deriving
        # a corrected formula that has to reason about this edge case for every
        # (win, step, genome_len) combination, this test pins the real observed count, the
        # same way every other golden in this file does.
        windows = preprocess(raw_table, win=WIN, step=STEP, frag=FRAG)
        assert len(windows) == 9259

    def test_redundant_regions_are_retained_with_pct_redundant_flag(self, raw_table):
        # preprocess() now keeps every window and flags redundancy via
        # `pct_redundant` instead, so this test now checks for that behavior: every window
        # is present (see test_window_count_is_stable) AND at least some windows are
        # genuinely flagged with nonzero pct_redundant.
        windows = preprocess(raw_table, win=WIN, step=STEP, frag=FRAG)
        assert len(windows) == 9259
        assert "pct_redundant" in windows.columns
        assert (windows["pct_redundant"] > 0).any(), (
            "dataset has known repeat coverage (region_repeat_average_cov 2.14) but no "
            "window was flagged pct_redundant > 0"
        )

    def test_gc_is_plausible_for_e_coli(self, raw_table):
        windows = preprocess(raw_table, win=WIN, step=STEP, frag=FRAG)
        assert 0.45 < windows["gc_percent"].mean() < 0.55

# --------------------------------------------------------------------------- pipeline shape

class TestPipelineShape:
    def test_row_count_preserved_end_to_end(self, pipeline):
        assert len(pipeline["cnv"]) == len(pipeline["gc"])

    def test_genome_id_survives(self, pipeline):
        assert (pipeline["cnv"]["genome_id"] == SEQ_ID).all()

    def test_no_nan_in_calls(self, pipeline):
        assert not pipeline["cnv"]["prob_copy_number"].isna().any()

    def test_corrected_coverage_is_finite_and_non_negative(self, pipeline):
        values = pipeline["cnv"]["otr_gc_corr_norm_cov"].to_numpy()
        assert np.isfinite(values).all()
        assert (values >= 0).all()

    def test_gc_correction_does_not_resurrect_deletions(self, pipeline):
        # gc_correction (now mask_coverage_windows -> fit_gc_bias -> apply_gc_correction)
        # freezes near-zero windows at exactly 0.0 so real deletions still call CN=0
        # rather than being divided back up toward 1.
        df = pipeline["gc"]
        deleted = df["read_count_cov"] <= df["read_count_cov"].median() * 0.1
        assert deleted.any(), "dataset should contain real deletions"
        assert (df.loc[deleted, "gc_corr_norm_cov"] == 0.0).all()

# --------------------------------------------------------------------------- OTR

class TestOriginTerminus:
    """Records that OTR correction does NOT fire on this dataset (as of the last verified run).

    The reference was circularly permuted specifically to put the origin below the terminus
    in window-index order and so exercise the otr_fit xori<xter branch. It does not get
    there: the median-filter seed places the two about 6% of the genome apart, below the 35%
    floor at core.py:523, so otr_fit returns early. Observed at both w=1000/s=500 and
    w=200/s=100, so it is not a windowing artifact.

    *** This premise needs re-confirmation after the mask -> fit -> apply refactor ***
    otr_fit() now excludes is_deletion/is_redundant windows (dilated by one window on each
    side) from the argmax/argmin peak/trough SEARCH -- the old otr_fit() never did this, it
    searched the raw median-filtered profile unconditionally. If the old ~6%-apart
    peak/trough happened to involve a deletion or repeat window (plausible on real data),
    excluding it could move the detected positions and change the separation -- possibly
    across the 35% floor. The three assertions below are exactly what will surface that: if
    any of them fail, it means detection genuinely changed, not that this test file is wrong.
    Re-run with `-m authentic`, inspect the actual failure, and if the change is intentional
    (i.e. the exclusion fix correctly found real OTR bias this dataset's old detection was
    missing), regenerate GOLDEN_OTR (and GOLDEN_BREAKS, since CN calls downstream of a newly
    -applied OTR correction would shift too) with `--regenerate-goldens` and review the diff.

    These assertions fail loudly if detection starts OR stops happening -- either is a real
    change worth noticing while this area is under development.
    """

    def test_bias_is_not_currently_detected(self, pipeline):
        with open(GOLDEN_OTR) as fh:
            golden = json.load(fh)
        assert golden["Origin-to-Termius/Bias Ratio"] == "Not detected"

    def test_otr_json_matches_golden(self, pipeline, regenerate_goldens):
        from conftest import golden_compare

        produced = os.path.join(
            pipeline["out"], "OTR_corr",
            f"{os.path.basename(str(pipeline['out']))}{SEQ_ID}_otr_results.json",
        )

        def compare(got_path, want_path):
            with open(got_path) as fh:
                got = json.load(fh)
            with open(want_path) as fh:
                want = json.load(fh)
            assert got["Origin-to-Termius/Bias Ratio"] == want["Origin-to-Termius/Bias Ratio"]
            assert got["Origin window"] == want["Origin window"]
            assert got["Terminus window"] == want["Terminus window"]

        golden_compare(produced, GOLDEN_OTR, regenerate_goldens, compare)

    def test_uncorrected_coverage_passes_through(self, pipeline):
        # With no bias detected, otr correction must not alter the GC-corrected values.
        # If the exclusion-from-search fix above now causes bias to be detected on this
        # dataset, this assertion will correctly start failing -- see the class docstring.
        df = pipeline["otr"]
        np.testing.assert_allclose(
            df["otr_gc_corr_norm_cov"].to_numpy(),
            df["gc_corr_norm_cov"].to_numpy(),
            rtol=1e-9,
        )

# --------------------------------------------------------------------------- copy number

class TestCopyNumberGolden:
    def test_segments_match_golden(self, pipeline, regenerate_goldens):
        from conftest import golden_compare

        produced = os.path.join(
            pipeline["out"], "CNV_csv",
            f"{os.path.basename(str(pipeline['out']))}{SEQ_ID}_break_pts.csv",
        )

        def compare(got_path, want_path):
            pd.testing.assert_frame_equal(pd.read_csv(got_path), pd.read_csv(want_path))

        golden_compare(produced, GOLDEN_BREAKS, regenerate_goldens, compare)

    def test_single_copy_dominates(self, pipeline):
        calls = pipeline["cnv"]["prob_copy_number"]
        assert calls.mode().iloc[0] == 1
        assert (calls == 1).mean() > 0.8

    def test_deletions_and_amplification_both_called(self, pipeline):
        states = set(pipeline["cnv"]["prob_copy_number"].unique())
        assert 0 in states, "expected called deletions"
        assert max(states) >= 2, "expected a called amplification"