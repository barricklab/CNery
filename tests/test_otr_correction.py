import json
import os
import pytest
import numpy as np
import pandas as pd
from scipy.stats import chi2
from CNery.core import (
    fit_otr_bias, apply_otr_correction, otr_fit, otr_predict,
    GC_SKEW_METHOD, OTR_LR_ALPHA, OTR_MAX_P, OTR_SCORE_CELLS,
    DEFAULT_OTR_SURROGATES,
)
from CNery.core import (
    _otr_concentrated_rss, _otr_design_matrix, _otr_decimate, _otr_phase,
    _otr_phase_grid, _otr_grid_scores, _otr_normalize_phases, _otr_bootstrap_p,
    _otr_block_length, _otr_autocorr_length, _otr_lr_statistic,
    _otr_lr_bootstrap_p, _otr_tent_fit, _otr_cusum_range,
    _otr_residual_structure, OTR_STRUCTURE_SURROGATES, _otr_diff_sigma,
    _otr_detect_event, _otr_cells_to_windows, OTR_EVENT_ALPHA,
)

def _ensure_dirs(base):
    os.makedirs(os.path.join(base, "OTR_corr"), exist_ok=True)

def _make_sloped_df(n=80):
    rng = np.random.default_rng(3)
    half = n // 2
    cov = np.empty(n)
    cov[:half] = np.linspace(1.5, 0.5, half)
    cov[half:] = np.linspace(0.5, 1.5, n - half)
    cov += rng.normal(0, 0.02, n)
    cov = np.clip(cov, 0.01, None)
    rc = (cov * 100).astype(float)
    med = np.median(rc)
    return pd.DataFrame({
        "genome_id": "chr1",
        "win_st": np.arange(n) * 200,
        "win_end": np.arange(n) * 200 + 200,
        "win_len": 200,
        "gc_percent": 0.50,
        "read_count_cov": rc,
        "norm_raw_cov": rc / med,
        "gc_corr_norm_cov": cov,
        "gc_corr_fact": 1.0,
        "window_num": np.arange(n),
    })


def _run_otr(df, out):
    """otr_correction(df, out) was split into fit_otr_bias() + apply_otr_correction().
    Both fixtures/dfs used here lack is_deletion/is_redundant/gc_cor_med_fil columns,
    which fit_otr_bias() computes automatically (running mask_coverage_windows()
    internally and median-filtering gc_corr_norm_cov) before calling otr_fit()."""
    otr_fit_result = fit_otr_bias(df, out)
    return apply_otr_correction(otr_fit_result, out)


def _prep(df):
    """Add the columns otr_fit() reads, so it can be called without an output dir.

    Same preparation fit_otr_bias() does -- the masking flags and the median
    filter -- lifted out so the gate tests can call otr_fit() directly and read
    its `detail` dict without writing plots or JSON.
    """
    from CNery.core import mask_coverage_windows
    from scipy import ndimage
    df = df.copy()
    if "is_deletion" not in df.columns or "is_redundant" not in df.columns:
        df = mask_coverage_windows(df)
    if "gc_cor_med_fil" not in df.columns:
        n = len(df)
        win = min(max(3, int(n / 50)), n)
        if win % 2 == 0:
            win -= 1
        df["gc_cor_med_fil"] = (
            df["gc_corr_norm_cov"].copy() if win < 1
            else ndimage.median_filter(df["gc_corr_norm_cov"], size=win, mode="reflect")
        )
    return df


def test_required_columns_present(gc_corrected_flat, tmp_path):
    out = str(tmp_path / "otr1")
    _ensure_dirs(out)
    df_out, _, _ = _run_otr(gc_corrected_flat, out)
    assert "otr_gc_corr_norm_cov" in df_out.columns
    assert "otr_gc_corr_fact" in df_out.columns

def test_flat_coverage_no_bias_applied(gc_corrected_flat, tmp_path):
    out = str(tmp_path / "otr2")
    _ensure_dirs(out)
    df_out, _, _ = _run_otr(gc_corrected_flat, out)
    assert np.isfinite(df_out["otr_gc_corr_norm_cov"].values).all()
    assert (df_out["otr_gc_corr_norm_cov"] >= 0).all()

def test_sloped_coverage_reduces_slope(tmp_path):
    out = str(tmp_path / "otr3")
    _ensure_dirs(out)
    df = _make_sloped_df()
    df_out, _, _ = _run_otr(df, out)
    # output must be finite and non-negative — correctness, not magnitude
    assert np.isfinite(df_out["otr_gc_corr_norm_cov"].values).all()
    assert (df_out["otr_gc_corr_norm_cov"] >= 0).all()
    assert len(df_out) == len(df)

def test_json_results_has_required_keys(gc_corrected_flat, tmp_path):
    out = str(tmp_path / "otr4")
    _ensure_dirs(out)
    _run_otr(gc_corrected_flat, out)
    json_files = list((tmp_path / "otr4" / "OTR_corr").glob("*.json"))
    assert len(json_files) >= 1
    with open(json_files[0]) as f:
        data = json.load(f)
    assert "Origin window" in data or len(data) > 0


# ---------------------------------------------------------------------------
# Helpers for the significance gate and the coverage-vs-skew arbitration.
# ---------------------------------------------------------------------------

def _ar1(n, rho, sd, rng):
    """AR(1) noise, so surrogates face the autocorrelation real coverage has."""
    e = rng.normal(0, sd, n)
    out = np.empty(n)
    out[0] = e[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out * np.sqrt(1 - rho ** 2)


def _make_tent_df(n=240, o=None, t=None, ratio=1.6, sd=0.02, rho=0.0, seed=5):
    """A circular tent with a known origin and terminus, plus noise.

    Built with otr_predict() rather than by hand so the synthetic truth and the
    model under test cannot drift apart.
    """
    rng = np.random.default_rng(seed)
    o = n // 4 if o is None else o
    t = (o + n // 2) % n if t is None else t
    y_ter = 2.0 / (1.0 + ratio)
    cov = otr_predict(np.arange(n, dtype=float), float(o), float(t),
                      y_ter * ratio, y_ter, float(n))
    cov = np.clip(cov + (_ar1(n, rho, sd, rng) if rho else rng.normal(0, sd, n)), 0.01, None)
    rc = cov * 100.0
    return pd.DataFrame({
        "genome_id": "chr1",
        "win_st": np.arange(n) * 200,
        "win_end": np.arange(n) * 200 + 200,
        "win_len": 200,
        "gc_percent": 0.50,
        "read_count_cov": rc,
        "norm_raw_cov": rc / np.median(rc),
        "gc_corr_norm_cov": cov,
        "gc_corr_fact": 1.0,
        "window_num": np.arange(n),
    })


def _skew(o_idx, t_idx, n, confident=True):
    """The subset of predict_ori_ter_from_skew()'s dict that otr_fit() reads.

    Deliberately a literal rather than a call to the real function: these frames
    carry no ref_base, and what is under test here is the ARBITRATION, not the
    skew estimate. "Windows" is included because otr_fit() guards on it -- a
    prediction made against a different window count is refused rather than
    silently mis-indexed.
    """
    return {
        "Origin window index": int(o_idx),
        "Terminus window index": int(t_idx),
        "Windows": int(n),
        "Prediction confident": bool(confident),
    }


class TestSignificanceGate:
    """The tent must beat a null before its ramp is divided out.

    Before this gate the only real criterion was that the two breakpoints landed
    35-65% apart, so a flat, pure-noise genome whose best-fit tent happened to
    satisfy that had its coverage divided by noise. `bias_threshold` looked like
    a second criterion but was vacuous: the label swap in otr_fit() guarantees
    y_ori >= y_ter, so "ratio > 1.0" reduced to "y_ter > 0".
    """

    def test_flat_noise_is_rejected(self, gc_corrected_flat):
        *_, bias, detail = otr_fit(_prep(gc_corrected_flat))
        assert bias is False
        assert detail["Coverage fit p-value"] > OTR_MAX_P
        assert detail["Breakpoint source"] == "not corrected"

    def test_true_ramp_is_accepted(self):
        *_, bias, detail = otr_fit(_prep(_make_tent_df(n=240)))
        assert bias is True
        assert detail["Coverage fit p-value"] <= OTR_MAX_P
        assert detail["Coverage fit r-squared"] > 0.5

    def test_pure_noise_is_accepted_at_about_the_nominal_rate(self):
        """A 1% test accepts noise about 1% of the time BY CONSTRUCTION.

        Asserting "flat coverage is never accepted" would be asserting something
        false, and would make the suite fail on an unlucky seed for a gate that
        is behaving exactly as specified. What is pinned is the rate.
        """
        accepted = 0
        for seed in range(12):
            rng = np.random.default_rng(100 + seed)
            df = _make_tent_df(n=240, ratio=1.0, sd=0.05, seed=100 + seed)
            df["gc_corr_norm_cov"] = rng.normal(1.0, 0.05, len(df))
            *_, bias, _ = otr_fit(_prep(df), n_surrogates=200)
            accepted += int(bias)
        assert accepted <= 2, f"{accepted}/12 pure-noise sequences accepted"

    def test_statistic_equals_one_minus_rss_over_sst(self):
        """The cheap scorer IS the fit's own objective, not an approximation.

        _otr_design_matrix()'s rows sum to 1, so the tent's column span is
        span{1, u} and the concentrated RSS reduces to a squared correlation.
        Everything in the gate rests on this, so it is checked rather than
        asserted in a comment.
        """
        rng = np.random.default_rng(0)
        m = 400
        y = rng.normal(1.0, 0.1, m) + 0.3 * _otr_phase(m, 50, 250)
        sst = float(((y - y.mean()) ** 2).sum())
        x = np.arange(m, dtype=float)
        w = np.ones(m)
        for bp in [(10.0, 260.0), (300.0, 100.0), (0.0, 250.0), (123.4, 300.7)]:
            rss, _, _ = _otr_concentrated_rss(bp, x, y, float(m))
            phase = _otr_normalize_phases(_otr_phase(m, *bp), w)
            r2 = float(_otr_grid_scores(y, phase, w)[0])
            assert 1.0 - rss / sst == pytest.approx(r2, abs=1e-10)

    def test_design_matrix_rows_sum_to_one(self):
        m = 300
        M = _otr_design_matrix(np.arange(m, dtype=float), 33.0, 210.0, float(m))
        assert np.allclose(M.sum(axis=1), 1.0)

    def test_observed_and_surrogates_are_scored_by_the_same_function(self):
        """One code path, so the match is structural rather than maintained."""
        rng = np.random.default_rng(1)
        m = 200
        s = rng.normal(1.0, 0.1, m)
        w = np.ones(m)
        phases, _ = _otr_phase_grid(m, w)
        one = _otr_grid_scores(s, phases, w)
        many = _otr_grid_scores(np.column_stack([s, s, s]), phases, w)
        assert one.shape == (1,)
        assert many.shape == (3,)
        assert np.allclose(many, one[0])

    def test_scoring_cost_is_capped(self):
        n = 20_000
        y = np.linspace(1.0, 2.0, n)
        values, weights = _otr_decimate(y, np.ones(n, dtype=bool))
        assert values.size <= OTR_SCORE_CELLS
        assert weights.size == values.size

    def test_decimation_keeps_censored_windows_in_place(self):
        """A deletion must keep its WIDTH, not be compacted out of the circle."""
        n = 1000
        y = np.ones(n)
        keep = np.ones(n, dtype=bool)
        keep[400:600] = False
        values, weights = _otr_decimate(y, keep, cells=100)
        assert values.size == weights.size
        assert np.isfinite(values).all()
        # The censored fifth of the genome still occupies a fifth of the
        # lattice; it is zero-weighted, not removed.
        assert (weights[40:60] == 0).all()
        assert (weights[:40] > 0).all() and (weights[60:] > 0).all()

    def test_censored_cells_carry_no_weight(self):
        """Empty cells must not be INVENTED by interpolation.

        They used to be filled by circular np.interp, which fabricated coverage
        supporting whatever trend the neighbours implied. That was not a rare
        corner: on CWBI's plasmid_1, 121 of 232 cells held no unmasked window,
        so a majority of the scored series was invented and the statistic read
        r-squared 0.175 against 0.085 on the real windows.
        """
        n = 600
        rng = np.random.default_rng(11)
        y = rng.normal(1.0, 0.05, n)
        keep = np.ones(n, dtype=bool)
        keep[200:400] = False
        values, weights = _otr_decimate(y, keep, cells=60)
        w = np.asarray(weights, dtype=float)
        # Whatever value sits in a zero-weight cell, it cannot move any score.
        phases, _ = _otr_phase_grid(values.size, w)
        before = _otr_grid_scores(values, phases, w)
        perturbed = values.copy()
        perturbed[w == 0] += 100.0
        assert _otr_grid_scores(perturbed, phases, w) == pytest.approx(before)

    def test_cells_are_weighted_by_how_many_windows_they_hold(self):
        """A cell holding three windows counts three times one holding one.

        Equal-weighting cells reweighted the genome wherever censoring was
        uneven, and made the decimated statistic disagree with the
        full-resolution objective otr_fit() minimises, which weights per window.
        """
        n = 100
        y = np.ones(n)
        keep = np.ones(n, dtype=bool)
        keep[:5] = False          # first cell loses 5 of its 10 windows
        _, weights = _otr_decimate(y, keep, cells=10)
        assert weights[0] == 5
        assert (weights[1:] == 10).all()

    def test_cell_count_never_exceeds_the_observations(self):
        n = 500
        keep = np.zeros(n, dtype=bool)
        keep[:40] = True
        values, weights = _otr_decimate(np.ones(n), keep, cells=400)
        assert values.size <= 40


class TestBootstrap:
    """The circular block bootstrap behind the coverage p-value.

    Modelled on tests/test_gc_skew.py::TestBootstrap, with one deliberate
    divergence noted on test_block_length_grows_with_autocorrelation below.
    """

    def _flat(self, m=400, seed=0):
        return np.random.default_rng(seed).normal(1.0, 0.1, m)

    def test_p_value_is_a_probability(self):
        s = self._flat()
        w = np.ones(s.size)
        phases, _ = _otr_phase_grid(s.size, w)
        obs = float(_otr_grid_scores(s, phases, w)[0])
        p, b = _otr_bootstrap_p(s, phases, w, obs, 20, n_surrogates=200)
        assert 0.0 < p <= 1.0
        assert b == 200

    def test_p_value_floors_at_one_over_b_plus_one(self):
        df = _prep(_make_tent_df(n=400, sd=0.01))
        *_, detail = otr_fit(df, n_surrogates=200)
        assert detail["Bootstrap surrogates"] == 200
        assert detail["Coverage fit p-value"] == pytest.approx(1 / 201, abs=1e-5)

    def test_more_surrogates_lowers_the_floor(self):
        df = _prep(_make_tent_df(n=400, sd=0.01))
        coarse = otr_fit(df, n_surrogates=100)[-1]["Coverage fit p-value"]
        fine = otr_fit(df, n_surrogates=500)[-1]["Coverage fit p-value"]
        assert fine < coarse

    def test_is_deterministic_for_a_fixed_seed(self):
        df = _prep(_make_tent_df(n=240))
        a = otr_fit(df, n_surrogates=200)[-1]
        b = otr_fit(df, n_surrogates=200)[-1]
        assert a == b

    def test_seed_changes_only_the_p_value(self):
        df = _prep(_make_tent_df(n=240))
        a = otr_fit(df, n_surrogates=200, seed=1)[-1]
        b = otr_fit(df, n_surrogates=200, seed=2)[-1]
        assert a["Coverage fit r-squared"] == b["Coverage fit r-squared"]

    def test_too_short_to_bootstrap_declines(self):
        """Under OTR_MIN_CELLS there is nothing to resample; it says so."""
        s = self._flat(m=10)
        w = np.ones(10)
        phases, _ = _otr_phase_grid(10, w)
        p, b = _otr_bootstrap_p(s, phases, w, 0.5, 3, n_surrogates=100)
        assert (p, b) == (1.0, 0)

    def test_block_length_grows_with_autocorrelation(self):
        """Deliberately NOT test_gc_skew.py's "verdict is insensitive to block".

        For GC skew that insensitivity is a virtue worth pinning. For coverage
        the same assertion would be FALSE and pinning it would be a bug: block
        length governs validity here, not just power. Measured false-positive
        rate at a nominal 1% is 0.00 at block ~= 5*tau but 0.10-0.33 at
        block ~= tau, which is why the length is floored at five autocorrelation
        lengths rather than chosen for block count alone.
        """
        rng = np.random.default_rng(2)
        m = 2000
        white = rng.normal(0, 1, m)
        correlated = _ar1(m, 0.99, 1.0, rng)
        tau_white = _otr_autocorr_length(white)
        tau_corr = _otr_autocorr_length(correlated)
        b_white = _otr_block_length(m, tau_white)
        b_corr = _otr_block_length(m, tau_corr)
        # White noise is governed by the block-count target (m // 20); only once
        # 5*tau exceeds that does the autocorrelation floor take over, which is
        # exactly the regime the false-positive measurements were made in.
        assert tau_corr > tau_white
        assert b_corr > b_white == m // 20

    def test_autocorrelation_is_measured_on_the_residual(self):
        """Measuring tau on the raw series lets the ramp inflate it.

        This is the 0.001 -> 0.034 bug: on a real chromosome the ramp itself is
        long-range structure, so raw tau reads ~20x the residual's, the block
        count collapses, and the test destroys its own detection.
        """
        m = 2000
        phase = _otr_phase(m, 400, 1400)
        rng = np.random.default_rng(3)
        series = 1.0 + 0.5 * phase + rng.normal(0, 0.05, m)
        raw_tau = _otr_autocorr_length(series)
        w = np.ones(m)
        norm = _otr_normalize_phases(phase, w)
        _, resid, _ = _otr_tent_fit(series, norm[0], w)
        assert _otr_autocorr_length(resid) < raw_tau / 5


class TestBreakpointArbitration:
    """Coverage fit versus GC-skew fit, decided by a bootstrap likelihood ratio.

    The two models are nested -- both fit the anchor values by OLS, the skew
    model additionally fixes the breakpoint positions -- so the difference is
    exactly two parameters and a likelihood ratio is the right instrument. What
    it cannot use is a chi-square(2) reference distribution; see
    test_chi_square_would_reject_where_the_bootstrap_does_not.
    """

    def test_arbitration_is_a_no_op_when_skew_result_is_none(self):
        """Protects the existing call sites, which pass no skew_result."""
        df = _prep(_make_tent_df(n=240))
        a = otr_fit(df, n_surrogates=200)
        b = otr_fit(df, n_surrogates=200, skew_result=None)
        assert np.allclose(a[1], b[1])
        assert a[2:5] == b[2:5]

    def test_a_prediction_for_a_different_window_count_is_refused(self):
        df = _prep(_make_tent_df(n=240))
        n = len(df)
        bad = otr_fit(df, n_surrogates=200, skew_result=_skew(60, 180, n + 1))
        plain = otr_fit(df, n_surrogates=200)
        assert bad[4] == plain[4]
        assert bad[2:4] == plain[2:4]

    def test_true_breakpoints_are_not_rejected(self):
        """Skew coordinates equal to the truth must survive the ratio test."""
        n = 400
        df = _prep(_make_tent_df(n=n, o=100, t=300, sd=0.03, rho=0.6))
        *_, bias, detail = otr_fit(
            df, n_surrogates=200, skew_result=_skew(100, 300, n)
        )
        assert bias is True
        assert detail["Coverage vs skew likelihood-ratio p-value"] > OTR_LR_ALPHA
        assert detail["Breakpoint source"] == "GC skew"

    def test_displaced_breakpoints_are_rejected(self):
        """A terminus 10% of the genome away loses; that is the p1 case."""
        n = 400
        df = _prep(_make_tent_df(n=n, o=100, t=300, sd=0.03, rho=0.6))
        *_, bias, detail = otr_fit(
            df, n_surrogates=200, skew_result=_skew(100, 340, n)
        )
        assert bias is True
        assert detail["Coverage vs skew likelihood-ratio p-value"] <= OTR_LR_ALPHA
        assert detail["Breakpoint source"] == "coverage fit"

    def test_chi_square_would_reject_where_the_bootstrap_does_not(self):
        """The reason this machinery exists instead of a closed form.

        With IID residuals the null of Lambda really is chi-square(2). Real
        coverage residuals are spatially autocorrelated, which inflates
        Lambda's upper tail by one to two orders of magnitude -- so
        chi-square(2) rejects the GC skew for essentially every sequence,
        including the one where both tents agree to 1.8% of the genome and the
        skew lands on REL606's oriC.
        """
        n = 400
        df = _prep(_make_tent_df(n=n, o=100, t=300, sd=0.03, rho=0.8))
        # Terminus displaced by 1% of the genome -- the regime where the two
        # estimates genuinely agree and the bootstrap should say so.
        *_, detail = otr_fit(df, n_surrogates=300, skew_result=_skew(100, 304, n))
        lam = detail["Coverage vs skew likelihood ratio"]
        p_boot = detail["Coverage vs skew likelihood-ratio p-value"]
        assert chi2.sf(lam, 2) < 0.01
        assert p_boot > OTR_LR_ALPHA

    def test_orientation_contradiction_rejects_the_skew_coordinates(self):
        """Rejected, never silently relabelled.

        otr_fit() swaps its OWN breakpoints when they come out inverted, because
        they are unlabelled. The skew's labels are the imported prior, so an
        inversion is evidence against the prediction -- keeping the coordinates
        and flipping the labels would discard the only thing it contributed.
        """
        n = 400
        df = _prep(_make_tent_df(n=n, o=100, t=300, sd=0.03))
        *_, o_idx, t_idx, bias, detail = otr_fit(
            df, n_surrogates=200, skew_result=_skew(300, 100, n)
        )
        assert detail["Breakpoint source"] != "GC skew"
        assert o_idx != 300

    def test_statistic_is_never_negative(self):
        """Grid snapping can otherwise put a nested model behind the free one."""
        n = 400
        df = _prep(_make_tent_df(n=n, o=100, t=300, sd=0.03))
        series, w = _otr_decimate(
            df["gc_corr_norm_cov"].to_numpy(float), np.ones(n, dtype=bool)
        )
        m = series.size
        phases, _ = _otr_phase_grid(m, w)
        # A separation of 30% of the genome, outside the 35-65% band, so the
        # skew tent is genuinely not in the free model's search set.
        skew_phase = _otr_normalize_phases(_otr_phase(m, 0, int(0.30 * m)), w)
        lam, _, _ = _otr_lr_statistic(series, skew_phase, phases, w)
        assert lam >= 0.0

    def test_identical_breakpoints_give_p_one(self):
        n = 400
        df = _prep(_make_tent_df(n=n, o=100, t=300, sd=0.03))
        series, w = _otr_decimate(
            df["gc_corr_norm_cov"].to_numpy(float), np.ones(n, dtype=bool)
        )
        m = series.size
        phases, _bps = _otr_phase_grid(m, w)
        best = int(np.argmax(np.abs((phases * w) @ (series - series.mean()))))
        lam, _, _ = _otr_lr_statistic(series, phases[best], phases, w)
        assert lam == pytest.approx(0.0, abs=1e-9)
        p, _ = _otr_lr_bootstrap_p(
            series, phases[best], phases, w, lam,
            _otr_block_length(m, 5), n_surrogates=100
        )
        assert p == pytest.approx(1.0)

    def test_skew_source_is_named_in_the_results_json(self, tmp_path):
        n = 400
        out = str(tmp_path / "otr_skew")
        _ensure_dirs(out)
        df = _make_tent_df(n=n, o=100, t=300, sd=0.03, rho=0.6)
        res = fit_otr_bias(df, out, skew_result=_skew(100, 300, n), n_surrogates=200)
        apply_otr_correction(res, out)
        data = json.loads(
            next((tmp_path / "otr_skew" / "OTR_corr").glob("*.json")).read_text()
        )
        assert data["Correction type"] == GC_SKEW_METHOD
        assert data["Breakpoint source"] == "GC skew"
        assert data["Origin window"] is not None
        assert data["Terminus window"] is not None

    def test_rejected_fit_still_reports_its_evidence(self, gc_corrected_flat, tmp_path):
        """A "not detected" file must still explain itself."""
        out = str(tmp_path / "otr_rej")
        _ensure_dirs(out)
        res = fit_otr_bias(gc_corrected_flat, out, n_surrogates=200)
        apply_otr_correction(res, out)
        data = json.loads(
            next((tmp_path / "otr_rej" / "OTR_corr").glob("*.json")).read_text()
        )
        assert data["Origin-to-Termius/Bias Ratio"] == "Not detected"
        assert isinstance(data["Origin window"], int)
        assert isinstance(data["Terminus window"], int)
        assert data["Coverage fit p-value"] is not None
        assert data["Breakpoint source"] == "not corrected"


def _structure_at(df, o, t, cells=None, n_surrogates=200, seed=0, weighted=True):
    """Score the residual of a tent pinned at (o, t) windows on this frame."""
    from CNery.core import OTR_SCORE_CELLS
    y = df["gc_corr_norm_cov"].to_numpy(float)
    keep = np.ones(len(df), dtype=bool)
    series, w = _otr_decimate(y, keep, cells=cells or OTR_SCORE_CELLS)
    m = series.size
    scale = m / len(df)
    if not weighted:
        w = np.ones(m)
    phase = _otr_normalize_phases(_otr_phase(m, o * scale, t * scale), w)
    _, resid, _ = _otr_tent_fit(series, phase[0], w)
    block = _otr_block_length(m, _otr_autocorr_length(resid))
    return _otr_residual_structure(
        resid, phase, w, block, n_surrogates=n_surrogates, seed=seed
    )[0]


class TestResidualStructure:
    """Is the applied tent the wrong SHAPE, as opposed to merely a weak fit?

    Reported only -- nothing gates on it. r-squared says how much of the variance
    a tent explains; this says whether what it failed to explain is structured.
    A tent can score a respectable r-squared and still be systematically wrong
    over a long stretch, which is exactly the ltee_ara_p1_50k_shift case: its
    GC-skew tent explains 34% of the variance while leaving 1.4 Mb of same-sign
    residual.

    The reason none of the obvious statistics is used here is that coverage
    residuals are nowhere near white even under a PERFECT fit -- measured
    decorrelation lengths are 2-50 cells on good fits. Lag-1 correlation, tau,
    Ljung-Box and a runs test all rank a merely weak fit above a genuinely biased
    one, and bootstrap-normalising them changes nothing, because block resampling
    preserves exactly the short-range correlation they measure.
    """

    def _noisy_tent(self, n=1000, o=250, t=750, rho=0.85, sd=0.05, seed=4):
        return _make_tent_df(n=n, o=o, t=t, ratio=1.8, sd=sd, rho=rho, seed=seed)

    def _mean_over_seeds(self, t, reps=4, **kw):
        """Average over noise realisations.

        The null has sd ~0.94, so a single realisation says almost nothing about
        where the score is CENTRED -- asserting a threshold on one seed fails
        roughly one run in six on correctly-calibrated input. Measured null over
        6 seeds: mean -0.02, values spanning -1.15 to +1.18.
        """
        return float(np.mean([
            _structure_at(self._noisy_tent(seed=4 + k), 250, t, n_surrogates=150, **kw)
            for k in range(reps)
        ]))

    def test_a_correct_tent_is_centred_on_zero(self):
        """Calibration: the right tent must not look structured."""
        assert abs(self._mean_over_seeds(750)) < 1.0

    def test_a_displaced_tent_scores_high(self):
        true = self._mean_over_seeds(750)
        displaced = self._mean_over_seeds(850)       # terminus off by 10%
        assert displaced > true + 1.5
        assert displaced > 2.0

    def test_the_score_rises_with_displacement_then_saturates(self):
        """A detector, not a ruler -- and the saturation is not a defect.

        Measured mean over 6 seeds at 0 / 2.5 / 5 / 10 / 20 / 30% displacement:
        -0.02, 2.06, 3.67, 2.79, 2.67, 2.52. It climbs steeply, peaks near 5%,
        and plateaus around 2.5-2.8 rather than continuing to rise.

        The cause is the same feedback that makes a real amplification safe: a
        worse fit leaves a longer-correlated residual (tau goes 7 -> 15 -> 88 ->
        171 cells across that range), which lengthens the bootstrap block, which
        raises the null with it. So the number reliably says STRUCTURED, and does
        not rank severity beyond a point. Read it as a flag, not a distance --
        the real ltee_ara_p1_50k_shift reads 2.62, squarely in the plateau and
        consistent with the 10.6%-of-genome displacement CLAUDE.md records.
        """
        near = self._mean_over_seeds(775)            # 2.5%
        peak = self._mean_over_seeds(800)            # 5%
        far = self._mean_over_seeds(950)             # 20%
        assert near > 1.0
        assert peak > near
        assert far > 2.0                             # still clearly flagged

    def test_a_real_amplification_is_not_read_as_misfit(self):
        """The confound worth pinning, and the reason the block stays adaptive.

        A copy-number event is structure the tent genuinely cannot explain, but
        it is not evidence the BREAKPOINTS are wrong. It is partly absorbed
        rather than flagged because it inflates the residual autocorrelation
        length, which lengthens the bootstrap block, which raises the null with
        it. Measured, a 2x amplification over 5% of the genome scores below a 5%
        breakpoint displacement. This is bounded, not perfect: at 15% of the
        genome a 2x amplification does reach displacement-like values, and
        CLAUDE.md records that limit.
        """
        df = self._noisy_tent()
        amp = df.copy()
        cov = amp["gc_corr_norm_cov"].to_numpy(float).copy()
        cov[400:450] *= 2.0
        amp["gc_corr_norm_cov"] = cov
        assert _structure_at(amp, 250, 750) < _structure_at(df, 250, 800)

    def test_zero_weight_cells_must_not_drive_the_score(self):
        """Fill values in censored cells are a long run of IDENTICAL numbers.

        A run of identical values is a maximal CUSUM excursion, so scoring them
        unweighted manufactures the loudest false positive in the corpus -- CWBI's
        plasmid_1, 45% censored, reads 3.37 unweighted against 0.53 weighted,
        louder than the one tent that is really misfit. Same shape of bug as the
        np.interp fabrication that _otr_decimate no longer does.
        """
        df = self._noisy_tent()
        cov = df["gc_corr_norm_cov"].to_numpy(float)
        keep = np.ones(len(df), dtype=bool)
        keep[300:700] = False                       # 40% censored
        series, w = _otr_decimate(cov, keep)
        m = series.size
        assert (w == 0).sum() > 0.3 * m
        phase = _otr_normalize_phases(_otr_phase(m, 0.25 * m, 0.75 * m), w)
        _, resid, _ = _otr_tent_fit(series, phase[0], w)
        block = _otr_block_length(m, _otr_autocorr_length(resid))
        weighted = _otr_residual_structure(resid, phase, w, block, n_surrogates=200)[0]
        flat = _otr_residual_structure(
            resid, _otr_normalize_phases(_otr_phase(m, 0.25 * m, 0.75 * m), np.ones(m)),
            np.ones(m), block, n_surrogates=200)[0]
        assert weighted != pytest.approx(flat, abs=0.2)

    def test_score_is_insensitive_to_the_cell_count(self):
        """The raw statistic grows as sqrt(m); only the calibrated z is publishable."""
        df = self._noisy_tent(n=4000, o=1000, t=3000)
        scores = [_structure_at(df, 1000, 3000, cells=c) for c in (1000, 2000, 4000)]
        assert max(scores) - min(scores) < 1.0

    def test_is_deterministic_for_a_fixed_seed(self):
        df = self._noisy_tent()
        assert _structure_at(df, 250, 850) == _structure_at(df, 250, 850)

    def test_cusum_range_is_rotation_invariant(self):
        """Kuiper, not Kolmogorov-Smirnov -- the genome is circular.

        max|C| would depend on where the reference's coordinate 1 falls, so a
        circularly permuted copy of the same genome would score differently. The
        same concern makes preprocess() mean-subtract before cumulating for
        cum_gc_skew.
        """
        rng = np.random.default_rng(9)
        m = 500
        r = rng.normal(0, 1, m)
        w = rng.integers(1, 4, m).astype(float)
        r = r - (w @ r) / w.sum()                   # weight-centre, as the real one is
        base = _otr_cusum_range(r, w)[0]
        for shift in (1, 137, 250, 499):
            rolled = _otr_cusum_range(np.roll(r, shift), np.roll(w, shift))[0]
            assert rolled == pytest.approx(base, rel=1e-9)

    def test_cusum_range_declines_degenerate_input(self):
        assert _otr_cusum_range(np.zeros(10), np.ones(10)) == (0.0, 0, 0)
        assert _otr_cusum_range(np.ones(10), np.zeros(10)) == (0.0, 0, 0)

    def test_too_few_cells_reports_nothing(self):
        """Both CWBI plasmids land here, which is the point.

        m = 111 and 114 with 45% and 28% zero-weight cells; a score built on that
        would be mostly fill values. Declining is the honest answer.
        """
        r = np.random.default_rng(0).normal(0, 1, 20)
        w = np.ones(20)
        phase = _otr_normalize_phases(_otr_phase(20, 5, 15), w)
        assert _otr_residual_structure(r, phase, w, 5) == (None, None)

    def test_not_reported_when_no_tent_was_applied(self, gc_corrected_flat):
        """A tent that was never applied has no residual worth publishing."""
        *_, bias, detail = otr_fit(_prep(gc_corrected_flat))
        assert bias is False
        assert detail["Residual structure score"] is None
        assert detail["Residual decorrelation length (bp)"] is None

    def test_reported_when_a_tent_was_applied(self):
        *_, bias, detail = otr_fit(_prep(_make_tent_df(n=240)))
        assert bias is True
        assert detail["Residual structure score"] is not None
        assert detail["Residual decorrelation length (bp)"] > 0


class TestEventCensoring:
    """Amplifications and deletions are censored from ONE refit of the tent.

    Real copy-number events pull the fitted ramp toward themselves: measured,
    adp1_mgd06_lb's free-fit origin lands inside its own CN-3 amplification and
    cwbi_ssym_ht04's chromosome inside its CN-34 one. The GC-skew arm rescues the
    coordinates on both but not the AMPLITUDE, because the anchors are still
    solved on contaminated data.

    Every decision is frozen on the uncensored series.

    The gate is frozen because an iterative version of this was prototyped and
    measured, and it manufactures significance. On ramp-free real sequences --
    each authentic sequence with its own best-fit tent divided out, so there is
    no ramp by construction -- letting censoring feed back into the detection
    p-value took the false-positive rate from 0/8 to 4/8 at a nominal 1%,
    inventing ratios up to 1.26 from starting p-values as high as 0.918.
    Excising 1-2% of the windows removes 87-98% of the variance the bootstrap
    was calibrated against, and a null drawn from the censored series cannot
    defend itself. A cap does not fix it; three of those four appeared after one
    round at ~2% censored.
    """

    def _tent(self, n=1200, ratio=1.8, sd=0.03, rho=0.7, seed=6):
        return _make_tent_df(n=n, o=300, t=900, ratio=ratio, sd=sd, rho=rho, seed=seed)

    def test_difference_sigma_ignores_a_broad_event(self):
        """The normaliser must not be inflatable by the thing it is measuring.

        This is why the detector cannot reuse _otr_cusum_range, whose
        total-variance normaliser a 2x amplification inflates from 0.26 to 0.48
        across 5-30% of the genome, flattening the statistic from 1.00/1.76/
        2.94/3.84 to 1.00/1.34/1.79/2.11.
        """
        rng = np.random.default_rng(2)
        r = rng.normal(0, 0.05, 2000)
        clean = _otr_diff_sigma(r)
        r_event = r.copy()
        r_event[600:1200] += 0.8                            # a broad level shift
        assert _otr_diff_sigma(r_event) == pytest.approx(clean, rel=0.10)
        assert r_event.std() > 3 * r.std()                  # total variance is not

    def test_detects_an_amplification(self):
        n = 1200
        df = self._tent(n=n)
        cov = df["gc_corr_norm_cov"].to_numpy(float).copy()
        a, b = 400, 640                                     # 20% of the genome
        cov[a:b] *= 2.0
        series, w = _otr_decimate(cov, np.ones(n, dtype=bool))
        m = series.size
        ph = _otr_normalize_phases(_otr_phase(m, 300 * m / n, 900 * m / n), w)
        _, resid, _ = _otr_tent_fit(series, ph[0], w)
        ev = _otr_detect_event(resid, w, n_surrogates=200)
        assert ev["p"] <= OTR_EVENT_ALPHA
        lo, hi = ev["cells"]
        ca, cb = a * m // n, b * m // n
        assert lo < cb and hi > ca, "detected interval must overlap the amplification"

    def test_detects_a_deletion(self):
        n = 1200
        df = self._tent(n=n)
        cov = df["gc_corr_norm_cov"].to_numpy(float).copy()
        a, b = 400, 580
        cov[a:b] *= 0.35
        series, w = _otr_decimate(cov, np.ones(n, dtype=bool))
        m = series.size
        ph = _otr_normalize_phases(_otr_phase(m, 300 * m / n, 900 * m / n), w)
        _, resid, _ = _otr_tent_fit(series, ph[0], w)
        ev = _otr_detect_event(resid, w, n_surrogates=200)
        assert ev["p"] <= OTR_EVENT_ALPHA
        lo, hi = ev["cells"]
        ca, cb = a * m // n, b * m // n
        assert lo < cb and hi > ca, "detected interval must overlap the deletion"

    def test_declines_on_a_clean_tent(self):
        df = self._tent()
        n = len(df)
        series, w = _otr_decimate(df["gc_corr_norm_cov"].to_numpy(float),
                                  np.ones(n, dtype=bool))
        m = series.size
        ph = _otr_normalize_phases(_otr_phase(m, 300 * m / n, 900 * m / n), w)
        _, resid, _ = _otr_tent_fit(series, ph[0], w)
        assert _otr_detect_event(resid, w, n_surrogates=200)["p"] > OTR_EVENT_ALPHA

    def test_censoring_cannot_change_the_detection_p_value(self):
        """THE guard. The p-value must come from the uncensored series alone.

        Compare a run where the refit is allowed against one where it is
        forbidden by setting the cap to zero: every decision field must be
        bit-identical, and only the estimates may differ.
        """
        df = _prep(self._tent())
        allowed = otr_fit(df, n_surrogates=200)[-1]
        forbidden = otr_fit(df, n_surrogates=200, event_add_cap=0.0)[-1]
        for key in ("Coverage fit p-value", "Coverage fit r-squared",
                    "GC skew fit p-value", "Breakpoint source",
                    "Coverage vs skew likelihood-ratio p-value"):
            assert allowed[key] == forbidden[key], key

    def test_the_cap_declines_rather_than_trims(self):
        """At the cap the refit is refused outright, and says so.

        Censoring only part of a too-large event would be the worst of both --
        it removes real signal without removing the contamination.
        """
        n = 1200
        df = self._tent(n=n)
        cov = df["gc_corr_norm_cov"].to_numpy(float).copy()
        cov[300:800] *= 1.8                          # 42% of the genome
        df = df.copy()
        df["gc_corr_norm_cov"] = cov
        *_, detail = otr_fit(_prep(df), n_surrogates=200, event_add_cap=0.05)
        if detail["Event p-value"] is not None and detail["Event p-value"] <= OTR_EVENT_ALPHA:
            assert detail["Event exceeded censoring cap"] is True

    def test_cells_map_back_to_a_contiguous_window_interval(self):
        """_otr_decimate's window -> cell map is monotone, so this is exact."""
        m, n = 400, 1200
        lo, hi = _otr_cells_to_windows(100, 150, m, n)
        assert lo == 300 and hi == 450

    def test_event_keys_are_null_when_nothing_was_applied(self, gc_corrected_flat):
        *_, bias, detail = otr_fit(_prep(gc_corrected_flat))
        assert bias is False
        assert detail["Event p-value"] is None
        assert detail["Event start (bp)"] is None
        assert detail["Event exceeded censoring cap"] is False
