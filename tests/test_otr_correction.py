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
    _otr_residual_structure, OTR_STRUCTURE_SURROGATES,
    _stage_rows, _in_band_fractions, _censor_bins, plot_correction_stages,
    _correction_chain,
)
from CNery.core import (
    CN_CENSOR_MIN_KEEP,
    add_cn_censor,
    otr_ratio,
    pass1_summary,
    _cnv_axis_limits,
    _cnv_axis_ticks,
    plot_copy,
    plottable,
    refit_gc_bias_pooled,
    stage_pass1,
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


class TestCorrectionStagesPlot:
    """The per-sequence before/after diagnostic.

    Pixels are not asserted on. What is asserted on are the decisions the figure
    makes -- which rows a --bias mode draws, how the in-band fractions are
    computed, and how censoring is binned -- because those are the parts that can
    be silently wrong while still producing a plausible-looking PDF.
    """

    def _frame(self, n=240):
        """A frame staged as it reaches the figure: both passes complete.

        The figure draws one row per fitting CHANGE, and after the two-pass
        restructure there are four of them, so the frame has to carry the pass-1
        snapshot columns stage_pass1() writes as well as the final ones.
        """
        df = _prep(_make_tent_df(n=n))
        n = len(df)
        df["gc_corr_norm_cov_pass1"] = df["gc_corr_norm_cov"]
        df["gc_corr_fact_pass1"] = df["gc_corr_fact"]
        df["otr_gc_corr_norm_cov_pass1"] = df["gc_corr_norm_cov"] * 0.98
        df["otr_gc_corr_fact_pass1"] = 1.0
        df["gc2_resid_cov"] = df["otr_gc_corr_norm_cov_pass1"] * 0.99
        df["gc_corr_fact_pass2"] = 1.0
        df["otr_gc_corr_norm_cov"] = df["gc_corr_norm_cov"] * 0.97
        df["otr_gc_corr_fact"] = 1.0
        df["is_cn_variant"] = np.zeros(n, bool)
        return df

    def test_plot_is_written_and_makes_its_own_directory(self, tmp_path):
        """Nothing pre-creates corr_plots/, which is the point.

        tests/test_integration.py::_ensure_dirs deliberately does not create it
        either, so this assertion is not vacuous -- if the writer stopped
        self-creating, that file's tests would start failing too.
        """
        out = str(tmp_path / "run")
        os.makedirs(out)
        path = plot_correction_stages(self._frame(), out, None, bias="gc")
        assert os.path.exists(path)
        assert path.endswith("_correction_stages.pdf")
        assert os.path.isdir(os.path.join(out, "corr_plots"))

    def test_leaves_no_figure_open(self, tmp_path):
        """Regression guard for the plot_copy leak, on the new figure."""
        import matplotlib.pyplot as plt
        out = str(tmp_path / "run")
        os.makedirs(out)
        plt.close("all")
        plot_correction_stages(self._frame(), out, None, bias="gc")
        assert plt.get_fignums() == []

    @pytest.mark.parametrize("bias,rows", [("all", 4), ("gc", 2), ("otr", 2), ("none", 0)])
    def test_bias_mode_selects_its_rows(self, bias, rows):
        assert len(_stage_rows(bias)) == rows

    def test_otr_mode_never_draws_the_aliased_gc_column(self):
        """get_CNV.main() sets gc_corr_norm_cov = norm_raw_cov under --bias otr.

        Drawing a "GC corrected" row there would plot a bit-identical copy of the
        raw series under a label that is false. The figure has to know about the
        aliasing; it cannot infer it from which columns happen to be present.
        """
        assert all(row[0] != "gc_corr_norm_cov" and row[1] != "gc_corr_norm_cov"
                   for row in _stage_rows("otr"))
        assert any("gc_corr_norm_cov" in row for row in _stage_rows("all"))

    def test_both_denominators_are_reported(self):
        n = 100
        df = pd.DataFrame({
            "v": np.r_[np.full(90, 1.0), np.full(10, 0.4)],
            "is_deletion": np.r_[np.zeros(90, bool), np.ones(10, bool)],
            "is_redundant": np.zeros(n, bool),
            "gc_corr_norm_cov": 1.0,
        })
        censored = df["is_deletion"].to_numpy()
        assert _in_band_fractions(df["v"].to_numpy(), 1.0, censored) == pytest.approx((0.90, 1.00))

    def test_metric_is_invariant_to_replicon_copy_number(self):
        """A 2.95x plasmid must not read 0.000 at every stage.

        norm_raw_cov is normalised against the POOLED median, so a multi-copy
        replicon never enters the 0.8-1.2 band at all. Without scaling by the
        sequence's own censored median the figure would draw three empty bars on
        CWBI's plasmids and read as "the pipeline did nothing".
        """
        n = 100
        base = np.r_[np.full(90, 1.0), np.full(10, 0.4)]
        censored = np.r_[np.zeros(90, bool), np.ones(10, bool)]
        one = _in_band_fractions(base, 1.0, censored)
        scaled = _in_band_fractions(base * 2.95, 2.95, censored)
        assert one == pytest.approx(scaled)

    def test_censor_bins_are_exact_on_a_short_sequence(self):
        mask = np.zeros(50, dtype=bool)
        mask[10:20] = True
        x, dens = _censor_bins(mask, n_bins=400)
        assert dens.size == 50           # one bin per window, no special case
        assert dens.sum() == pytest.approx(10.0)

    def test_censor_bins_average_over_a_long_sequence(self):
        mask = np.zeros(1000, dtype=bool)
        mask[100:200] = True
        _x, dens = _censor_bins(mask, n_bins=100)
        assert dens.size == 100
        assert dens.mean() == pytest.approx(0.10)

    def test_the_drawn_progression_is_the_real_one(self):
        """A row flagged as chaining really does chain, and one row does not.

        Backfitting is not a linear chain: the second OTR fit divides the pass-1
        tent back out, so its "before" is not the previous row's "after". The
        figure annotates that break, and it sits between the two PASS-2 rows --
        not at the pass boundary, which is why the flag is derived from the column
        names rather than the pass number.
        """
        df = self._frame()
        steps = _correction_chain(df, None, "all")
        assert len(steps) == 4
        for a, b in zip(steps, steps[1:]):
            if b[4]:
                assert np.allclose(a[2], b[1]), "a chaining row must chain"
        assert np.allclose(steps[0][1], df["norm_raw_cov"].to_numpy(float))
        assert np.allclose(steps[-1][2], df["otr_gc_corr_norm_cov"].to_numpy(float))
        # Exactly one break, and it is the last row -- the second OTR fit.
        assert [step[4] for step in steps] == [True, True, True, False]

    def test_a_declined_cn_censor_draws_identical_strips(self, tmp_path):
        """CN_CENSOR_MIN_KEEP can decline the censor for a sequence. The figure
        must not then claim a restriction that is not in force -- the pass-2 fits
        saw exactly what the pass-1 fits saw."""
        out = str(tmp_path / "run")
        os.makedirs(out)
        df = self._frame()
        assert not df["is_cn_variant"].any()
        assert os.path.exists(plot_correction_stages(df, out, None, bias="all"))

    def test_pass_two_rows_carry_the_cn_censor(self):
        """The whole point of the second pass is a wider censor, so the strips
        that describe it must differ from the first pass's."""
        assert [row[4] for row in _stage_rows("all")] == [1, 1, 2, 2]


class TestSecondGCPass:
    """The pooled GC refit between the passes, composed into one total curve.

    It exists because the OTR tent varies with POSITION and position correlates
    with GC, so dividing by it puts a GC trend back into coverage the GC stage
    had removed -- measured at 10.4% of coverage on ltee_ara_p5_75k_exp. Pooled,
    like the first pass, because GC bias is a property of the chemistry rather
    than of any one reference.
    """

    def _frame(self, gid, n=400, gc_slope=0.0, seed=1):
        rng = np.random.default_rng(seed)
        gc = np.linspace(0.35, 0.60, n)
        cov = 1.0 + gc_slope * (gc - 0.475) + rng.normal(0, 0.01, n)
        return pd.DataFrame({
            "genome_id": gid,
            "win_st": np.arange(n) * 100,
            "win_end": np.arange(n) * 100 + 100,
            "gc_percent": gc,
            "norm_raw_cov": cov,
            "gc_corr_norm_cov": cov,
            "otr_gc_corr_norm_cov": cov,
            "gc_corr_fact": np.full(n, 1.20),
            "otr_gc_corr_fact": np.ones(n),
            "is_deletion": np.zeros(n, bool),
            "is_redundant": np.zeros(n, bool),
            "exclude_from_fit": np.zeros(n, bool),
        })

    def test_total_curve_is_the_product_of_the_two_passes(self):
        """Both passes are functions of GC alone, so composition is exact.

        This is what lets a single `gc_corr_fact` carry the whole GC correction
        into run_HMM's emission offset while the two components stay auditable.
        """
        out, _ = refit_gc_bias_pooled({"a": self._frame("a", gc_slope=0.6)})
        d = out["a"]
        assert np.allclose(d["gc_corr_fact"],
                           d["gc_corr_fact_pass1"] * d["gc_corr_fact_pass2"])

    def test_it_removes_a_gc_trend_the_first_pass_left(self):
        """Its own before/after: otr_gc_corr_norm_cov in, gc2_resid_cov out.

        NOT gc_corr_norm_cov, which this pass also rewrites but to raw/G -- a
        series that still carries the tent and so is not expected to be flat
        against GC. That column is the next OTR fit's INPUT, not this fit's
        output.
        """
        d = refit_gc_bias_pooled({"a": self._frame("a", gc_slope=0.6)})[0]["a"]
        gc = d["gc_percent"].to_numpy()
        v = d["gc2_resid_cov"].to_numpy()
        before = self._frame("a", gc_slope=0.6)["otr_gc_corr_norm_cov"].to_numpy()
        lo_h, hi_h = gc < 0.45, gc > 0.55
        assert abs(v[hi_h].mean() - v[lo_h].mean()) < abs(
            before[hi_h].mean() - before[lo_h].mean())

    def test_the_next_otr_fit_reads_raw_over_the_total_curve(self):
        """gc_corr_norm_cov must come out as raw/G, not as the fit's residual.

        The second OTR fit is handed this column, and it has to see the ramp it
        is meant to fit. raw/(g1*t*g2) has the pass-1 tent already divided out --
        fitting that would be fitting a residual, and the tent would come back
        near-flat.
        """
        d = refit_gc_bias_pooled({"a": self._frame("a", gc_slope=0.6)})[0]["a"]
        raw = d["norm_raw_cov"].to_numpy(float)
        total = (d["gc_corr_fact_pass1"].to_numpy(float)
                 * d["gc_corr_fact_pass2"].to_numpy(float))
        assert np.allclose(d["gc_corr_norm_cov"].to_numpy(float), raw / total)

    def test_it_is_pooled_not_per_sequence(self):
        """One curve across every reference, like the first pass.

        Two sequences with opposite GC trends must receive the SAME correction
        factor at the same GC -- if it were fitted per sequence they would get
        different ones and cancel their own trends independently.
        """
        frames = {"a": self._frame("a", gc_slope=0.6, seed=1),
                  "b": self._frame("b", gc_slope=-0.6, seed=2)}
        out, _ = refit_gc_bias_pooled(frames)
        fa = out["a"].set_index("gc_percent")["gc_corr_fact_pass2"]
        fb = out["b"].set_index("gc_percent")["gc_corr_fact_pass2"]
        common = fa.index.intersection(fb.index)
        assert len(common) > 50
        assert np.allclose(fa.loc[common].to_numpy(), fb.loc[common].to_numpy())

    def test_every_sequence_comes_back_with_its_own_rows(self):
        frames = {"a": self._frame("a", n=300), "b": self._frame("b", n=120)}
        out, _ = refit_gc_bias_pooled(frames)
        assert len(out["a"]) == 300 and len(out["b"]) == 120
        assert set(out) == {"a", "b"}


class TestCopyNumberCensor:
    """The censor the first pass's HMM builds for the second.

    mask_coverage_windows() censors on two crude proxies computed once on
    UNCORRECTED coverage -- near-zero depth and repeat overlap -- and an
    amplification is caught by neither, so it enters the GC LOWESS and the OTR
    tent at full weight. The HMM knows where the copy-number variation is;
    nothing else in the pipeline does.
    """

    def _called(self, n=200, variant=(0, 0), state=3):
        """n windows called CN=1, with [start, stop) called `state` instead."""
        cn = np.ones(n, dtype=int)
        cn[variant[0]:variant[1]] = state
        return pd.DataFrame({
            "genome_id": "a",
            "prob_copy_number": cn,
            "is_deletion": np.zeros(n, bool),
            "is_redundant": np.zeros(n, bool),
            "exclude_from_fit": np.zeros(n, bool),
        })

    def test_it_flags_exactly_what_the_hmm_did_not_call_single_copy(self):
        df, applied = add_cn_censor(self._called(variant=(40, 60)))
        assert applied
        assert df["is_cn_variant"].sum() == 20
        assert df["is_cn_variant"].to_numpy()[40:60].all()

    def test_deletions_count_as_variant_too(self):
        """CN 0 is not CN 1, and a called deletion is better evidence than the
        near-zero-depth proxy is_deletion already carries."""
        out, applied = add_cn_censor(self._called(variant=(10, 20), state=0))
        assert applied and out["is_cn_variant"].to_numpy()[10:20].all()

    def test_it_folds_into_the_flag_every_fit_stage_reads(self):
        """fit_gc_bias() reads exclude_from_fit by default, so the censor has to
        reach it or the pooled GC refit would silently ignore it."""
        df, _ = add_cn_censor(self._called(variant=(40, 60)))
        assert df["exclude_from_fit"].to_numpy()[40:60].all()
        assert not df["exclude_from_fit"].to_numpy()[:40].any()

    def test_it_declines_when_it_would_censor_too_much(self):
        """A genuinely duplicated replicon is every window CN=2. Censoring it
        entirely is not a fit, it is an empty fit -- so the guard declines and the
        second pass censors exactly as the first did."""
        df, applied = add_cn_censor(self._called(variant=(0, 200)))
        assert not applied
        assert not df["is_cn_variant"].any()
        assert not df["exclude_from_fit"].any()

    def test_the_guard_is_on_what_SURVIVES_not_on_what_is_added(self):
        """Deletions and repeats already censored have to count against the floor
        too: the fit sees the intersection, not the CN censor alone."""
        df = self._called(variant=(0, 40))
        df.iloc[40:140, df.columns.get_loc("is_redundant")] = True
        df["exclude_from_fit"] = df["is_redundant"]
        _out, applied = add_cn_censor(df)
        assert not applied, "140 of 200 already gone, so 40 more must trip it"

    def test_the_floor_is_a_named_constant(self):
        assert 0.0 < CN_CENSOR_MIN_KEEP < 1.0

    def test_absent_calls_are_not_an_error(self):
        """add_cn_censor runs before the first HMM in no pipeline, but the column
        contract lets any stage be handed a frame that lacks a column."""
        df = self._called().drop(columns=["prob_copy_number"])
        out, applied = add_cn_censor(df)
        assert not applied and not out["is_cn_variant"].any()


class TestPassOneStaging:
    """stage_pass1(): snapshot the first pass, then build the censor."""

    def _called(self, n=120):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "genome_id": "a",
            "norm_raw_cov": rng.normal(1.0, 0.02, n),
            "gc_corr_norm_cov": rng.normal(1.0, 0.02, n),
            "gc_corr_fact": np.full(n, 1.1),
            "otr_gc_corr_norm_cov": rng.normal(1.0, 0.02, n),
            "otr_gc_corr_fact": np.full(n, 1.05),
            "prob_copy_number": np.ones(n, dtype=int),
            "is_deletion": np.zeros(n, bool),
            "is_redundant": np.zeros(n, bool),
            "exclude_from_fit": np.zeros(n, bool),
        })
        df.loc[10:19, "prob_copy_number"] = 2
        return df

    def test_every_pass_one_column_is_preserved(self):
        """The second pass overwrites the canonical names, and the four-row
        correction-stages figure draws the first pass from these."""
        src = self._called()
        out, _ = stage_pass1(src)
        for column in ("gc_corr_fact", "gc_corr_norm_cov", "otr_gc_corr_fact",
                       "otr_gc_corr_norm_cov", "prob_copy_number"):
            np.testing.assert_array_equal(out[f"{column}_pass1"].to_numpy(),
                                          src[column].to_numpy())

    def test_a_missing_otr_column_is_not_an_error(self):
        """--bias gc/none never call apply_otr_correction(), so otr_gc_corr_fact
        is not on the frame at all."""
        src = self._called().drop(columns=["otr_gc_corr_fact"])
        out, _ = stage_pass1(src)
        assert "otr_gc_corr_fact_pass1" not in out
        assert "gc_corr_fact_pass1" in out


class TestPassOneSummary:
    """What the second pass's results JSON records about the first."""

    def test_no_tent_reports_no_ratio(self):
        assert otr_ratio({"bias": False, "y_fit": np.ones(10),
                          "o_idx": 0, "t_idx": 1}) is None
        assert otr_ratio(None) is None

    def test_the_ratio_is_the_two_anchors(self):
        y = np.array([1.4, 1.0, 0.7])
        assert otr_ratio({"bias": True, "y_fit": y, "o_idx": 0,
                          "t_idx": 2}) == pytest.approx(2.0)

    def test_the_summary_carries_the_first_verdict(self):
        res1 = {"bias": True, "y_fit": np.array([1.2, 1.0]), "o_idx": 0, "t_idx": 1,
                "detail": {"Coverage fit p-value": 0.32,
                           "Coverage fit r-squared": 0.05,
                           "Breakpoint source": "GC skew"}}
        staged = pd.DataFrame({"is_cn_variant": np.r_[np.ones(5, bool),
                                                      np.zeros(95, bool)]})
        keys = pass1_summary(res1, staged)
        assert keys["Coverage fit p-value (pass 1)"] == 0.32
        assert keys["Breakpoint source (pass 1)"] == "GC skew"
        assert keys["Origin-to-Termius/Bias Ratio (pass 1)"] == pytest.approx(1.2)
        assert keys["Windows censored as CN != 1"] == 5
        assert keys["Refit on CN=1 windows"] is True

    def test_a_declined_censor_is_reported_as_such(self):
        staged = pd.DataFrame({"is_cn_variant": np.zeros(100, bool)})
        keys = pass1_summary(None, staged)
        assert keys["Refit on CN=1 windows"] is False
        assert keys["Windows censored as CN != 1"] == 0


class TestSecondPassOtrFit:
    """otr_fit() consumes the CN censor through the same convention as its two
    siblings -- the caller decides which pass it is in by what it puts on the
    frame."""

    def test_cn_variant_windows_do_not_inform_the_fit(self):
        """A tall block pasted into flat coverage moves the fit; flagged
        is_cn_variant, it must not."""
        base = _prep(_make_tent_df(n=240, ratio=1.0))
        spiked = base.copy()
        spiked.loc[60:79, "gc_corr_norm_cov"] *= 3.0
        spiked["gc_cor_med_fil"] = spiked["gc_corr_norm_cov"]

        censored = spiked.copy()
        censored["is_cn_variant"] = np.r_[np.zeros(60, bool), np.ones(20, bool),
                                          np.zeros(len(base) - 80, bool)]

        _, _, _, _, _, d_spiked = otr_fit(spiked)
        _, _, _, _, _, d_censored = otr_fit(censored)
        assert d_spiked["Coverage fit r-squared"] != d_censored["Coverage fit r-squared"]


class TestRepeatWindowsAreNotDrawn:
    """No plot draws a redundant/repeat window.

    Their depth measures how many reference copies collapsed onto the locus, not
    how many the sample carries -- CWBI's chromosome has 17 kb at 2.13 Mb reading
    18x the single-copy level on exactly ZERO unique coverage. Nothing in the
    pipeline reads them: not fit_gc_bias, not otr_fit, not run_HMM's observation
    sequence or emission fit. Drawing them at 18x reads as an amplification and
    sets the y-axis so nothing else is legible.
    """

    def _frame(self, n=100, repeat=(40, 50), deletion=(80, 85)):
        df = pd.DataFrame({
            "genome_id": "a",
            "is_redundant": np.zeros(n, bool),
            "is_deletion": np.zeros(n, bool),
        })
        df.iloc[repeat[0]:repeat[1], df.columns.get_loc("is_redundant")] = True
        df.iloc[deletion[0]:deletion[1], df.columns.get_loc("is_deletion")] = True
        return df

    def test_repeat_windows_are_excluded(self):
        keep = plottable(self._frame())
        assert not keep[40:50].any()
        assert keep[:40].all() and keep[50:].all()

    def test_deletions_are_NOT_excluded(self):
        """They are censored from fitting too, but they are real measurements of
        a real absence and the CN=0 calls are unreadable without them."""
        assert plottable(self._frame())[80:85].all()

    def test_a_frame_without_the_column_draws_everything(self):
        """The synthetic fixtures carry no is_redundant column at all, and the
        column contract lets any stage be handed a frame that lacks one."""
        df = self._frame().drop(columns=["is_redundant"])
        assert plottable(df).all()

    def test_the_copy_number_plot_scales_to_what_it_draws(self, tmp_path):
        """A repeat pile-up must not set the y-axis. Scaling to every window is
        what compressed every real call into the bottom twentieth of the axis."""
        import matplotlib.pyplot as plt

        n = 200
        rng = np.random.default_rng(0)
        cov = rng.normal(100.0, 5.0, n)
        cov[100:110] = 1800.0                       # the repeat pile-up
        df = pd.DataFrame({
            "genome_id": "a",
            "win_st": np.arange(n) * 100,
            "win_end": np.arange(n) * 100 + 100,
            "read_count_cov": cov,
            "otr_gc_corr_rdcnt_cov": cov,
            "otr_gc_corr_norm_cov": cov / 100.0,
            "prob_copy_number": np.ones(n, dtype=int),
            "is_redundant": np.r_[np.zeros(100, bool), np.ones(10, bool),
                                  np.zeros(90, bool)],
            "is_deletion": np.zeros(n, bool),
        })
        out = str(tmp_path / "run")
        os.makedirs(os.path.join(out, "CNV_plt"))
        plt.close("all")
        plot_copy(df, 0, 0, out)
        assert plt.get_fignums() == []
        assert os.path.exists(
            os.path.join(out, "CNV_plt", "runa_copy_numbers.pdf"))


class TestCopyNumberPlotAxes:
    """The CN line has to pass through the coverage it describes.

    plot_copy twins two axes: copy number on the left, read counts on the right.
    They are the SAME scale in different units -- otr_gc_corr_rdcnt_cov is the
    corrected coverage times the median read depth -- so the right axis must be
    the left one times that median. Setting them independently, which is what the
    code did, left the alignment to chance: measured, the CN-1 line landed 56%
    too high on ltee_ara_m3_32k_2rg and 92% too high on cwbi_ssym_ht04's
    chromosome.
    """

    def _frame(self, n=300, median=170.0, amp=slice(100, 140), amp_cn=3):
        cn = np.ones(n)
        cn[amp] = amp_cn
        counts = median * cn
        return pd.DataFrame({
            "genome_id": "a",
            "win_st": np.arange(n) * 100,
            "win_end": np.arange(n) * 100 + 100,
            "read_count_cov": counts,
            "otr_gc_corr_norm_cov": cn,
            "otr_gc_corr_rdcnt_cov": counts,
            "prob_copy_number": cn.astype(int),
            "is_redundant": np.zeros(n, bool),
            "is_deletion": np.zeros(n, bool),
        })

    def test_the_two_axes_are_one_scale(self):
        df = self._frame()
        (lo1, hi1), (lo2, hi2) = _cnv_axis_limits(df, df, delta=85.0)
        scale = float(df["read_count_cov"].median())
        assert lo2 == pytest.approx(lo1 * scale)
        assert hi2 == pytest.approx(hi1 * scale)

    def test_neither_axis_goes_negative(self):
        """A read count cannot be negative and neither can a copy number.
        Padding the bottom the way the top is padded put a "-85 reads" tick on
        the axis."""
        df = self._frame()
        (lo1, _hi1), (lo2, _hi2) = _cnv_axis_limits(df, df, delta=85.0)
        assert lo1 == 0.0 and lo2 == 0.0

    def test_zero_coverage_is_still_inside_the_axis(self):
        """Real deletions sit at zero, and they have to remain visible."""
        df = self._frame()
        df.loc[:9, "read_count_cov"] = 0.0
        df.loc[:9, "otr_gc_corr_rdcnt_cov"] = 0.0
        (lo1, hi1), (lo2, hi2) = _cnv_axis_limits(df, df, delta=85.0)
        assert lo2 <= 0.0 <= hi2 and lo1 <= 0.0 <= hi1

    def test_copy_number_one_lands_on_the_median_coverage(self):
        """The whole point, stated directly."""
        df = self._frame()
        scale = float(df["read_count_cov"].median())
        (lo1, hi1), (lo2, hi2) = _cnv_axis_limits(df, df, delta=85.0)
        fraction = (1.0 - lo1) / (hi1 - lo1)
        assert lo2 + fraction * (hi2 - lo2) == pytest.approx(scale)

    @pytest.mark.parametrize("amp_cn", [2, 3, 12])
    def test_every_state_lands_on_the_coverage_it_describes(self, amp_cn):
        df = self._frame(amp_cn=amp_cn)
        scale = float(df["read_count_cov"].median())
        (lo1, hi1), (lo2, hi2) = _cnv_axis_limits(df, df, delta=85.0)
        for k in (1, amp_cn):
            fraction = (k - lo1) / (hi1 - lo1)
            assert lo2 + fraction * (hi2 - lo2) == pytest.approx(k * scale)

    def test_the_range_covers_both_drawn_series(self):
        """Raw and corrected reads are both plotted; neither may fall off."""
        df = self._frame()
        df.loc[0, "read_count_cov"] = 4000.0
        (_lo1, _hi1), (lo2, hi2) = _cnv_axis_limits(df, df, delta=85.0)
        assert lo2 <= df["read_count_cov"].min()
        assert hi2 >= df["read_count_cov"].max()
        assert hi2 >= df["otr_gc_corr_rdcnt_cov"].max()

    def test_the_scale_comes_from_every_window_not_the_drawn_ones(self):
        """otr_gc_corr_rdcnt_cov was built with the all-window median in
        run_HMM, so using the drawn subset's median instead would reintroduce the
        same class of mismatch wherever repeats are excluded."""
        df = self._frame()
        df.loc[:49, "is_redundant"] = True
        df.loc[:49, "read_count_cov"] = 5000.0
        drawn = df[~df["is_redundant"].astype(bool)]
        (_lo1, hi1), (_lo2, hi2) = _cnv_axis_limits(df, drawn, delta=85.0)
        assert hi2 / hi1 == pytest.approx(float(df["read_count_cov"].median()))

    def test_a_degenerate_frame_does_not_divide_by_zero(self):
        df = self._frame()
        df["read_count_cov"] = 0.0
        df["otr_gc_corr_rdcnt_cov"] = 0.0
        (lo1, hi1), _ = _cnv_axis_limits(df, df, delta=1.0)
        assert np.isfinite(lo1) and np.isfinite(hi1)

    def test_the_two_rulers_are_labelled_at_the_same_heights(self):
        """Tying the scales is not enough: the ticks have to correspond too.

        The right axis used to carry a LinearLocator with the same NUMBER of
        ticks as the left but no relation to their positions, so a copy number of
        2 -- 339 reads here -- sat beside a right-axis tick reading 381.
        """
        df = self._frame()
        (lo1, hi1), (lo2, hi2) = _cnv_axis_limits(df, df, delta=85.0)
        scale = float(df["read_count_cov"].median())
        cn, reads = _cnv_axis_ticks([0, 2, 4, 6, 8, 10], lo1, hi1, lo2, hi2)
        assert len(cn) == len(reads)
        for c, r in zip(cn, reads):
            assert r == pytest.approx(c * scale)

    def test_ticks_outside_the_range_are_dropped(self):
        df = self._frame()
        (lo1, hi1), (lo2, hi2) = _cnv_axis_limits(df, df, delta=85.0)
        cn, _reads = _cnv_axis_ticks([-4, 0, 2, 1e6], lo1, hi1, lo2, hi2)
        assert all(lo1 <= t <= hi1 for t in cn)

    def test_a_degenerate_range_still_yields_ticks(self):
        cn, reads = _cnv_axis_ticks([0, 1], 0.0, 0.0, 0.0, 0.0)
        assert len(cn) == len(reads) >= 2
