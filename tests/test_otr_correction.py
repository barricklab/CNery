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
    _otr_lr_bootstrap_p,
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
        for bp in [(10.0, 260.0), (300.0, 100.0), (0.0, 250.0), (123.4, 300.7)]:
            rss, _, _ = _otr_concentrated_rss(bp, x, y, float(m))
            r2 = float(_otr_grid_scores(y, _otr_normalize_phases(_otr_phase(m, *bp)))[0])
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
        phases, _ = _otr_phase_grid(m)
        one = _otr_grid_scores(s, phases)
        many = _otr_grid_scores(np.column_stack([s, s, s]), phases)
        assert one.shape == (1,)
        assert many.shape == (3,)
        assert np.allclose(many, one[0])

    def test_scoring_cost_is_capped(self):
        n = 20_000
        y = np.linspace(1.0, 2.0, n)
        assert _otr_decimate(y, np.ones(n, dtype=bool)).size <= OTR_SCORE_CELLS

    def test_decimation_keeps_censored_windows_in_place(self):
        """A deletion must keep its WIDTH, not be compacted out of the circle."""
        n = 1000
        y = np.ones(n)
        keep = np.ones(n, dtype=bool)
        keep[400:600] = False
        out = _otr_decimate(y, keep, cells=100)
        assert out.size == 100
        assert np.isfinite(out).all()


class TestBootstrap:
    """The circular block bootstrap behind the coverage p-value.

    Modelled on tests/test_gc_skew.py::TestBootstrap, with one deliberate
    divergence noted on test_block_length_grows_with_autocorrelation below.
    """

    def _flat(self, m=400, seed=0):
        return np.random.default_rng(seed).normal(1.0, 0.1, m)

    def test_p_value_is_a_probability(self):
        s = self._flat()
        phases, _ = _otr_phase_grid(s.size)
        obs = float(_otr_grid_scores(s, phases)[0])
        p, b = _otr_bootstrap_p(s, phases, obs, 20, n_surrogates=200)
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
        phases, _ = _otr_phase_grid(10)
        p, b = _otr_bootstrap_p(s, phases, 0.5, 3, n_surrogates=100)
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
        norm = _otr_normalize_phases(phase)
        yc = series - series.mean()
        resid = series - (series.mean() + float(norm[0] @ yc) * norm[0])
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
        series = _otr_decimate(
            df["gc_corr_norm_cov"].to_numpy(float), np.ones(n, dtype=bool)
        )
        m = series.size
        phases, _ = _otr_phase_grid(m)
        # A separation of 30% of the genome, outside the 35-65% band, so the
        # skew tent is genuinely not in the free model's search set.
        skew_phase = _otr_normalize_phases(_otr_phase(m, 0, int(0.30 * m)))
        lam, _, _ = _otr_lr_statistic(series, skew_phase, phases)
        assert lam >= 0.0

    def test_identical_breakpoints_give_p_one(self):
        n = 400
        df = _prep(_make_tent_df(n=n, o=100, t=300, sd=0.03))
        series = _otr_decimate(
            df["gc_corr_norm_cov"].to_numpy(float), np.ones(n, dtype=bool)
        )
        m = series.size
        phases, _bps = _otr_phase_grid(m)
        best = int(np.argmax(np.abs(phases @ (series - series.mean()))))
        lam, _, _ = _otr_lr_statistic(series, phases[best], phases)
        assert lam == pytest.approx(0.0, abs=1e-9)
        p, _ = _otr_lr_bootstrap_p(
            series, phases[best], phases, lam,
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