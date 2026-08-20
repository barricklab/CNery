import pytest
import numpy as np
import pandas as pd
from CNery.core import (
    DEFAULT_FRAG_SIZE,
    FRAG_SCAN_GRID,
    FRAG_SCAN_MIN_CANDIDATES,
    frag_candidates,
    frag_scan_target,
    gc_percent_for_frag,
    preprocess,
    reference_gc_flags,
    select_frag_size,
    _gc_curve_tau,
    apply_gc_correction,
    fit_gc_bias,
    mask_coverage_windows,
    refit_gc_bias_pooled,
)


def gc_correction(df, zero_frac=0.1):
    """gc_correction(df, zero_frac=...) was split into mask -> fit -> apply.
    Kept as a thin wrapper here so every test below is unchanged except for
    the import line, since the combined behavior (and output columns
    gc_corr_norm_cov/gc_corr_fact) is identical to the original function."""
    df_masked = mask_coverage_windows(df, zero_frac=zero_frac)
    gc_fit = fit_gc_bias(df_masked)
    return apply_gc_correction(df_masked, gc_fit)


def _gc_df(read_counts, gc_values):
    rc = np.asarray(read_counts, dtype=float)
    med = np.median(rc[rc > 0]) if np.any(rc > 0) else 1.0
    return pd.DataFrame({
        "read_count_cov": rc,
        "norm_raw_cov": rc / med,
        "gc_percent": np.asarray(gc_values, dtype=float),
    })

def test_true_zero_windows_stay_zero():
    rc = [100, 90, 0, 0, 0, 95, 105]
    gc = [0.50, 0.52, 0.49, 0.51, 0.48, 0.50, 0.53]
    out = gc_correction(_gc_df(rc, gc), zero_frac=0.05)
    assert out.iloc[2]["gc_corr_norm_cov"] == 0.0
    assert out.iloc[3]["gc_corr_norm_cov"] == 0.0
    assert out.iloc[4]["gc_corr_norm_cov"] == 0.0

def test_no_inf_in_output(windowed_flat):
    out = gc_correction(windowed_flat)
    assert np.isfinite(out["gc_corr_norm_cov"].values).all()

def test_no_negative_values():
    rng = np.random.default_rng(1)
    rc = rng.normal(100, 5, 60).clip(1.0)
    gc = np.linspace(0.35, 0.65, 60)
    out = gc_correction(_gc_df(rc, gc))
    assert (out["gc_corr_norm_cov"] >= 0).all()

def test_deletion_block_stays_zero(windowed_with_deletion):
    out = gc_correction(windowed_with_deletion, zero_frac=0.05)
    assert (out.iloc[30:50]["gc_corr_norm_cov"] == 0.0).all()

class TestGCSpan:
    """GC% is measured over max(frag, win) bases, centred on the window.

    GC bias acts at the scale of the sequenced fragment, so `-f` sets the span
    and `-w` only takes over when it is the wider of the two. This used to be
    two branches, and the `frag <= win` one spanned `2 * win - frag` while its
    comment claimed it used the window length -- at -w 200 -f 150 it measured
    GC over 250 bases, neither the window nor the fragment.
    """

    @staticmethod
    def _observed_span(df, win, step, frag):
        from CNery.core import preprocess
        gc = preprocess(df.copy(), win=win, step=step, frag=frag)["gc_percent"].to_numpy()
        # GC over an N-base span is always an exact multiple of 1/N.
        for n in range(20, 2001):
            if np.allclose(gc * n, np.round(gc * n), atol=1e-9):
                return n
        return None

    @pytest.fixture
    def coverage(self):
        rng = np.random.default_rng(11)
        n = 6000
        bases = rng.choice(list("ACGT"), size=n)
        # The --total-only schema preprocess() normalizes from.
        return pd.DataFrame({
            "position": np.arange(1, n + 1),
            "ref_base": bases,
            "unique_cov": np.full(n, 50.0),
            "redundant_cov": np.zeros(n),
        })

    @pytest.mark.parametrize("win,frag", [(100, 400), (100, 150), (200, 150),
                                          (500, 150), (400, 400), (1000, 150)])
    def test_span_is_max_of_frag_and_win(self, coverage, win, frag):
        assert self._observed_span(coverage, win, win, frag) == max(frag, win)

    def test_padding_is_not_capped_by_the_reference_length(self):
        """A 300 bp contig still gets the full 400 bp fragment span.

        The wrap-around buffer used to be a fixed +/-25% of the genome, which
        is unrelated to the fragment size: too much on a chromosome and not
        enough whenever the fragment exceeds half the reference, where it
        silently sliced out of range.
        """
        rng = np.random.default_rng(12)
        short = pd.DataFrame({
            "position": np.arange(1, 301),
            "ref_base": rng.choice(list("ACGT"), size=300),
            "unique_cov": np.full(300, 50.0),
            "redundant_cov": np.zeros(300),
        })
        assert self._observed_span(short, win=100, step=100, frag=400) == 400

    def test_a_reference_shorter_than_the_fragment_does_not_crash(self):
        """The span then wraps over the whole replicon, which is the right
        answer: for a contig shorter than a fragment, the fragment's GC IS the
        replicon's GC."""
        from CNery.core import preprocess
        rng = np.random.default_rng(13)
        tiny = pd.DataFrame({
            "position": np.arange(1, 121),
            "ref_base": rng.choice(list("ACGT"), size=120),
            "unique_cov": np.full(120, 50.0),
            "redundant_cov": np.zeros(120),
        })
        out = preprocess(tiny, win=100, step=100, frag=400)
        assert len(out) == 1
        assert 0.0 < float(out["gc_percent"].iloc[0]) < 1.0


class TestCurveUncertainty:
    """How well the GC curve is determined, per GC.

    Measured by resampling the fit windows, never chosen. The obvious proxy --
    "few windows out here" -- is wrong for this smoother: LOWESS uses a
    NEAREST-NEIGHBOUR bandwidth (frac=0.3), so every fitted point averages the
    same ~0.3n windows and local support is constant by construction. What
    degrades at the tails is that the neighbourhood becomes ONE-SIDED, so the
    local linear fit extrapolates within its own window. Resampling sees that;
    no count- or percentile-based rule does.
    """

    def _data(self, n=4000, seed=0):
        """GC drawn from a normal, so the tails really are sparse."""
        rng = np.random.default_rng(seed)
        gc = rng.normal(0.5, 0.05, n)
        cov = 1.0 + 0.8 * (gc - 0.5) + rng.normal(0, 0.05, n)
        return gc, cov

    def test_tau_is_larger_at_the_tails_than_in_the_middle(self):
        gc, cov = self._data()
        grid = np.linspace(gc.min(), gc.max(), 60)
        tau = _gc_curve_tau(gc, cov, np.ones(gc.size, bool), grid,
                            n_surrogates=40, seed=0)
        lo, hi = np.percentile(gc, [2, 98])
        interior = (grid > lo) & (grid < hi)
        assert tau[~interior].mean() > 2.0 * tau[interior].mean()

    def test_it_is_a_relative_sd_so_scale_free(self):
        """The offset enters multiplicatively, so tau must not change when the
        coverage units do."""
        gc, cov = self._data()
        grid = np.linspace(gc.min(), gc.max(), 40)
        mask = np.ones(gc.size, bool)
        a = _gc_curve_tau(gc, cov, mask, grid, n_surrogates=30, seed=1)
        b = _gc_curve_tau(gc, cov * 1000.0, mask, grid, n_surrogates=30, seed=1)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)

    def test_it_is_seeded(self):
        gc, cov = self._data()
        grid = np.linspace(gc.min(), gc.max(), 30)
        mask = np.ones(gc.size, bool)
        np.testing.assert_array_equal(
            _gc_curve_tau(gc, cov, mask, grid, n_surrogates=20, seed=7),
            _gc_curve_tau(gc, cov, mask, grid, n_surrogates=20, seed=7))

    def test_degenerate_input_returns_zero_rather_than_failing(self):
        grid = np.linspace(0.4, 0.6, 20)
        tau = _gc_curve_tau(np.array([0.5, 0.5]), np.array([1.0, 1.0]),
                            np.ones(2, bool), grid, n_surrogates=5)
        np.testing.assert_array_equal(tau, np.zeros_like(grid))

    def test_the_fit_publishes_it_on_a_grid(self):
        rng = np.random.default_rng(0)
        n = 800
        gc = rng.normal(0.5, 0.05, n)
        df = pd.DataFrame({
            "genome_id": "a",
            "gc_percent": gc,
            "read_count_cov": rng.poisson(100, n).astype(float),
            "norm_raw_cov": rng.normal(1.0, 0.05, n),
        })
        df = mask_coverage_windows(df)
        fit = fit_gc_bias(df, tau_surrogates=20)
        assert fit["gc_grid"].shape == fit["gc_tau"].shape
        assert np.all(fit["gc_tau"] >= 0)
        out = apply_gc_correction(df, fit)
        assert "gc_corr_tau" in out
        assert np.all(np.isfinite(out["gc_corr_tau"].to_numpy()))


class TestPooledUncertaintyComposition:
    """G = g1 * g2, so log G = log g1 + log g2 and the RELATIVE variances add."""

    def _frame(self, gid, n=400, seed=1):
        rng = np.random.default_rng(seed)
        gc = np.linspace(0.35, 0.60, n)
        cov = 1.0 + rng.normal(0, 0.01, n)
        return pd.DataFrame({
            "genome_id": gid,
            "win_st": np.arange(n) * 100,
            "win_end": np.arange(n) * 100 + 100,
            "gc_percent": gc,
            "norm_raw_cov": cov,
            "gc_corr_norm_cov": cov,
            "otr_gc_corr_norm_cov": cov,
            "gc_corr_fact": np.full(n, 1.2),
            "gc_corr_tau": np.full(n, 0.03),
            "otr_gc_corr_fact": np.ones(n),
            "is_deletion": np.zeros(n, bool),
            "is_redundant": np.zeros(n, bool),
            "exclude_from_fit": np.zeros(n, bool),
        })

    def test_relative_variances_add_in_quadrature(self):
        out, _ = refit_gc_bias_pooled({"a": self._frame("a")})
        d = out["a"]
        np.testing.assert_allclose(
            d["gc_corr_tau"].to_numpy(),
            np.sqrt(d["gc_corr_tau_pass1"].to_numpy() ** 2
                    + d["gc_corr_tau_pass2"].to_numpy() ** 2),
            rtol=1e-12)

    def test_the_total_is_at_least_each_component(self):
        d = refit_gc_bias_pooled({"a": self._frame("a")})[0]["a"]
        assert np.all(d["gc_corr_tau"] >= d["gc_corr_tau_pass1"] - 1e-12)
        assert np.all(d["gc_corr_tau"] >= d["gc_corr_tau_pass2"] - 1e-12)


class TestFragmentCandidates:
    """Which fragment sizes are worth scoring at all."""

    def test_the_lower_bound_is_the_window_size(self):
        """preprocess() measures GC over max(frag, win), so every candidate at or
        below the window size is the SAME computation -- not a candidate, a
        duplicate. Offering them would let the scan "choose" between identical
        options and report a difference that is pure noise."""
        for win in (100, 250, 600):
            assert all(f >= win for f in frag_candidates(win))

    def test_candidates_come_from_the_published_grid(self):
        assert set(frag_candidates(100)) <= set(FRAG_SCAN_GRID)

    def test_the_default_is_on_the_grid(self):
        """Otherwise "did any candidate beat the default" compares the default
        against something the grid cannot express."""
        assert DEFAULT_FRAG_SIZE in FRAG_SCAN_GRID

    def test_the_grid_is_increasing_and_geometric_ish(self):
        grid = list(FRAG_SCAN_GRID)
        assert grid == sorted(set(grid))
        ratios = [b / a for a, b in zip(grid, grid[1:])]
        assert all(1.15 < r < 1.6 for r in ratios), ratios

    def test_a_large_window_collapses_the_grid(self):
        """At -w 2000 almost nothing is distinguishable, and the caller declines
        rather than pretending to choose."""
        assert len(frag_candidates(2000)) < FRAG_SCAN_MIN_CANDIDATES

    def test_the_cli_default_window_leaves_plenty(self):
        assert len(frag_candidates(100)) >= FRAG_SCAN_MIN_CANDIDATES


class TestGcPercentForFrag:
    """The scan recomputes gc_percent from prefix sums instead of re-running
    preprocess(). If the two ever disagree, every candidate is scored against a
    GC definition the pipeline would not actually use."""

    def _table(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        bases = rng.choice(list("ACGT"), n, p=[0.3, 0.2, 0.2, 0.3])
        return pd.DataFrame(
            {"ref_base": bases,
             "unique_cov": rng.poisson(60, n).astype(float),
             "redundant_cov": np.zeros(n)},
            index=np.arange(1, n + 1))

    @pytest.mark.parametrize("win,frag", [(100, 400), (100, 150), (100, 2000),
                                          (200, 600), (500, 150)])
    def test_it_reproduces_preprocess_exactly(self, win, frag):
        raw = self._table()
        df = preprocess(raw, win=win, step=win, frag=frag)
        flags = reference_gc_flags(raw)
        off = df["win_st"].to_numpy(dtype=np.int64) - int(df["win_st"].min())
        np.testing.assert_allclose(
            gc_percent_for_frag(flags, off, win, frag),
            df["gc_percent"].to_numpy(dtype=float), atol=1e-12)

    def test_a_fragment_below_the_window_is_the_window(self):
        """The clamp, from the other side: 150 and 400 are identical at -w 500."""
        raw = self._table()
        flags = reference_gc_flags(raw)
        off = np.arange(0, 3000, 500, dtype=np.int64)
        np.testing.assert_array_equal(
            gc_percent_for_frag(flags, off, 500, 150),
            gc_percent_for_frag(flags, off, 500, 400))

    def test_it_wraps_circularly(self):
        """A window at the very start draws its padding from the far end."""
        raw = self._table(n=1000)
        flags = reference_gc_flags(raw)
        got = gc_percent_for_frag(flags, np.array([0]), 100, 600)
        assert 0.0 <= float(got[0]) <= 1.0


class TestFragmentScanTarget:
    """What the scan is allowed to score."""

    def _frame(self, n=400):
        """A sequence sitting at 2x the pooled median, with a real ramp in its
        coverage and the tent that describes it."""
        rng = np.random.default_rng(0)
        tent = np.linspace(0.9, 1.1, n)
        return pd.DataFrame({
            "genome_id": "a",
            "norm_raw_cov": 2.0 * tent * rng.normal(1.0, 0.01, n),
            "otr_gc_corr_fact": tent,
            "exclude_from_fit": np.r_[np.ones(20, bool), np.zeros(n - 20, bool)],
            "is_cn_variant": np.r_[np.zeros(n - 30, bool), np.ones(30, bool)],
        })

    def test_censored_and_cn_variant_windows_are_dropped(self):
        _t, keep = frag_scan_target(self._frame())
        assert not keep[:20].any() and not keep[-30:].any()
        assert keep[20:-30].all()

    def test_it_is_normalised_to_this_sequences_own_level(self):
        """The scan pools every sequence, and norm_raw_cov is normalised against
        the POOLED median -- so CWBI's plasmids sit at 2.95x and 1.90x against a
        chromosome at 1.0. At a large fragment size a short plasmid's GC is
        nearly constant, which turns GC into a REPLICON LABEL predicting those
        levels. Measured, that alone made the scan pick the top of the grid.
        """
        target, keep = frag_scan_target(self._frame())
        assert np.median(target[keep]) == pytest.approx(1.0, abs=1e-9)

    def test_the_ramp_is_divided_out(self):
        """Otherwise a large-frag GC predicts coverage by proxying position --
        the replication ramp -- rather than by modelling fragment chemistry."""
        df = self._frame()
        target, keep = frag_scan_target(df)
        raw = df["norm_raw_cov"].to_numpy()
        assert np.std(target[keep]) < 0.5 * np.std(raw[keep] / np.median(raw[keep]))


class TestFragmentSelection:
    """Choosing the size, and declining to."""

    def _reference(self, n=30000, seed=1):
        rng = np.random.default_rng(seed)
        # Long-wavelength GC structure, so different fragment scales genuinely
        # see different things.
        x = np.arange(n)
        p_gc = 0.5 + 0.18 * np.sin(2 * np.pi * x / 1500)
        return (rng.random(n) < p_gc)

    def _windows(self, flags, win=100, true_frag=None, strength=0.0, seed=2):
        n = flags.size
        starts = np.arange(0, n - win + 1, win, dtype=np.int64)
        rng = np.random.default_rng(seed)
        cov = np.ones(starts.size)
        if true_frag is not None:
            gc = gc_percent_for_frag(flags, starts, win, true_frag)
            cov = 1.0 + strength * (gc - gc.mean()) / max(gc.std(), 1e-9)
        cov = cov * rng.normal(1.0, 0.02, starts.size)
        return pd.DataFrame({
            "genome_id": "a",
            "win_st": starts + 1,
            "norm_raw_cov": cov,
            "exclude_from_fit": np.zeros(starts.size, bool),
            "is_cn_variant": np.zeros(starts.size, bool),
        })

    def test_it_recovers_a_known_fragment_size(self):
        """GROUND TRUTH. Coverage is generated so its GC bias acts at exactly one
        fragment scale; the scan has to find it. Everything else here tests the
        machinery -- this tests whether the idea works."""
        flags = self._reference()
        for true_frag in (200, 800):
            df = self._windows(flags, true_frag=true_frag, strength=0.30)
            frag, detail = select_frag_size({"a": df}, {"a": flags}, 100,
                                            DEFAULT_FRAG_SIZE, repeats=3)
            assert detail["Fragment size scanned"]
            grid = frag_candidates(100)
            i_true = min(range(len(grid)), key=lambda k: abs(grid[k] - true_frag))
            i_got = grid.index(frag)
            assert abs(i_got - i_true) <= 1, (
                f"true {true_frag}, selected {frag}, grid {grid}")

    def test_it_keeps_the_default_when_gc_carries_nothing(self):
        """Pure noise: no candidate can beat any other, so the default stands."""
        flags = self._reference()
        df = self._windows(flags, true_frag=None)
        frag, detail = select_frag_size({"a": df}, {"a": flags}, 100,
                                        DEFAULT_FRAG_SIZE, repeats=3)
        assert frag == DEFAULT_FRAG_SIZE
        assert "default kept" in detail["Fragment size reason"]

    def test_it_declines_when_the_window_is_too_large(self):
        flags = self._reference()
        df = self._windows(flags, win=100)
        frag, detail = select_frag_size({"a": df}, {"a": flags}, 2000,
                                        DEFAULT_FRAG_SIZE, repeats=2)
        assert frag == DEFAULT_FRAG_SIZE
        assert detail["Fragment size scanned"] is False
        assert "candidate" in detail["Fragment size reason"]

    def test_the_report_carries_the_evidence(self):
        flags = self._reference()
        df = self._windows(flags, true_frag=250, strength=0.30)
        _frag, detail = select_frag_size({"a": df}, {"a": flags}, 100,
                                         DEFAULT_FRAG_SIZE, repeats=3)
        assert detail["Fragment size candidates"] == frag_candidates(100)
        assert len(detail["Fragment size picks per split"]) == 3
        assert detail["Fragment size improvement se"] is not None


class TestNothingToFit:
    """fit_gc_bias declines with an identity curve rather than raising.

    LOWESS on empty arrays returns a (0, 2) array and the np.interp that follows
    raises "array of sample points is empty". Three real inputs reach it: every
    window at zero coverage (all flagged is_deletion), every window overlapping a
    repeat (all flagged is_redundant), and a reference with no windows at all.

    The declined curve is flat 1.0 so that apply_gc_correction is an exact
    no-op. That matters beyond the corrected column: run_HMM composes its
    emission offsets from gc_corr_fact, so a fabricated curve would be applied to
    the copy-number model even where no bias could be measured.
    """

    def _frame(self, n=200, cov=0.0, redundant=0.0):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "genome_id": "a",
            "gc_percent": rng.normal(0.5, 0.05, n),
            "read_count_cov": np.full(n, cov),
            "norm_raw_cov": np.full(n, cov),
            "pct_redundant": np.full(n, redundant),
        })

    def test_all_zero_coverage_gives_an_identity_curve(self):
        df = mask_coverage_windows(self._frame())
        fit = fit_gc_bias(df)
        np.testing.assert_allclose(fit["fit_sorted"], 1.0)

    def test_all_repeat_windows_give_an_identity_curve(self):
        df = mask_coverage_windows(self._frame(cov=25.0, redundant=1.0))
        assert df["exclude_from_fit"].all()
        np.testing.assert_allclose(fit_gc_bias(df)["fit_sorted"], 1.0)

    def test_an_empty_frame_gives_an_identity_curve(self):
        df = mask_coverage_windows(self._frame(n=0))
        np.testing.assert_allclose(fit_gc_bias(df)["fit_sorted"], 1.0)

    def test_applying_it_changes_nothing(self):
        # The point of a flat 1.0: "no GC correction was applied" has to mean
        # the coverage is untouched and the factor the HMM reads is exactly one.
        df = mask_coverage_windows(self._frame(cov=25.0, redundant=1.0))
        out = apply_gc_correction(df, fit_gc_bias(df))
        np.testing.assert_allclose(out["gc_corr_fact"], 1.0)
        np.testing.assert_allclose(out["gc_corr_norm_cov"], out["norm_raw_cov"])
