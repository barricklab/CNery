import pytest
import numpy as np
import pandas as pd
from CNery.core import mask_coverage_windows, fit_gc_bias, apply_gc_correction


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
