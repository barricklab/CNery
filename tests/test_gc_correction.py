import pytest
import numpy as np
import pandas as pd
from CNery.core import gc_correction


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