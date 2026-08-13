import pytest
import numpy as np
import pandas as pd
from CNery.core import gc_correction


def test_reg001_gc_correction_zero_windows_stay_zero():
    n = 60
    rc = np.full(n, 100.0)
    rc[20:30] = 0.0
    gc = np.linspace(0.40, 0.60, n)
    med = np.median(rc[rc > 0])
    df = pd.DataFrame({
        "read_count_cov": rc,
        "norm_raw_cov": rc / med,
        "gc_percent": gc,
    })
    out = gc_correction(df, zero_frac=0.05)
    assert (out.iloc[20:30]["gc_corr_norm_cov"] == 0.0).all()
    assert np.isfinite(out["gc_corr_norm_cov"].values).all()


def test_reg005_gc_correction_factor_never_negative():
    n = 60
    rng = np.random.default_rng(55)
    rc = rng.normal(100, 5, n).clip(1.0)
    gc = np.linspace(0.35, 0.65, n)
    med = np.median(rc)
    df = pd.DataFrame({
        "read_count_cov": rc,
        "norm_raw_cov": rc / med,
        "gc_percent": gc,
    })
    out = gc_correction(df, zero_frac=0.05)
    assert (out["gc_corr_norm_cov"] >= 0).all()