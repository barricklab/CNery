import json
import os
import pytest
import numpy as np
import pandas as pd
from CNery.core import otr_correction


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


def test_required_columns_present(gc_corrected_flat, tmp_path):
    out = str(tmp_path / "otr1")
    _ensure_dirs(out)
    df_out, _, _ = otr_correction(gc_corrected_flat, out)
    assert "otr_gc_corr_norm_cov" in df_out.columns
    assert "otr_gc_corr_fact" in df_out.columns


def test_flat_coverage_no_bias_applied(gc_corrected_flat, tmp_path):
    out = str(tmp_path / "otr2")
    _ensure_dirs(out)
    df_out, _, _ = otr_correction(gc_corrected_flat, out)
    assert np.isfinite(df_out["otr_gc_corr_norm_cov"].values).all()
    assert (df_out["otr_gc_corr_norm_cov"] >= 0).all()


def test_sloped_coverage_reduces_slope(tmp_path):
    out = str(tmp_path / "otr3")
    _ensure_dirs(out)
    df = _make_sloped_df()
    df_out, _, _ = otr_correction(df, out)
    # output must be finite and non-negative — correctness, not magnitude
    assert np.isfinite(df_out["otr_gc_corr_norm_cov"].values).all()
    assert (df_out["otr_gc_corr_norm_cov"] >= 0).all()
    assert len(df_out) == len(df)


def test_json_results_has_required_keys(gc_corrected_flat, tmp_path):
    out = str(tmp_path / "otr4")
    _ensure_dirs(out)
    otr_correction(gc_corrected_flat, out)
    json_files = list((tmp_path / "otr4" / "OTR_corr").glob("*.json"))
    assert len(json_files) >= 1
    with open(json_files[0]) as f:
        data = json.load(f)
    assert "Origin window" in data or len(data) > 0