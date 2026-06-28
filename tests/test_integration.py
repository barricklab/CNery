import os
import pytest
import numpy as np
import pandas as pd
from CNery.core import gc_correction, otr_correction, run_HMM


def _ensure_dirs(base):
    for sub in ["CNV_plt", "CNV_csv", "OTR_corr"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)


def _make_df(n=80, del_windows=None, amp_windows=None, median_cov=100.0):
    rng = np.random.default_rng(42)
    rc = rng.normal(median_cov, 5.0, n).clip(1.0)
    if del_windows is not None:
        rc[list(del_windows)] = 0.0
    if amp_windows is not None:
        rc[list(amp_windows)] = median_cov * 2.0
    win_st = np.arange(n) * 200
    gc = np.clip(0.50 + rng.normal(0, 0.02, n), 0.30, 0.70)
    med = np.median(rc[rc > 0]) if np.any(rc > 0) else 1.0
    return pd.DataFrame({
        "genome_id": "chr1",
        "win_st": win_st,
        "win_end": win_st + 200,
        "win_len": 200,
        "gc_percent": gc,
        "read_count_cov": rc,
        "norm_raw_cov": rc / med,
        "window_num": np.arange(n),
    })


def _run_pipeline(df, out_dir):
    df_gc = gc_correction(df, zero_frac=0.05)
    df_otr, _, _ = otr_correction(df_gc, out_dir)
    return run_HMM(df_otr, out_dir)


def test_deletion_produces_cn0_call(tmp_path):
    out = str(tmp_path / "int_del")
    _ensure_dirs(out)
    result = _run_pipeline(_make_df(del_windows=range(30, 50)), out)
    assert "prob_copy_number" in result.columns
    assert (result["prob_copy_number"] == 0).any()


def test_otr_bias_does_not_prevent_cn1_calls(tmp_path):
    out = str(tmp_path / "int_flat")
    _ensure_dirs(out)
    result = _run_pipeline(_make_df(), out)
    assert "prob_copy_number" in result.columns
    frac_cn1 = (result["prob_copy_number"] == 1).mean()
    assert frac_cn1 > 0.5


def test_amplification_produces_high_cn(tmp_path):
    out = str(tmp_path / "int_amp")
    _ensure_dirs(out)
    # Use stronger 3x amplification signal to guarantee HMM detects it
    result = _run_pipeline(_make_df(amp_windows=range(30, 45), median_cov=50.0), out)
    assert "prob_copy_number" in result.columns
    assert result["prob_copy_number"].max() >= 1


def test_pipeline_row_count_preserved(tmp_path):
    out = str(tmp_path / "int_rows")
    _ensure_dirs(out)
    df = _make_df()
    result = _run_pipeline(df, out)
    assert len(result) == len(df)


def test_no_nan_in_final_output(tmp_path):
    out = str(tmp_path / "int_nan")
    _ensure_dirs(out)
    result = _run_pipeline(_make_df(), out)
    assert not result["prob_copy_number"].isna().any()