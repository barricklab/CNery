import pytest
import numpy as np
import pandas as pd


def _make_windowed_df(n=80, del_start=None, del_end=None,
                      amp_start=None, amp_end=None, median_cov=100.0):
    rng = np.random.default_rng(7)
    rc = rng.normal(median_cov, 5.0, n).clip(1).astype(float)
    if del_start is not None:
        rc[del_start:del_end] = 0.0
    if amp_start is not None:
        rc[amp_start:amp_end] = median_cov * 2.0
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


@pytest.fixture
def windowed_flat():
    return _make_windowed_df()


@pytest.fixture
def windowed_with_deletion():
    return _make_windowed_df(del_start=30, del_end=50)


@pytest.fixture
def windowed_with_amplification():
    return _make_windowed_df(amp_start=30, amp_end=45)


@pytest.fixture
def gc_corrected_flat(windowed_flat):
    df = windowed_flat.copy()
    df["gc_corr_norm_cov"] = df["norm_raw_cov"].copy()
    df["gc_corr_fact"] = np.ones(len(df))
    return df


@pytest.fixture
def otr_corrected_flat(gc_corrected_flat):
    df = gc_corrected_flat.copy()
    df["otr_gc_corr_norm_cov"] = df["gc_corr_norm_cov"].copy()
    df["otr_gc_corr_fact"] = np.ones(len(df))
    return df


@pytest.fixture
def single_fasta(tmp_path):
    fa = tmp_path / "single.fasta"
    fa.write_text(">chr1\n" + "ACGT" * 1000 + "\n")
    return str(fa)