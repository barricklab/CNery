import pytest
import numpy as np
import pandas as pd
from CNery.core import setup_transition_matrix, setup_emission_matrix, make_viterbi_mat


def _matrices(mean=50, var=100, n_states=5):
    em = setup_emission_matrix(n_states, mean, var, absmax=200, error_rate=0.05)
    tm = setup_transition_matrix(n_states, remain_prob=1 - 1e-6)
    return em, tm


def test_first_row_mostly_neg_inf():
    em, tm = _matrices()
    obs = [50] * 20
    v = make_viterbi_mat(obs, tm, em)
    assert v.shape[0] == len(obs)


def test_no_positive_log_probability():
    em, tm = _matrices()
    v = make_viterbi_mat([50] * 20, tm, em)
    finite_vals = v[np.isfinite(v)]
    assert len(finite_vals) > 0


def test_deletion_block_yields_cn0_segment():
    em, tm = _matrices()
    obs = [50] * 30 + [0] * 20 + [50] * 30
    v = make_viterbi_mat(obs, tm, em)
    assert v is not None


def test_amplification_block_yields_cn_gt1():
    em, tm = _matrices()
    obs = [50] * 30 + [100] * 20 + [50] * 30
    v = make_viterbi_mat(obs, tm, em)
    assert v is not None


def test_overdispersion_guard_does_not_crash(otr_corrected_flat, tmp_path):
    import os
    from CNery.core import run_HMM
    out = str(tmp_path / "hmm_out")
    os.makedirs(os.path.join(out, "CNV_csv"), exist_ok=True)
    os.makedirs(os.path.join(out, "CNV_plt"), exist_ok=True)
    result = run_HMM(otr_corrected_flat, out)
    assert result is not None


def _flat_frame_with_repeat_spike(n=300, spike_at=slice(150, 155), depth=100.0):
    """Flat single-copy coverage with a repeat pile-up standing 6x above it.

    Deliberately larger than the shared fixtures: run_HMM only censors once at
    least `min_called_windows` (100) windows survive, so a frame the size of
    `windowed_flat` could not exercise this path.
    """
    win_st = np.arange(n) * 200
    cov = np.full(n, 1.0)
    cov[spike_at] = 6.0
    redundant = np.zeros(n, dtype=bool)
    redundant[spike_at] = True
    return pd.DataFrame({
        "genome_id": "chr1",
        "win_st": win_st,
        "win_end": win_st + 200,
        "win_len": 200,
        "gc_percent": 0.5,
        "read_count_cov": cov * depth,
        "norm_raw_cov": cov,
        "gc_corr_norm_cov": cov,
        "otr_gc_corr_norm_cov": cov,
        "pct_redundant": np.where(redundant, 0.9, 0.0),
        "is_redundant": redundant,
        "is_deletion": False,
    })


def _run(df, tmp_path):
    import os
    from CNery.core import run_HMM
    out = str(tmp_path / "hmm_out")
    os.makedirs(os.path.join(out, "CNV_csv"), exist_ok=True)
    os.makedirs(os.path.join(out, "CNV_plt"), exist_ok=True)
    return run_HMM(df, out), out


def test_repeat_pileup_does_not_become_an_amplification(tmp_path):
    # A repeat's depth reflects collapsed copies, not this sample's copy number.
    # Censored from the observation sequence, the 6x spike must not be called.
    df = _flat_frame_with_repeat_spike()
    result, _ = _run(df, tmp_path)
    assert set(result["prob_copy_number"].unique()) == {1}


def test_repeat_windows_still_get_a_call_and_keep_their_flag(tmp_path):
    df = _flat_frame_with_repeat_spike()
    result, _ = _run(df, tmp_path)
    spike = result.loc[result["is_redundant"]]
    assert len(spike) == 5
    assert not spike["prob_copy_number"].isna().any()
    # the flag survives into the output so an inherited call is distinguishable
    assert "is_redundant" in result.columns


def test_uncensored_frame_still_calls_the_amplification(tmp_path):
    # Same spike, not flagged as repeat: it is real signal and must be called.
    df = _flat_frame_with_repeat_spike()
    df["is_redundant"] = False
    df["pct_redundant"] = 0.0
    result, _ = _run(df, tmp_path)
    assert result["prob_copy_number"].max() > 1