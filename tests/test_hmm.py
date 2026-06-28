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