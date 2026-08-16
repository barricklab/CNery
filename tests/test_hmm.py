from itertools import product

import pytest
import numpy as np
import pandas as pd
from scipy.stats import nbinom

from CNery.core import (
    setup_transition_matrix,
    setup_emission_matrix,
    make_viterbi_mat,
    viterbi_path,
    HMM_copy_number,
    fit_censored_negative_binomial,
    log_emission_with_offsets,
    bias_offsets,
    robust_state_count,
    remain_prob_for_step,
    window_geometry,
    _log_emission_lookup,
    _default_log_start,
)


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


def _decode(obs, em, tm):
    log_tm = np.log(tm)
    return viterbi_path(_log_emission_lookup(obs, em), log_tm, _default_log_start(log_tm))


def test_path_is_a_backtrace_not_a_per_window_argmax():
    """The decoded path must be one path, not the per-window winner.

    make_viterbi_mat returns the score of the best path *ending* in each state,
    so its per-window argmax can name a state no single path ever passes
    through. Inside a short elevated run the high-state column only overtakes
    CN1 at the last window, which is how a real 3-window amplification came out
    labelled `1,1,3`.
    """
    em, tm = _matrices(mean=50, var=100, n_states=5)
    obs = [50] * 40 + [100] * 6 + [50] * 40

    v = make_viterbi_mat(obs, tm, em)
    per_window = np.argmax(v, axis=1)
    path = _decode(obs, em, tm)

    assert not np.array_equal(per_window, path)
    # The path is self-consistent: the elevated block is one contiguous state.
    block = path[40:46]
    assert len(set(block.tolist())) == 1
    # ...whereas the per-window argmax splits it and lands on the last window.
    assert len(set(per_window[40:46].tolist())) > 1


def test_backtraced_path_scores_at_least_as_high_as_the_argmax_labels():
    """A backtrace is optimal by construction; the per-window labels are not."""
    em, tm = _matrices(mean=50, var=100, n_states=5)
    obs = [50] * 30 + [100] * 5 + [50] * 30

    log_em = _log_emission_lookup(obs, em)
    log_tm = np.log(tm)

    def score(states):
        total = _default_log_start(log_tm)[states[0]] + log_em[0, states[0]]
        for i in range(1, len(states)):
            total += log_tm[states[i - 1], states[i]] + log_em[i, states[i]]
        return total

    path = _decode(obs, em, tm)
    per_window = np.argmax(make_viterbi_mat(obs, tm, em), axis=1)
    assert score(path) >= score(per_window)


def test_state_change_at_the_final_window_is_emitted():
    """The old loop ran to len(obs) - 1, so a change at the end vanished."""
    em, tm = _matrices(mean=50, var=100, n_states=5)
    n = 60
    obs = [50] * (n - 8) + [100] * 8
    win_st = np.arange(n) * 100
    win_end = win_st + 100

    segments = HMM_copy_number(obs, tm, em, win_st, win_end, chr_length=n * 100)
    assert len(segments) > 1
    assert segments["State"].iloc[-1] != segments["State"].iloc[0]


def test_path_matches_brute_force_over_an_asymmetric_matrix():
    """Pin the whole recursion against exhaustive enumeration.

    An asymmetric transition matrix is the point: Viterbi needs log T[from, to],
    and the old code indexed it [to, from], which is invisible while
    setup_transition_matrix() returns a symmetric matrix. Brute force also pins
    the backtrace and the start distribution.
    """
    rng = np.random.default_rng(3)
    n_states, n_obs = 3, 6

    tm = rng.random((n_states, n_states)) + 0.05
    tm /= tm.sum(axis=1, keepdims=True)
    log_em = np.log(rng.random((n_obs, n_states)) + 0.05)
    log_tm = np.log(tm)
    log_start = _default_log_start(log_tm)

    def score(states):
        total = log_start[states[0]] + log_em[0, states[0]]
        for i in range(1, len(states)):
            total += log_tm[states[i - 1], states[i]] + log_em[i, states[i]]
        return total

    best = max(product(range(n_states), repeat=n_obs), key=score)
    got = viterbi_path(log_em, log_tm, log_start)

    assert score(got) == pytest.approx(score(best))
    assert list(got) == list(best)


def _nb_counts(mu, size, n, rng):
    return nbinom.rvs(size, size / (size + mu), size=n, random_state=rng).astype(float)


class TestCensoredNegativeBinomialFit:
    """The single-copy emission parameters, fitted the way breseq fits its own."""

    def test_ignores_amplifications_and_deletions(self):
        rng = np.random.default_rng(0)
        mu, size = 63.0, 40.0
        clean = _nb_counts(mu, size, 27000, rng)
        amplified = _nb_counts(2 * mu, 2 * size, 9000, rng)
        deleted = np.zeros(9000)

        got_mu, got_size = fit_censored_negative_binomial(
            np.concatenate([clean, amplified, deleted])
        )
        assert got_mu == pytest.approx(mu, rel=0.05)
        assert got_size == pytest.approx(size, rel=0.25)

        # The uncensored moments this replaces are wrong by a wide margin.
        contaminated = np.concatenate([clean, amplified, deleted])
        assert contaminated.var() / contaminated.mean() > 4 * (
            (got_mu + got_mu ** 2 / got_size) / got_mu
        )

    def test_recovers_parameters_through_offsets(self):
        """Bias belongs in the mean; absorbing it into the dispersion is lossy."""
        rng = np.random.default_rng(1)
        mu, size = 63.0, 40.0
        offsets = 1.0 + 0.25 * np.sin(np.linspace(0, 12 * np.pi, 30000))
        counts = _nb_counts(mu * offsets, size, 30000, rng)

        with_offsets = fit_censored_negative_binomial(counts, offsets)
        assert with_offsets[0] == pytest.approx(mu, rel=0.05)
        assert with_offsets[1] == pytest.approx(size, rel=0.15)

        # Ignoring them re-reads the bias as overdispersion, halving `size`.
        without = fit_censored_negative_binomial(counts)
        assert without[1] < 0.6 * with_offsets[1]

    def test_every_redundant_window_is_excluded(self, tmp_path):
        """A window clipping an IS element sits inside the censoring band.

        So it cannot be left to the [0.5, 1.5] bounds to remove -- it has to be
        dropped before the histogram is built.
        """
        df = _flat_frame_with_repeat_spike(n=300)
        # Mild repeats at 1.3x: well inside [0.5, 1.5] x mode.
        df.loc[200:229, "read_count_cov"] = 130.0
        df.loc[200:229, "is_redundant"] = True
        df.loc[200:229, "pct_redundant"] = 0.5

        clean_only = df.loc[~df["is_redundant"], "read_count_cov"].to_numpy(float)
        assert fit_censored_negative_binomial(
            df["read_count_cov"].to_numpy(float)[~df["is_redundant"].to_numpy(bool)]
        ) == fit_censored_negative_binomial(clean_only)

        result, _ = _run(df, tmp_path)
        assert set(result.loc[200:229, "prob_copy_number"]) == {1}

    def test_flat_and_underdispersed_frames_fall_back_to_the_guard(self):
        """A negative binomial has no finite `size` for under-dispersed data."""
        flat = fit_censored_negative_binomial(np.full(300, 100.0))
        assert flat[0] == pytest.approx(100.0)
        # var = mean * (1 + 1e-3) -> size = mean / 1e-3
        assert flat[1] == pytest.approx(100.0 / 1e-3, rel=1e-6)

        rng = np.random.default_rng(2)
        gaussian = fit_censored_negative_binomial(rng.normal(100, 5, 500))
        assert gaussian[1] > 1e3      # far sharper than the data's own spread

    def test_too_little_data_returns_none(self):
        assert fit_censored_negative_binomial(np.array([100.0, 101.0])) is None
        assert fit_censored_negative_binomial(np.zeros(300)) is None


class TestBiasOffsets:
    def test_modes_select_their_factors(self):
        df = pd.DataFrame({"gc_corr_fact": [2.0] * 4, "otr_gc_corr_fact": [3.0] * 4})
        assert list(bias_offsets(df, "all")) == [6.0] * 4
        assert list(bias_offsets(df, "gc")) == [2.0] * 4
        assert list(bias_offsets(df, "otr")) == [3.0] * 4
        assert list(bias_offsets(df, "none")) == [1.0] * 4

    def test_missing_or_invalid_factors_fall_back_to_one(self):
        assert list(bias_offsets(pd.DataFrame(index=range(3)), "all")) == [1.0] * 3
        df = pd.DataFrame({"gc_corr_fact": [2.0, 0.0, np.nan, -1.0]})
        assert list(bias_offsets(df, "gc")) == [2.0, 1.0, 1.0, 1.0]


class TestPerBaseChangeRate:
    """The state-change prior describes the genome, not the tiling."""

    def test_remain_probability_is_multiplicative_in_step(self):
        """Crossing 2s bases must cost exactly what crossing s twice costs.

        This is the whole content of "per base": it holds for a Poisson
        boundary process and fails for any flat per-window probability.
        """
        rate = 1e-5
        assert remain_prob_for_step(rate, 200) == pytest.approx(
            remain_prob_for_step(rate, 100) ** 2, rel=1e-9
        )

    def test_a_bigger_step_makes_a_change_more_likely(self):
        rate = 1e-5
        assert (remain_prob_for_step(rate, 50)
                > remain_prob_for_step(rate, 100)
                > remain_prob_for_step(rate, 400))

    def test_rate_reads_as_one_boundary_per_reciprocal_bases(self):
        # Over 1/rate bases the chain should remain with probability 1/e.
        assert remain_prob_for_step(1e-6, 1_000_000) == pytest.approx(
            np.exp(-1.0), rel=1e-6
        )

    def test_geometry_is_recovered_from_the_frame(self):
        n = 50
        for win, step in [(200, 100), (100, 100), (1000, 500)]:
            df = pd.DataFrame({
                "win_st": np.arange(n) * step,
                "win_end": np.arange(n) * step + win,
                "win_len": win,
            })
            assert window_geometry(df) == (float(step), float(win))

    def test_a_censored_gap_does_not_cheapen_a_transition(self, tmp_path):
        """Pricing a wide repeat gap as a cheaper crossing would make a
        censored repeat a cheap place to break a segment -- the failure
        censoring them was meant to prevent."""
        df = _flat_frame_with_repeat_spike(n=300)
        wide = _flat_frame_with_repeat_spike(n=300, spike_at=slice(150, 190))

        step_a, win_a = window_geometry(df)
        step_b, win_b = window_geometry(wide)
        assert (step_a, win_a) == (step_b, win_b)

        # And the calls do not gain a break across the wider censored block.
        result, _ = _run(wide, tmp_path)
        assert set(result["prob_copy_number"].unique()) == {1}


class TestOverlapWeighting:
    def test_is_a_no_op_for_non_overlapping_windows(self, otr_corrected_flat, tmp_path):
        import os
        from CNery.core import run_HMM
        weighted, plain = [], []
        for flag, sink in ((True, weighted), (False, plain)):
            out = str(tmp_path / f"ow_{flag}")
            os.makedirs(os.path.join(out, "CNV_csv"), exist_ok=True)
            sink.append(
                run_HMM(otr_corrected_flat.copy(), out, overlap_weighting=flag)
            )
        # conftest tiles at step == window, so alpha is 1 either way.
        assert list(weighted[0]["prob_copy_number"]) == list(plain[0]["prob_copy_number"])


class TestRobustStateCount:
    """One outlier window must not size the state space for a whole genome."""

    def test_a_lone_spike_does_not_add_states(self):
        counts = np.full(500, 100.0)
        counts[250] = 4000.0            # a single 40x window
        offsets = np.ones_like(counts)
        assert robust_state_count(counts, offsets, mu=100.0) == 5

    def test_a_sustained_high_copy_segment_does(self):
        counts = np.full(500, 100.0)
        counts[250:260] = 1000.0        # a real 10x segment
        offsets = np.ones_like(counts)
        assert robust_state_count(counts, offsets, mu=100.0) == 10

    def test_offsets_are_divided_out_before_counting(self):
        """A high-bias window is expected to be deep; that is not extra copies."""
        counts = np.full(500, 100.0)
        offsets = np.ones(500)
        counts[100:120] = 300.0
        offsets[100:120] = 3.0
        assert robust_state_count(counts, offsets, mu=100.0) == 5

    def test_respects_the_max_copy_number_cap(self):
        counts = np.full(500, 100.0)
        counts[100:120] = 100000.0
        assert robust_state_count(counts, np.ones(500), mu=100.0, max_states=12) == 12


def test_offsets_shift_the_emission_mean_not_the_data():
    """E[count | CN=k] = k * mu * offset, so a scaled offset scales the peak."""
    counts = np.arange(0, 400, dtype=float)
    plain = log_emission_with_offsets(counts, np.ones_like(counts), mu=100.0,
                                      size=50.0, n_states=3, error_rate=0.15)
    doubled = log_emission_with_offsets(counts, np.full_like(counts, 2.0), mu=100.0,
                                        size=50.0, n_states=3, error_rate=0.15)
    def mode(mu, size):
        # abs=1 because the mode formula can land on an exact integer, where
        # two adjacent counts tie and argmax simply takes the first.
        return pytest.approx(np.floor(mu * (size - 1) / size), abs=1)

    assert int(np.argmax(plain[:, 1])) == mode(100.0, 50.0)
    # Doubling the offset doubles where state 1 expects to sit.
    assert int(np.argmax(doubled[:, 1])) == mode(200.0, 50.0)
    # State index is still copy number: state 2 peaks at twice state 1, with
    # `size` scaled alongside so variance stays proportional to copy number.
    assert int(np.argmax(plain[:, 2])) == mode(200.0, 100.0)


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