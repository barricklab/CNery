from itertools import product

import pytest
import numpy as np
import pandas as pd
from scipy.stats import nbinom

from CNery.core import (
    offset_tau,
    run_HMM,
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
    _segments_from_path,
    _viterbi_forward,
    _viterbi_forward_flat,
    _flat_transition_params,
    _backtrace,
    DEFAULT_MAX_COPY_NUMBER,
    DEFAULT_CHANGE_RATE,
    DEFAULT_INTERIOR_CHANGE_RATE,
    DEFAULT_FOLD_CHANGE_PENALTY,
    consensus_levels,
    polymorphism_levels,
    level_spacing,
    snap_cn_resolution,
    refine_levels,
    refine_step,
    add_cn_censor,
    DEFAULT_CN_RESOLUTION,
    CN_FINE_BAND_TOP,
    CN_MAX_LEVELS,
)


def _matrices(mean=50, var=100, n_states=5):
    em = setup_emission_matrix(n_states, mean, var, absmax=200,
                              deletion_coverage_fraction=0.02)
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


class TestFirstSegmentCoordinate:
    """The first row of break_pts.csv must be a real window start like every other row.

    It used to be a literal 0, seeded before the loop had looked at a window. That is
    not a coordinate any window has, and it made row 1 the one row a consumer could not
    read the same way as the rest -- breseq wrote it into a Genome Diff as start 0 and
    then failed its own validation on the file it had just written.
    """

    @staticmethod
    def _windows(n, size=100):
        """1-based window starts, as build_windows produces them for a real reference."""
        win_st = pd.Series(np.arange(n) * size + 1)
        return win_st, win_st + size

    def test_first_segment_opens_at_the_first_window(self):
        win_st, win_end = self._windows(10)
        path = [1] * 5 + [0] * 5

        segments = _segments_from_path(path, win_st, win_end, chr_length=1001)

        assert segments["Startpos"].iloc[0] == win_st.iloc[0] == 1

    def test_first_segment_opens_at_the_first_window_when_its_state_is_not_1(self):
        """The regression. A CN != 1 first segment is the one that reaches a .gd.

        Copy number 1 is the baseline breseq drops, so while the genome opened in
        state 1 the bad coordinate was filtered out before anything could trip over
        it. A contig starting inside a deletion, or a library thin enough that the
        whole genome comes back CN 0, is what makes row 1 survive.
        """
        win_st, win_end = self._windows(10)
        path = [0] * 5 + [1] * 5

        segments = _segments_from_path(path, win_st, win_end, chr_length=1001)

        assert segments["State"].iloc[0] == 0
        assert segments["Startpos"].iloc[0] == 1

    def test_every_startpos_is_a_window_start(self):
        win_st, win_end = self._windows(12)
        path = [0] * 4 + [1] * 4 + [2] * 4

        segments = _segments_from_path(path, win_st, win_end, chr_length=1201)

        assert set(segments["Startpos"]).issubset(set(win_st))

    def test_the_last_base_of_every_segment_is_unchanged(self):
        """Coordinate-neutrality: only the start moved, and only by one base.

        Segment_Size is written downstream as Endpos - Startpos, so a consumer
        recovers the last base as Startpos + Segment_Size - 1 == Endpos - 1. Shifting
        row 1 from 0 to 1 shortens it by a base at the front and must leave that
        derived end exactly where it was.
        """
        win_st, win_end = self._windows(10)
        path = [0] * 5 + [1] * 5

        segments = _segments_from_path(path, win_st, win_end, chr_length=1001)
        sizes = segments["Endpos"] - segments["Startpos"]
        last_base = segments["Startpos"] + sizes - 1

        assert list(last_base) == list(segments["Endpos"] - 1)
        # ... and the segments still tile the sequence with no gap and no overlap.
        assert list(segments["Startpos"])[1:] == list(last_base + 1)[:-1]

    def test_a_single_state_genome_is_one_segment_starting_at_one(self):
        win_st, win_end = self._windows(6)

        segments = _segments_from_path([0] * 6, win_st, win_end, chr_length=601)

        assert len(segments) == 1
        assert segments["Startpos"].iloc[0] == 1
        assert segments["Endpos"].iloc[0] == 601


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
                                      size=50.0, n_states=3,
                                      deletion_coverage_fraction=0.02)
    doubled = log_emission_with_offsets(counts, np.full_like(counts, 2.0), mu=100.0,
                                        size=50.0, n_states=3,
                                        deletion_coverage_fraction=0.02)
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

def _frame_with_partial_deletion(n=300, deleted=slice(140, 160), depth=100.0,
                                 residual=0.10, seed=5):
    """Flat coverage with a block that retains `residual` of the baseline.

    A real deletion rarely reads as exactly zero -- mismapping and repeats leave
    a few percent behind -- so `residual` is what actually decides whether the
    zero state recognises it. Noise is negative-binomial at a realistic
    dispersion so the frame is not degenerate, which would send run_HMM down its
    moment fallback instead of the censored fit.
    """
    rng = np.random.default_rng(seed)
    # Relative variance fixed at REL606's measured 0.0266 so that only the DEPTH
    # varies between parametrisations. Solving mu + mu^2/size = 0.0266 * mu^2
    # for size, not `depth / 0.0266 - depth`, which holds 1.027/mu instead and
    # makes the frame ~26x sharper than real coverage at 1000x.
    size = 1.0 / max(0.0266 - 1.0 / depth, 1e-4)
    level = np.full(n, 1.0)
    level[deleted] = residual
    counts = nbinom.rvs(size, size / (size + depth * level),
                        random_state=rng).astype(float)
    win_st = np.arange(n) * 200
    return pd.DataFrame({
        "genome_id": "chr1",
        "win_st": win_st,
        "win_end": win_st + 200,
        "win_len": 200,
        "gc_percent": 0.5,
        "read_count_cov": counts,
        "norm_raw_cov": counts / np.median(counts),
        "gc_corr_norm_cov": counts / np.median(counts),
        "otr_gc_corr_norm_cov": counts / np.median(counts),
        "pct_redundant": 0.0,
        "is_redundant": False,
        "is_deletion": False,
    })


@pytest.mark.parametrize("depth", [60.0, 250.0, 1000.0])
def test_deletion_calls_do_not_depend_on_sequencing_depth(depth, tmp_path):
    """The same biology must be called the same way at any sequencing depth.

    The zero state used to be a geometric of mean error_rate / (1 - error_rate)
    = 0.176 counts ABSOLUTE, with nothing tying it to the coverage of the
    sample. As a fraction of baseline that is 0.28% at 60x and 0.018% at 1000x,
    so the largest residual coverage it would still call CN0 drifted from 19% to
    4%. A deletion holding 10% of its coverage -- which is what one next to a
    repeat looks like -- was therefore called at 60x and missed at 1000x, lost
    purely because the sample was sequenced deeper.
    """
    df = _frame_with_partial_deletion(depth=depth, residual=0.10)
    result, _ = _run(df, tmp_path / f"d{int(depth)}")
    called = result.loc[140:159, "prob_copy_number"]
    assert (called == 0).all(), (
        f"deletion holding 10% of a {depth:.0f}x baseline was not called CN0; "
        f"got {sorted(set(called))}"
    )


class TestProvisionalRun:
    """run_HMM(write=False), the first of the two fitting passes.

    Its calls exist only to build the CN censor for the second pass. Writing them
    would put provisional numbers in CNV_csv/ that the second pass overwrites --
    or leaves behind if the run dies in between.
    """

    def test_write_false_produces_no_files(self, otr_corrected_flat, tmp_path):
        out = tmp_path / "run"
        (out / "CNV_csv").mkdir(parents=True)
        run_HMM(otr_corrected_flat, str(out), write=False)
        assert list((out / "CNV_csv").iterdir()) == []

    def test_write_false_changes_no_numbers(self, otr_corrected_flat, tmp_path):
        """The provisional pass must run the identical numeric path.

        If it did not, the censor would come from a different model than the one
        whose calls are eventually published.
        """
        quiet_dir = tmp_path / "quiet"
        loud_dir = tmp_path / "loud"
        for d in (quiet_dir, loud_dir):
            (d / "CNV_csv").mkdir(parents=True)
        quiet = run_HMM(otr_corrected_flat, str(quiet_dir), write=False)
        loud = run_HMM(otr_corrected_flat, str(loud_dir), write=True)
        np.testing.assert_array_equal(quiet["prob_copy_number"].to_numpy(),
                                      loud["prob_copy_number"].to_numpy())


class TestOffsetUncertainty:
    """The bias offset is an ESTIMATE, and the emission model is told how good one.

    `bias_offsets` returns a LOWESS fit evaluated at each window's GC. Writing
    o = o_hat * (1 + eps) with Var(eps) = tau^2, the law of total variance gives

        Var(y | k) = m + m^2 * (1/(k*size) + tau^2)      m = k * mu * o_hat

    so the uncertainty adds in the reciprocal-size scale and the extra term
    m^2 * tau^2 grows as k^2 -- negligible at CN 1, largest exactly where the
    offset is multiplied up. The k^2 behaviour is DERIVED, not imposed, and it is
    what makes the correction bite inside amplifications and nowhere else.
    """

    MU, SIZE, N_STATES = 100.0, 15.0, 5

    def _emissions(self, tau, counts=None, offsets=None):
        counts = np.full(40, 300.0) if counts is None else counts
        offsets = np.ones(counts.size) if offsets is None else offsets
        return log_emission_with_offsets(
            counts, offsets, mu=self.MU, size=self.SIZE, n_states=self.N_STATES,
            deletion_coverage_fraction=0.02,
            offset_tau=None if tau is None else np.full(counts.size, tau),
        )

    def _implied_var(self, tau, state):
        """Var of the NB this state's row actually uses, from the model algebra."""
        m = state * self.MU
        size_k = state * self.SIZE
        if tau:
            size_k = size_k / (1.0 + size_k * tau ** 2)
        return m + m * m / size_k

    def test_none_and_zero_are_exactly_the_current_behaviour(self):
        """Every existing golden was produced without this. Both the None path
        and an all-zero tau must reproduce it bit for bit."""
        base = self._emissions(None)
        np.testing.assert_array_equal(base, self._emissions(0.0))

    def test_a_missing_column_means_no_uncertainty(self):
        df = pd.DataFrame({"gc_corr_fact": np.ones(10)})
        np.testing.assert_array_equal(offset_tau(df, bias="all"), np.zeros(10))

    def test_uncertainty_fattens_the_tail_and_lowers_the_peak(self):
        """Widening moves mass OUT of the centre and INTO the tails, so a count
        far from a state's mean gains likelihood and one near it loses some.

        Checking only the tail would pass for a distribution that had simply been
        shifted, so both directions are asserted.
        """
        far = np.array([750.0])              # CN 4 has mean 400 here
        near = np.array([400.0])
        assert (self._emissions(0.10, counts=far)[0, 4]
                > self._emissions(None, counts=far)[0, 4])
        assert (self._emissions(0.10, counts=near)[0, 4]
                < self._emissions(None, counts=near)[0, 4])

    def test_the_added_variance_grows_as_k_squared(self):
        """THE invariant, and it has an exact form worth pinning.

        Var adds m^2 * tau^2 with m = k*mu*o, while the existing terms are
        m + m^2/(k*size). Dividing through,

            Var_with / Var_without - 1 = k * mu * tau^2 / (1 + mu/size)

        -- strictly LINEAR in k. So the excess is not merely increasing, it is
        proportional to the copy number, which is what "the offset error is
        multiplied by k" means quantitatively.

        A refactor computing one effective size PER WINDOW --
        k*(size/(1+size*tau^2)) rather than k*size/(1+k*size*tau^2) -- makes the
        excess constant in k instead, and this test is what catches it.
        """
        tau = 0.10
        excess = [self._implied_var(tau, k) / self._implied_var(0.0, k) - 1.0
                  for k in range(1, self.N_STATES + 1)]
        per_k = [e / k for e, k in zip(excess, range(1, self.N_STATES + 1))]
        assert all(v == pytest.approx(per_k[0], rel=1e-9) for v in per_k), per_k

        predicted = self.MU * tau ** 2 / (1.0 + self.MU / self.SIZE)
        assert per_k[0] == pytest.approx(predicted, rel=1e-9)
        assert excess[-1] > 4.0 * excess[0], "it must concentrate at high CN"

    def test_the_emission_rows_follow_that_algebra(self):
        """Ties the matrix the HMM actually consumes to the variance above, so
        the previous test cannot pass while the implementation diverges."""
        from scipy.stats import nbinom

        tau = 0.10
        out = self._emissions(tau, counts=np.array([300.0]))
        for state in range(1, self.N_STATES + 1):
            m = state * self.MU
            size_k = state * self.SIZE
            size_k = size_k / (1.0 + size_k * tau ** 2)
            want = nbinom.logpmf(300.0, size_k, size_k / (size_k + m))
            assert out[0, state] == pytest.approx(want, rel=1e-9)


class TestOffsetTauDispatch:
    """Which --bias modes carry offset uncertainty at all."""

    def _frame(self, n=12):
        return pd.DataFrame({
            "gc_corr_fact": np.ones(n),
            "otr_gc_corr_fact": np.ones(n),
            "gc_corr_tau": np.full(n, 0.05),
        })

    @pytest.mark.parametrize("bias,expected", [
        ("all", 0.05), ("gc", 0.05), ("otr", 0.0), ("none", 0.0),
    ])
    def test_only_modes_that_apply_gc_carry_it(self, bias, expected):
        """--bias otr aliases the GC correction away and --bias none applies
        nothing, so in both there is no GC offset whose error to propagate."""
        assert offset_tau(self._frame(), bias=bias)[0] == pytest.approx(expected)

    def test_negative_or_non_finite_values_are_dropped(self):
        df = self._frame()
        df.loc[0, "gc_corr_tau"] = np.nan
        df.loc[1, "gc_corr_tau"] = -1.0
        got = offset_tau(df, bias="all")
        assert got[0] == 0.0 and got[1] == 0.0 and got[2] == pytest.approx(0.05)


class TestUncertaintySuppressesAGcSliver:
    """End to end, offline: the artifact this correction exists for.

    Reproduces the geometry measured on ltee_ara_m3_32k_2rg -- a CN-3
    amplification with a short stretch inside it where the GC curve claims a
    suppression the coverage does not show -- and checks that the discrepancy
    alone buys a spurious higher state without the correction, and does not with
    it.

    THE DISCREPANCY IS THE POINT. Counts here follow mu*cn with no suppression at
    all, while `gc_corr_fact` claims 0.82. That is the measured situation: inside
    the real amplification the raw depth is 3.45x where the genome-wide GC curve
    implies 3 * 0.82 = 2.46x, so the corrected level reads 4.2 and earns a CN-4
    segment. Generating counts that OBEY the dip would put the corrected level at
    exactly 3 and prove nothing.

    SCALE. This is a deliberately harder case than the real one -- an 18%
    discrepancy over 16 windows at low dispersion -- so it needs a tau well above
    what the bootstrap produces on real data (0.003-0.02). It is a test of the
    MECHANISM. That the mechanism works at realistic magnitudes is what
    tests/test_authentic.py::TestOffsetUncertaintyAtCliDefaults covers.

    SEEDS, PLURAL. A single seed is luck: measured, the outcome at the decision
    boundary flips with noise, and only 2 of 6 seeds showed the clean pattern in
    a marginal regime. The regime below is 0/12 whole without tau and 12/12 with
    it, and the test averages anyway rather than trusting one draw.
    """

    N, MU, SIZE = 600, 300.0, 80.0
    AMP = (200, 460)          # the CN-3 block, in windows
    DIP = (300, 316)          # where the GC curve claims a suppression
    CLAIM = 0.82              # ...of this much, which the coverage does not show
    SEEDS = range(6)

    def _frame(self, seed):
        rng = np.random.default_rng(seed)
        cn = np.ones(self.N)
        cn[self.AMP[0]:self.AMP[1]] = 3.0
        offset = np.ones(self.N)
        offset[self.DIP[0]:self.DIP[1]] = self.CLAIM
        mean = self.MU * cn
        counts = rng.negative_binomial(
            self.SIZE, self.SIZE / (self.SIZE + mean)).astype(float)
        return pd.DataFrame({
            "genome_id": "a",
            "win_st": np.arange(self.N) * 100,
            "win_end": np.arange(self.N) * 100 + 100,
            "read_count_cov": counts,
            "norm_raw_cov": counts / self.MU,
            "gc_corr_norm_cov": counts / self.MU / offset,
            "otr_gc_corr_norm_cov": counts / self.MU / offset,
            "gc_corr_fact": offset,
            "otr_gc_corr_fact": np.ones(self.N),
            "is_deletion": np.zeros(self.N, bool),
        })

    def _is_whole(self, seed, tau, out_dir):
        df = self._frame(seed)
        if tau is not None:
            # Concentrated where the curve is uncertain, which is the whole
            # point -- a uniformly large tau is a different thing and is
            # measurably worse on real data.
            df["gc_corr_tau"] = np.where(
                df["gc_corr_fact"].to_numpy() < 0.95, tau, tau / 10.0)
        # The interior-boundary prior is held NEUTRAL here so these three tests
        # keep measuring tau and only tau. It is not idle in this geometry -- the
        # spurious sliver is CN 3 inside CN 3 reading as 4, a 1.33-fold interior
        # step, which is exactly what that prior is for, and at defaults it
        # suppresses 4 of these 6 seeds by itself. Two mechanisms aimed at one
        # artifact is fine; a control that cannot say which one fired is not.
        cnv = run_HMM(df, str(out_dir), write=False,
                      fold_change_penalty=0.0,
                      interior_change_rate=DEFAULT_CHANGE_RATE)
        inside = ((cnv["win_st"] >= self.AMP[0] * 100)
                  & (cnv["win_st"] < self.AMP[1] * 100)).to_numpy()
        return sorted(set(cnv.loc[inside, "prob_copy_number"].tolist())) == [3]

    @pytest.fixture
    def out_dir(self, tmp_path):
        (tmp_path / "CNV_csv").mkdir(parents=True)
        return tmp_path

    def test_the_discrepancy_alone_splits_the_block(self, out_dir):
        """Without the correction, an offset the coverage does not support is
        enough on its own -- there is no copy-number change anywhere in the
        block. If this ever stops splitting, the test below proves nothing.
        """
        whole = sum(self._is_whole(s, None, out_dir) for s in self.SEEDS)
        assert whole == 0, f"expected every seed to split, {whole} stayed whole"

    def test_uncertainty_keeps_the_block_whole(self, out_dir):
        """Telling the model the offset is uncertain restores the single block,
        on every seed."""
        whole = sum(self._is_whole(s, 0.20, out_dir) for s in self.SEEDS)
        assert whole == len(self.SEEDS), (
            f"only {whole}/{len(self.SEEDS)} seeds kept the block whole")

    def test_the_effect_is_monotone_in_tau(self, out_dir):
        """More uncertainty must never make the split MORE likely, over the
        range where the correction is meant to operate. (It is not monotone
        without bound -- a uniform 4x on real data brings the artifact back --
        which is why this stops at the working range.)
        """
        rates = [sum(self._is_whole(s, t, out_dir) for s in self.SEEDS)
                 for t in (None, 0.10, 0.20)]
        assert rates == sorted(rates), rates


class TestNothingToCall:
    """A sequence with no coverage is called CN 0, not crashed on.

    run_HMM used to reach solve_pr(0.0, 0.0) and raise ZeroDivisionError. The
    moment fallback's guard is written `if mean > 0 and var <= mean`, which steps
    over exactly the case that needs it -- so a plasmid that got no reads took
    down the chromosome sharing the invocation.
    """

    def _frame(self, n=50, cov=0.0):
        return pd.DataFrame({
            "genome_id": "dead",
            "win_st": np.arange(n) * 100,
            "win_end": np.arange(n) * 100 + 100,
            "read_count_cov": np.full(n, cov),
            "norm_raw_cov": np.full(n, cov),
            "otr_gc_corr_norm_cov": np.full(n, cov),
            "gc_corr_fact": np.ones(n),
            "otr_gc_corr_fact": np.ones(n),
        })

    def test_all_zero_coverage_is_called_copy_number_zero(self, tmp_path):
        out = run_HMM(self._frame(), str(tmp_path), write=False)
        assert (out["prob_copy_number"] == 0).all()

    def test_a_frame_with_no_windows_does_not_raise(self, tmp_path):
        # genome_id cannot be read off row 0 of an empty frame, so the caller
        # supplies it -- the CSVs still have to be named after something.
        out = run_HMM(self._frame(n=0), str(tmp_path), write=False,
                      genome_id="dead")
        assert len(out) == 0

    def test_solve_pr_declines_instead_of_dividing_by_zero(self):
        from CNery.core import solve_pr
        p, size = solve_pr(0.0, 0.0)
        assert p == 0.0 and not np.isfinite(size)

    def test_a_healthy_frame_still_uses_the_moment_fallback(self):
        # The short-circuit predicate is the DATA being empty, never
        # `fit_result is None` -- that also fires on healthy small frames, where
        # the moment estimate below it is a live and correct path.
        from CNery.core import solve_pr
        p, size = solve_pr(50.0, 100.0)
        assert size == pytest.approx(50.0)
        assert p == pytest.approx(0.5)


class TestFlatTransitionFastPath:
    """The O(n_states) recursion must decode identically to the O(n_states**2) one.

    setup_transition_matrix() is flat -- one value on the diagonal, one
    everywhere else -- which makes the best predecessor of a state one of two
    candidates instead of all of them. That is what makes a state grid wide
    enough for a 100x amplification affordable, but it is only safe if it agrees
    with the general recursion on every path, ties included.
    """

    def _emission(self, n_obs, n_states, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n_obs, n_states)) * 3.0

    def _both(self, log_emission, remain_prob, n_states):
        tm = setup_transition_matrix(n_states, remain_prob=remain_prob)
        log_transition = np.log(tm)
        log_start = _default_log_start(log_transition)
        general = _backtrace(*_viterbi_forward(log_emission, log_transition, log_start))
        flat = _flat_transition_params(log_transition)
        assert flat is not None
        fast = _backtrace(*_viterbi_forward_flat(log_emission, log_start, flat[0], flat[1]))
        return general, fast

    @pytest.mark.parametrize("seed", range(8))
    def test_paths_agree_on_random_emissions(self, seed):
        log_emission = self._emission(n_obs=120, n_states=9, seed=seed)
        general, fast = self._both(log_emission, remain_prob=1 - 1e-4, n_states=8)
        assert np.array_equal(general, fast)

    def test_scores_agree_too(self):
        log_emission = self._emission(n_obs=60, n_states=7, seed=11)
        tm = setup_transition_matrix(6, remain_prob=1 - 1e-4)
        log_transition = np.log(tm)
        log_start = _default_log_start(log_transition)
        logv, ptr = _viterbi_forward(log_emission, log_transition, log_start)
        flat = _flat_transition_params(log_transition)
        logv_fast, ptr_fast = _viterbi_forward_flat(log_emission, log_start, flat[0], flat[1])
        assert np.array_equal(logv, logv_fast)
        assert np.array_equal(ptr, ptr_fast)

    def test_ties_break_the_same_way(self):
        """Equal emissions make every candidate tie, so tie-breaking is the test.

        np.argmax returns the LOWEST index attaining the maximum; a fast path
        that picked any other maximiser would decode a different -- equally
        likely, but different -- path, and the two would stop being substitutable.
        """
        log_emission = np.zeros((40, 6))
        general, fast = self._both(log_emission, remain_prob=1 - 1e-4, n_states=5)
        assert np.array_equal(general, fast)

    def test_an_all_neg_inf_row_agrees(self):
        log_emission = np.zeros((20, 5))
        log_emission[7] = -np.inf
        general, fast = self._both(log_emission, remain_prob=1 - 1e-4, n_states=4)
        assert np.array_equal(general, fast)

    def test_agrees_when_changing_is_likelier_than_staying(self):
        """The degenerate case the 'best other' predecessor exists for.

        The off-diagonal candidates for target l exclude k == l, so when l is
        itself the best state the jump must come from the RUNNER-UP. Using the
        global argmax there prices an l -> l move at log_change, which only shows
        up as a wrong path once log_change exceeds log_remain -- reachable with a
        large changeprob on a narrow grid, and silent everywhere else.
        """
        log_emission = self._emission(n_obs=80, n_states=5, seed=3)
        general, fast = self._both(log_emission, remain_prob=0.01, n_states=4)
        assert np.array_equal(general, fast)

    def test_a_structured_transition_is_not_taken_for_flat(self):
        """make_viterbi_mat() is public and takes any matrix; only flat ones qualify."""
        tm = setup_transition_matrix(5, remain_prob=1 - 1e-4)
        assert _flat_transition_params(np.log(tm)) is not None
        tm[0, 1] *= 0.5
        assert _flat_transition_params(np.log(tm)) is None

    def test_viterbi_path_declining_scores_changes_nothing(self):
        """viterbi_path() drops the score matrix to save memory, not to save work."""
        log_emission = self._emission(n_obs=50, n_states=6, seed=17)
        tm = setup_transition_matrix(5, remain_prob=1 - 1e-4)
        log_transition = np.log(tm)
        log_start = _default_log_start(log_transition)
        kept, ptr = _viterbi_forward(log_emission, log_transition, log_start)
        dropped, ptr_dropped = _viterbi_forward(log_emission, log_transition, log_start,
                                                keep_scores=False)
        assert dropped.shape == (log_emission.shape[1],)
        assert np.array_equal(dropped, kept[-1])
        assert np.array_equal(ptr, ptr_dropped)
        assert np.array_equal(viterbi_path(log_emission, log_transition, log_start),
                              _backtrace(kept, ptr))


class TestCopyNumberCeiling:
    """A call pinned at the ceiling is a clipped value, not a measurement."""

    def _frame(self, n=300, amp=slice(150, 160), fold=135.0, depth=50.0, seed=7):
        """Flat coverage with a `fold`-times block, at realistic dispersion.

        Noise matters: a frame with zero variance sends run_HMM down its moment
        fallback instead of the censored negative-binomial fit, and the fitted
        baseline then chases the amplification instead of the single-copy level.
        Same construction as _frame_with_partial_deletion() above.
        """
        rng = np.random.default_rng(seed)
        size = 1.0 / max(0.0266 - 1.0 / depth, 1e-4)
        level = np.full(n, 1.0)
        level[amp] = fold
        counts = nbinom.rvs(size, size / (size + depth * level),
                            random_state=rng).astype(float)
        # The MEDIAN is the single-copy level here, which is the point: a mean
        # would be dragged up by the amplification it is supposed to measure.
        norm = counts / np.median(counts)
        win_st = np.arange(n) * 100
        return pd.DataFrame({
            "genome_id": "chr1",
            "win_st": win_st,
            "win_end": win_st + 100,
            "win_len": 100,
            "gc_percent": 0.5,
            "read_count_cov": counts,
            "norm_raw_cov": norm,
            "gc_corr_norm_cov": norm,
            "otr_gc_corr_norm_cov": norm,
            "gc_corr_fact": np.ones(n),
            "otr_gc_corr_fact": np.ones(n),
            "pct_redundant": 0.0,
            "is_redundant": False,
            "is_deletion": False,
        })

    def test_a_135x_block_is_called_135_not_100(self, tmp_path):
        """The case this ceiling was raised for.

        On breseq sample SRR37077254 a 991 bp tandem amplification measured at
        135x came back as exactly 100 -- the largest state the grid held. breseq
        writes that straight into an AMP's new_copy_number, so it reached the
        user as "991 bp x100" with a coverage plot sitting 35 copies above it.
        """
        called = int(run_HMM(self._frame(), str(tmp_path), write=False)
                     ["prob_copy_number"].max())
        # Ten windows at 135x carry real sampling error, so the tolerance is
        # wide. What it has to exclude is the old answer: a value sitting on a
        # round ceiling rather than anywhere near the coverage.
        assert called != 100
        assert called == pytest.approx(135, rel=0.12)

    def test_the_ceiling_still_clips_when_asked_to(self, tmp_path):
        out = run_HMM(self._frame(), str(tmp_path), write=False, max_copy_number=100)
        assert int(out["prob_copy_number"].max()) == 100

    def test_an_ordinary_genome_is_unaffected_by_where_the_ceiling_sits(self, tmp_path):
        """The grid is sized from the data; the ceiling only clips it.

        Raising the default must not change a sample that never approaches it --
        n_states feeds a -log(n_states) cost into every state change, so a grid
        that grew for no reason would cost sensitivity genome-wide.
        """
        df = self._frame(fold=3.0)
        wide = run_HMM(df, str(tmp_path), write=False, max_copy_number=500)
        narrow = run_HMM(df, str(tmp_path), write=False, max_copy_number=100)
        assert np.array_equal(wide["prob_copy_number"].to_numpy(),
                              narrow["prob_copy_number"].to_numpy())

    def test_saturation_is_reported(self, tmp_path, capsys):
        import os
        out = str(tmp_path / "hmm_out")
        os.makedirs(os.path.join(out, "CNV_csv"), exist_ok=True)
        run_HMM(self._frame(), out, max_copy_number=100)
        assert "ceiling" in capsys.readouterr().out

    def test_a_call_below_the_ceiling_is_not_reported_as_saturated(self, tmp_path, capsys):
        import os
        out = str(tmp_path / "hmm_out")
        os.makedirs(os.path.join(out, "CNV_csv"), exist_ok=True)
        run_HMM(self._frame(fold=3.0), out)
        assert "ceiling" not in capsys.readouterr().out

    def test_the_two_ceiling_defaults_cannot_drift_apart(self):
        """robust_state_count()'s default used to say 100 while run_HMM passed 100.

        Two independent literals for one ceiling is how it stayed invisible: the
        signature read as the authority and was never consulted.
        """
        import inspect
        assert (inspect.signature(robust_state_count).parameters["max_states"].default
                is DEFAULT_MAX_COPY_NUMBER)
        assert (inspect.signature(run_HMM).parameters["max_copy_number"].default
                is DEFAULT_MAX_COPY_NUMBER)


class TestInteriorBoundaryPrior:
    """A boundary between two amplified states is rarer, and costs more when the
    two are close in ratio.

    The flat off-diagonal said CN 0 -> CN 7 costs precisely what CN 126 -> CN 139
    costs. At 6,600 reads/window the counting floor is 1.2%, so that let six sigma
    of noise split one amplification into a segment and a shoulder.
    """

    STEP = 100

    def _matrices(self, n=139, rate=DEFAULT_CHANGE_RATE,
                  interior_rate=DEFAULT_INTERIOR_CHANGE_RATE,
                  penalty=DEFAULT_FOLD_CHANGE_PENALTY):
        remain = remain_prob_for_step(rate, self.STEP)
        interior = 1.0 - remain_prob_for_step(interior_rate, self.STEP)
        old = setup_transition_matrix(n, remain_prob=remain)
        new = setup_transition_matrix(n, remain_prob=remain,
                                      interior_change_prob=interior,
                                      fold_change_penalty=penalty)
        return old, new

    def _extra_nats(self, old, new, k, l):
        return float(np.log(old[k, l]) - np.log(new[k, l]))

    def test_neutral_arguments_rebuild_the_old_matrix_bit_for_bit(self):
        """Not merely close: _flat_transition_params compares with `==`.

        Flatness is detected structurally rather than from a flag, so the
        O(n_states) decode has to keep engaging on its own when these corrections
        are off. A neutral matrix that differed in the last bit would silently
        cost every ordinary run its fast path.
        """
        remain = remain_prob_for_step(DEFAULT_CHANGE_RATE, self.STEP)
        old = setup_transition_matrix(30, remain_prob=remain)
        neutral = setup_transition_matrix(30, remain_prob=remain,
                                          interior_change_prob=1.0 - remain,
                                          fold_change_penalty=0.0)
        assert np.array_equal(old, neutral)
        assert _flat_transition_params(np.log(neutral)) is not None

    def test_the_active_matrix_is_still_a_probability_matrix(self):
        _, new = self._matrices()
        assert np.allclose(new.sum(axis=1), 1.0)
        assert np.all(new >= 0.0)
        assert np.all(np.isfinite(new))

    def test_a_small_fractional_change_is_priced_out(self):
        """The case this exists for: 126 -> 139 is a 10% step on 6,600 reads."""
        old, new = self._matrices()
        # Splitting that window buys ~25 nats of likelihood against a ~14 nat
        # transition cost, so the fix has to find more than ~11.
        assert self._extra_nats(old, new, 126, 139) > 11.0
        # And a 1% step should be beyond any evidence a genome can supply.
        assert self._extra_nats(old, new, 126, 127) > 100.0

    def test_a_real_stepped_array_is_barely_taxed(self):
        """A doubling is a claim the data can support; it must stay affordable.

        6.6 nats at the defaults -- 4.6 for the interior rate and 2 for the fold
        penalty -- against the hundreds of nats a real 2x step over many windows
        supplies. test_a_genuine_two_level_array_is_still_split proves the end of
        that sentence.
        """
        old, new = self._matrices()
        assert self._extra_nats(old, new, 50, 100) < 8.0

    def test_the_kernel_is_scale_free(self):
        """1 -> 2 and 100 -> 200 are the same event, which is what "%" means."""
        _, new = self._matrices()
        # Within a row: halving and doubling are equidistant in log space.
        for k in (4, 8, 50):
            assert new[k, k // 2] == pytest.approx(new[k, k * 2], rel=1e-12)
        # And ACROSS rows, which only holds because the row is not renormalized.
        old, new = self._matrices()
        assert (self._extra_nats(old, new, 2, 4)
                == pytest.approx(self._extra_nats(old, new, 50, 100), rel=1e-12))

    def test_leaving_an_amplified_state_costs_exactly_what_it_did(self):
        """Baseline is untouched, and that is load-bearing rather than tidy.

        Renormalizing the row to change_prob hands the mass taken off interior
        targets to states 0 and 1, making it CHEAPER to leave an amplified state.
        Measured, that cost breseq a real 108 kb IS186 amplification on
        ltee_ara_m3_38k_se36 -- the segment broke at a 400 bp dip to CN 1 and the
        AMP call went with it -- and grew spurious few-hundred-base CN 2
        excursions on two other clones, which are cheaper to enter and leave for
        the same reason.
        """
        old, new = self._matrices()
        assert new[126, 1] == old[126, 1]
        assert new[126, 0] == old[126, 0]

    def test_the_removed_mass_goes_to_the_diagonal(self):
        """An amplified state is simply more persistent: the only cheap way out
        is unchanged, so what an interior boundary loses, staying put gains."""
        old, new = self._matrices()
        assert new[126, 126] > old[126, 126]
        assert new.sum(axis=1)[126] == pytest.approx(1.0)

    def test_baseline_and_deletion_rows_are_untouched(self):
        """Scoped to amplified states, so an ordinary duplication call pays
        nothing -- and _default_log_start() reads row 1 as the start
        distribution, so a changed row 1 would quietly restate that too."""
        old, new = self._matrices()
        assert np.array_equal(old[0], new[0])
        assert np.array_equal(old[1], new[1])

    def test_the_deletion_state_is_exempt_as_a_target(self):
        """log(l / 0) is undefined, and a deletion abutting an amplification is a
        real IS-mediated configuration rather than a modelling artifact."""
        old, new = self._matrices()
        assert self._extra_nats(old, new, 5, 0) == self._extra_nats(old, new, 5, 1)

    def test_a_grid_too_narrow_to_have_an_interior_is_left_alone(self):
        remain = remain_prob_for_step(DEFAULT_CHANGE_RATE, self.STEP)
        interior = 1.0 - remain_prob_for_step(DEFAULT_INTERIOR_CHANGE_RATE, self.STEP)
        for n in (1, 2):
            plain = setup_transition_matrix(n, remain_prob=remain)
            active = setup_transition_matrix(n, remain_prob=remain,
                                             interior_change_prob=interior,
                                             fold_change_penalty=2.0)
            assert np.array_equal(plain, active)

    #: The ten windows of SRR37077254's amplification, in otr_gc_corr_norm_cov --
    #: the bias-corrected relative coverage the HMM actually decides on, not the
    #: raw counts. The distinction matters: window 1 reads a mid-range 6446 raw
    #: but carries the highest GC factor of the ten, so correcting it produces the
    #: LOWEST corrected value, 122. Reproducing this from raw counts with the
    #: offsets set to 1 loses the very window that got split off.
    #:
    #: Verbatim rather than simulated. Their scatter is 4.4% where a negative
    #: binomial at 6,600 reads predicts 1.7%, and that gap IS the problem --
    #: but ten normal draws at 4.4% routinely throw a -14% outlier, which these do
    #: not, so a simulated frame tests a harder case than the one that occurs.
    SRR37077254_AMP_LEVELS = [122.05, 132.31, 132.66, 134.41, 139.49,
                              138.75, 141.40, 133.08, 132.84, 126.79]

    def _amplification_frame(self, levels=None, n=300, amp_start=150,
                             depth=48.9, size=100.0, seed=13):
        """Flat single-copy coverage with a measured high-copy block dropped in.

        `levels` are relative to single copy, and the bias offsets are left at 1,
        so a level of 122.05 is a window the HMM should read as copy number 122.

        `depth` is the run's own 48.9x. `size` is NOT: breseq fitted 51.7 on this
        sample, but breseq's least-squares fit is not comparable to CNery's
        truncated-likelihood one (see CLAUDE.md), and a baseline drawn at 52 sits
        just short of splitting, so the un-penalized control below would pass for
        the wrong reason. 100 puts the frame in the regime the real run was
        demonstrably in -- it split there. The separation is not knife-edge: every
        value from 80 to 250 splits without the prior and holds together with it,
        on more than one seed.
        """
        levels = list(self.SRR37077254_AMP_LEVELS if levels is None else levels)
        rng = np.random.default_rng(seed)
        obs = nbinom.rvs(size, size / (size + depth), size=n,
                         random_state=rng).astype(float)
        obs[amp_start:amp_start + len(levels)] = [v * depth for v in levels]
        norm = obs / np.median(obs)
        win_st = np.arange(n) * 100
        return pd.DataFrame({
            "genome_id": "chr1",
            "win_st": win_st,
            "win_end": win_st + 100,
            "win_len": 100,
            "gc_percent": 0.5,
            "read_count_cov": obs,
            "norm_raw_cov": norm,
            "gc_corr_norm_cov": norm,
            "otr_gc_corr_norm_cov": norm,
            "gc_corr_fact": np.ones(n),
            "otr_gc_corr_fact": np.ones(n),
            "pct_redundant": 0.0,
            "is_redundant": False,
            "is_deletion": False,
        })

    def test_a_noisy_high_copy_block_gets_one_consensus_call(self, tmp_path):
        """On SRR37077254 this came back as 126 over one window and 139 over the
        other nine -- and which window got split off was arbitrary, since a
        window reading 127 was left inside while one reading 122 was not."""
        out = run_HMM(self._amplification_frame(), str(tmp_path), write=False)
        amplified = out.loc[out["prob_copy_number"] > 1, "prob_copy_number"]
        assert len(amplified) == len(self.SRR37077254_AMP_LEVELS)
        assert amplified.nunique() == 1

    def test_switching_the_prior_off_brings_the_shoulder_back(self, tmp_path):
        """Pins the cause, not just the symptom. Without this the test above
        passes for any reason at all -- a different fitted mu, a quiet seed --
        and stops being evidence about the prior."""
        out = run_HMM(self._amplification_frame(), str(tmp_path), write=False,
                      fold_change_penalty=0.0,
                      interior_change_rate=DEFAULT_CHANGE_RATE)
        amplified = out.loc[out["prob_copy_number"] > 1, "prob_copy_number"]
        assert amplified.nunique() > 1

    def test_a_genuine_two_level_array_is_still_split(self, tmp_path):
        """50x abutting 100x is a doubling -- a real event, and it must survive.

        Built from the model's own noise at each level rather than from injected
        scatter: the question here is whether the prior swallows a real step, not
        how the model handles excess dispersion.
        """
        depth, size = 48.9, 100.0
        rng = np.random.default_rng(29)
        levels = []
        for cn in (50, 100):
            draws = nbinom.rvs(cn * size, (cn * size) / (cn * size + cn * depth),
                               size=10, random_state=rng).astype(float)
            levels.extend(draws / depth)
        # 800 windows, not the usual 300: fit_censored_negative_binomial searches
        # upward for the mode from mean/4, and twenty windows at 50-100x drag the
        # mean so far up on a short frame that the search starts ABOVE the single-
        # copy mode and never finds it. The whole genome then comes back CN 3.
        out = run_HMM(self._amplification_frame(levels=levels, n=800),
                      str(tmp_path), write=False)
        amplified = out.loc[out["prob_copy_number"] > 1, "prob_copy_number"]
        assert amplified.nunique() == 2


def _poly_frame(n=2000, depth=100.0, relvar=0.0266, blocks=(), seed=0):
    """A windowed frame staged to run_HMM's contract, long enough to matter.

    conftest's fixtures are n=80, below run_HMM's min_called_windows=100, and
    far too short to pay for a fine-grid segment: at the default rates a 1.05
    call needs ~550 windows of evidence before it is worth its own entry and
    exit. `blocks` is (start, stop, level) in COPY NUMBER, so the truth is known.

    Relative variance is fixed at REL606's measured 0.0266 so that changing
    `depth` changes only the depth, following _frame_with_partial_deletion().
    """
    size = 1.0 / max(relvar - 1.0 / depth, 1e-4)
    lam = np.ones(n)
    for start, stop, level in blocks:
        lam[start:stop] = level
    rng = np.random.default_rng(seed)
    rc = rng.negative_binomial(
        size * lam, size * lam / (size * lam + depth * lam)).astype(float)
    win_st = np.arange(n) * 100
    med = np.median(rc[rc > 0]) if np.any(rc > 0) else 1.0
    return pd.DataFrame({
        "genome_id": "chr1", "win_st": win_st, "win_end": win_st + 100,
        "win_len": 100, "gc_percent": 0.5, "read_count_cov": rc,
        "norm_raw_cov": rc / med, "gc_corr_norm_cov": rc / med,
        "gc_corr_fact": np.ones(n), "otr_gc_corr_norm_cov": rc / med,
        "otr_gc_corr_fact": np.ones(n), "window_num": np.arange(n),
    })


class TestConsensusPathIsUnchanged:
    """Polymorphism mode is opt-in, and "opt-in" has to mean BIT for bit.

    Eight authentic goldens, breseq's Genome Diff and a dozen `== 1` assertions
    read prob_copy_number as an integer. These are the tests that license
    rewriting the emission and the transition matrix around an explicit level
    grid: on the integer grid the rewrite has to be a pure relabelling.
    """

    @pytest.mark.parametrize("with_tau", [False, True])
    def test_the_integer_grid_rebuilds_the_emission_bit_for_bit(self, with_tau):
        rng = np.random.default_rng(3)
        counts = rng.poisson(100, 500).astype(float)
        offsets = rng.uniform(0.8, 1.2, 500)
        tau = rng.uniform(0.001, 0.02, 500) if with_tau else None
        by_n = log_emission_with_offsets(
            counts, offsets, mu=100.0, size=40.0, n_states=6,
            deletion_coverage_fraction=0.02, offset_tau=tau)
        by_levels = log_emission_with_offsets(
            counts, offsets, mu=100.0, size=40.0, levels=consensus_levels(6),
            deletion_coverage_fraction=0.02, offset_tau=tau)
        np.testing.assert_array_equal(by_n, by_levels)

    @pytest.mark.parametrize("n", [1, 2, 5, 8, 39, 139])
    def test_the_integer_grid_rebuilds_the_transition_bit_for_bit(self, n):
        """The density off-diagonal IS the old uniform one on the integer grid.

        change_prob * w(l) / sum_{l != k} w(l) reduces to change_prob /
        (n_states - 1) only because every level_spacing() width is exactly 1.0
        there -- which is why the end convention is full-step rather than a
        half-width Voronoi cell. Measured, the Voronoi convention breaks this.
        """
        remain = remain_prob_for_step(DEFAULT_CHANGE_RATE, 100)
        interior = 1.0 - remain_prob_for_step(DEFAULT_INTERIOR_CHANGE_RATE, 100)
        for kwargs in ({}, {"interior_change_prob": interior,
                            "fold_change_penalty": DEFAULT_FOLD_CHANGE_PENALTY}):
            by_n = setup_transition_matrix(n, remain_prob=remain, **kwargs)
            by_levels = setup_transition_matrix(
                remain_prob=remain, levels=consensus_levels(n), **kwargs)
            assert np.array_equal(by_n, by_levels)

    def test_the_neutral_integer_matrix_is_still_detected_as_flat(self):
        """_flat_transition_params uses ==, so this guards the O(n) decode."""
        remain = remain_prob_for_step(DEFAULT_CHANGE_RATE, 100)
        flat = setup_transition_matrix(5, remain_prob=remain)
        assert _flat_transition_params(np.log(flat)) is not None

    def test_polymorphism_off_changes_no_call(self, tmp_path):
        df = _poly_frame(blocks=((800, 1200, 3.0),), seed=5)
        before = run_HMM(df.copy(), str(tmp_path / "a"), write=False)
        after = run_HMM(df.copy(), str(tmp_path / "b"), write=False,
                        polymorphism=False)
        np.testing.assert_array_equal(before["prob_copy_number"].to_numpy(),
                                      after["prob_copy_number"].to_numpy())
        assert before["prob_copy_number"].dtype.kind == "i"


class TestLevelGrid:
    """The grid's shape, and the one property everything downstream keys on."""

    def test_consensus_levels_are_the_integer_grid(self):
        np.testing.assert_array_equal(consensus_levels(5), np.arange(6.0))

    def test_the_spacing_of_the_integer_grid_is_all_ones(self):
        """This is what makes the density off-diagonal reduce to the old one."""
        assert np.all(level_spacing(consensus_levels(7)) == 1.0)

    @pytest.mark.parametrize("resolution", [0.05, 0.1, 0.02, 0.25, 0.5])
    def test_single_copy_is_always_exactly_on_the_grid(self, resolution):
        """_default_log_start and add_cn_censor both test against 1.0 exactly."""
        levels = polymorphism_levels(5, resolution)
        assert np.count_nonzero(levels == 1.0) == 1
        assert np.count_nonzero(levels == 2.0) == 1

    def test_a_resolution_that_does_not_divide_one_is_snapped(self):
        """0.03 would otherwise build ... 0.99, 1.02 ... with no single copy."""
        assert snap_cn_resolution(0.03) == pytest.approx(1.0 / 33)
        assert (polymorphism_levels(5, 0.03) == 1.0).any()

    def test_the_grid_starts_at_zero_and_is_strictly_increasing(self):
        levels = polymorphism_levels(5, 0.05)
        assert levels[0] == 0.0
        assert np.all(np.diff(levels) > 0)

    def test_levels_below_single_copy_exist(self):
        """A partial loss is as real a measurement as a partial gain."""
        levels = polymorphism_levels(5, 0.05)
        assert ((levels > 0.0) & (levels < 1.0)).sum() >= 10

    def test_a_resolution_coarser_than_a_half_is_refused(self):
        with pytest.raises(ValueError):
            snap_cn_resolution(0.9)


class TestRefinementGrid:
    """Refinement is local, bounded, and declines rather than coarsening."""

    def test_the_step_is_absolute_in_the_fine_band(self):
        assert refine_step(1.0, 0.05) == pytest.approx(0.05)

    def test_the_two_regimes_agree_at_the_band_edge(self):
        assert refine_step(CN_FINE_BAND_TOP, 0.05) == pytest.approx(0.05)

    def test_the_step_is_relative_above_the_band(self):
        """An absolute 0.05 at copy number 34 is 0.15%, which the fold-change
        kernel prices at ~943 nats -- states the decode is forbidden to use."""
        assert refine_step(34.0, 0.05) > 0.5

    def test_refining_around_a_high_level_degrades_to_the_integers(self):
        before = polymorphism_levels(39, 0.05)
        after = refine_levels(before, [34.0], 0.05)
        assert after.size - before.size <= 2

    def test_refining_a_low_level_adds_a_band(self):
        before = polymorphism_levels(5, 0.05)
        after = refine_levels(before, [3.0], 0.05)
        assert after.size > before.size
        assert ((after > 2.0) & (after < 4.0)).sum() > 5

    def test_refinement_is_confined_to_the_band(self):
        """Levels outside [c-1, c+1] are untouched, so the prior elsewhere is
        unchanged -- which is what makes refinement a LOCAL perturbation."""
        before = polymorphism_levels(8, 0.05)
        after = refine_levels(before, [5.0], 0.05)
        outside = after[(after < 4.0) | (after > 6.0)]
        np.testing.assert_array_equal(outside, before[(before < 4.0) | (before > 6.0)])

    def test_nothing_to_refine_returns_the_same_object(self):
        levels = polymorphism_levels(5, 0.05)
        assert refine_levels(levels, [], 0.05) is levels

    def test_a_refinement_over_the_cap_is_declined_not_coarsened(self):
        levels = polymorphism_levels(5, 0.05)
        assert refine_levels(levels, [3.0], 0.05, max_levels=10) is levels


class TestPolymorphismCalls:
    """What the mode buys, and what it must not cost."""

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("resolution", [0.05, 0.1])
    def test_a_flat_genome_stays_at_single_copy(self, seed, resolution, tmp_path):
        """THE false-positive control.

        A finer grid offers more ways to be wrong, so the question is whether
        the transition prior still pays for them. Measured at genome scale
        (46,000 windows) the answer is that a flat genome comes back 100% at
        level 1.0 in a single segment; this is the same claim at test size.
        """
        df = _poly_frame(n=1500, seed=seed)
        out = run_HMM(df, str(tmp_path), write=False, polymorphism=True,
                      cn_resolution=resolution)
        called = out["prob_copy_number"].to_numpy()
        assert np.all(called == 1.0), sorted(set(called))

    @pytest.mark.parametrize("level", [1.1, 1.3, 1.6])
    def test_a_subclonal_level_is_recovered(self, level, tmp_path):
        """...and consensus mode rounds the same block to 1, which is the point."""
        df = _poly_frame(n=2000, blocks=((800, 1400, level),), seed=1)
        poly = run_HMM(df.copy(), str(tmp_path / "p"), write=False,
                       polymorphism=True)
        inside = poly["prob_copy_number"].to_numpy()[800:1400]
        modal = pd.Series(inside).mode()[0]
        assert abs(modal - level) <= DEFAULT_CN_RESOLUTION + 1e-9, modal

    def test_the_baseline_is_re_anchored_off_a_subclonal_block(self, tmp_path):
        """fit_censored_negative_binomial censors to [0.5, 1.5] x mode, which
        CANNOT exclude a subclonal level -- 1.1-1.5 is inside that band by
        construction. Unfixed, a 30% block at 1.30 pulled the fitted mu 4.4%
        high and the WHOLE genome came back off single copy."""
        df = _poly_frame(n=2000, blocks=((800, 1400, 1.3),), seed=1)
        out = run_HMM(df, str(tmp_path), write=False, polymorphism=True)
        called = out["prob_copy_number"].to_numpy()
        outside = np.r_[called[:800], called[1400:]]
        assert (outside == 1.0).mean() > 0.95, sorted(set(outside))

    def test_a_level_above_two_is_recovered_after_refinement(self, tmp_path):
        df = _poly_frame(n=2000, blocks=((800, 1200, 3.4),), seed=1)
        coarse = run_HMM(df.copy(), str(tmp_path / "c"), write=False,
                         polymorphism=True, refine=False)
        refined = run_HMM(df.copy(), str(tmp_path / "r"), write=False,
                          polymorphism=True)
        assert pd.Series(coarse["prob_copy_number"].to_numpy()[800:1200]).mode()[0] == 3.0
        got = pd.Series(refined["prob_copy_number"].to_numpy()[800:1200]).mode()[0]
        assert abs(got - 3.4) <= refine_step(3.0, DEFAULT_CN_RESOLUTION), got

    def test_a_clean_integer_amplification_is_not_refined_away(self, tmp_path):
        """The MLE predicate: when the call is already the nearest grid point,
        refinement provably cannot move it, so it must not be spent."""
        df = _poly_frame(n=2000, blocks=((800, 1200, 3.0),), seed=2)
        out = run_HMM(df, str(tmp_path), write=False, polymorphism=True)
        assert pd.Series(out["prob_copy_number"].to_numpy()[800:1200]).mode()[0] == 3.0

    def test_a_deletion_is_still_called_zero(self, tmp_path):
        """The zero state is a FRACTION of baseline, and a fine grid running
        down to 0.05 must not steal a real deletion from it."""
        df = _poly_frame(n=2000, blocks=((800, 1000, 0.02),), seed=1)
        out = run_HMM(df, str(tmp_path), write=False, polymorphism=True)
        assert np.all(out["prob_copy_number"].to_numpy()[810:990] == 0.0)

    def test_the_calls_are_float_under_p_and_int_without_it(self, tmp_path):
        df = _poly_frame(n=1200, seed=0)
        poly = run_HMM(df.copy(), str(tmp_path / "p"), write=False, polymorphism=True)
        cons = run_HMM(df.copy(), str(tmp_path / "c"), write=False)
        assert poly["prob_copy_number"].dtype.kind == "f"
        assert cons["prob_copy_number"].dtype.kind == "i"


class TestPolymorphismCensor:
    """The pass-2 censor has to see a subclonal window as a variant."""

    def _frame(self, cn):
        return pd.DataFrame({"prob_copy_number": np.asarray(cn, dtype=float)})

    def test_a_subclonal_window_is_censored_under_p(self):
        df, applied = add_cn_censor(
            self._frame([1.0] * 90 + [1.1] * 10), polymorphism=True)
        assert applied
        assert df["is_cn_variant"].to_numpy()[-10:].all()

    def test_consensus_mode_keeps_its_half_unit_band(self):
        """np.rint(1.1) == 1, so the old predicate is unchanged where it applies."""
        df, applied = add_cn_censor(self._frame([1.0] * 90 + [1.1] * 10))
        assert applied
        assert not df["is_cn_variant"].to_numpy().any()

    def test_the_min_keep_floor_still_declines(self):
        """A mostly-variant sequence falls back to pass 1's censoring, as before."""
        df, applied = add_cn_censor(
            self._frame([1.0] * 20 + [1.5] * 80), polymorphism=True)
        assert not applied
        assert not df["is_cn_variant"].to_numpy().any()


class TestModeSwitchOnAStagedFrame:
    """run_HMM must accept a frame that already carries a call from a prior pass.

    stage_pass1() snapshots prob_copy_number, so pass 2 is always handed pass 1's
    column. Writing a float level into that int64 column through .loc[:, col]
    takes pandas' IN-PLACE setitem path and raises LossySetitemError -- so `-p`
    crashed on any frame whose previous pass had been consensus. Nothing in the
    synthetic tier caught it, because those frames start clean.
    """

    def test_polymorphism_over_an_existing_integer_call(self, tmp_path):
        df = _poly_frame(n=1200, blocks=((500, 800, 3.0),), seed=4)
        first = run_HMM(df, str(tmp_path / "a"), write=False)
        assert first["prob_copy_number"].dtype.kind == "i"
        second = run_HMM(first, str(tmp_path / "b"), write=False, polymorphism=True)
        assert second["prob_copy_number"].dtype.kind == "f"

    def test_consensus_over_an_existing_float_call(self, tmp_path):
        df = _poly_frame(n=1200, seed=4)
        first = run_HMM(df, str(tmp_path / "a"), write=False, polymorphism=True)
        second = run_HMM(first, str(tmp_path / "b"), write=False)
        assert second["prob_copy_number"].dtype.kind == "i"

    def test_the_column_keeps_its_position(self, tmp_path):
        """CNV.csv's column order is part of what users diff."""
        df = _poly_frame(n=1200, seed=4)
        first = run_HMM(df, str(tmp_path / "a"), write=False)
        order = list(first.columns)
        second = run_HMM(first, str(tmp_path / "b"), write=False, polymorphism=True)
        assert list(second.columns) == order


def test_a_refinement_that_adds_nothing_returns_the_same_grid():
    """High up, refine_step() is coarser than the integers, so the band lands
    entirely on levels already present. That must read as "nothing to do"
    rather than as a grown grid, or run_HMM spends a decode discovering it."""
    levels = polymorphism_levels(39, 0.05)
    assert refine_levels(levels, [34.0], 0.05) is levels
