"""Cumulative GC skew, and the origin/terminus it predicts.

The prediction reads only `ref_base`, so every test here builds a coverage table
over a sequence it controls. Coverage is held flat throughout on purpose: if a
change ever makes the skew estimate depend on read depth, these tests go red.
"""
import json
import os

import numpy as np
import pytest

from CNery.core import (
    GC_SKEW_METHOD,
    plot_gc_skew,
    predict_ori_ter_from_skew,
    preprocess,
    read_coverage_table,
    write_gc_skew_results,
)

from conftest import write_coverage_table

WIN, STEP = 200, 100

# One replichore G-rich, the other its mirror image, so the skew switches sign
# at exactly two points. Both units are 6 bp with the same GC content, which
# keeps gc_percent flat across the join and leaves skew as the only signal.
LEADING, LAGGING = "GGGGCA", "CCCCGA"
# Long enough for the bootstrap to have power. At 24 kb this is 239 windows and
# ~21 blocks, which is where the p-value reaches its floor; a 6 kb toy gives
# only ~6 blocks and cannot clear 0.01 however clean its switch is. That is the
# honest statistical situation for a short sequence rather than a fixture quirk
# -- see TestBootstrap.
ARM_UNITS = 2000
ARM_BP = ARM_UNITS * len(LEADING)          # 12,000 bp per arm
SWITCH_SEQ = LEADING * ARM_UNITS + LAGGING * ARM_UNITS
SEQ_BP = len(SWITCH_SEQ)                   # 24,000 bp -> 239 windows


def _circular_delta(a, b, length):
    """Shortest distance between two coordinates on a circular genome.

    Needed because the two junctions of SWITCH_SEQ are the leading->lagging
    boundary in the middle and the lagging->leading boundary AT THE WRAP, and
    the latter is reported as the last window rather than the first -- they are
    the same locus on a circle.
    """
    d = abs(a - b) % length
    return min(d, length - d)


def _windows(tmp_path, seq, name="chrS", win=WIN, step=STEP):
    table = write_coverage_table(tmp_path / f"{name}.coverage.tsv", seq)
    df = preprocess(read_coverage_table(table), win=win, step=step, frag=150)
    df["genome_id"] = name
    return df


def _predict(tmp_path, seq, **kw):
    df = _windows(tmp_path, seq, **kw)
    win = kw.get("win", WIN)
    step = kw.get("step", STEP)
    return df, predict_ori_ter_from_skew(df, win=win, step=step)


class TestSkewColumns:
    def test_preprocess_keeps_its_historical_columns(self, tmp_path):
        df = _windows(tmp_path, SWITCH_SEQ)
        for col in ("win_st", "win_end", "win_len", "gc_percent",
                    "read_count_cov", "pct_redundant", "window_num",
                    "norm_raw_cov"):
            assert col in df.columns
        assert "gc_skew" in df.columns
        assert "cum_gc_skew" in df.columns

    def test_skew_is_a_bounded_ratio(self, tmp_path):
        df = _windows(tmp_path, SWITCH_SEQ)
        assert df["gc_skew"].between(-1.0, 1.0).all()

    def test_leading_and_lagging_arms_have_opposite_skew(self, tmp_path):
        df = _windows(tmp_path, SWITCH_SEQ)
        leading = df[df["win_end"] <= ARM_BP]["gc_skew"]
        lagging = df[df["win_st"] > ARM_BP]["gc_skew"]
        assert leading.mean() > 0
        assert lagging.mean() < 0

    def test_cumulative_curve_is_finite_and_returns_to_zero(self, tmp_path):
        # The mean is subtracted before cumulating, so the running sum must end
        # at 0. That is the property rotation invariance rests on.
        df = _windows(tmp_path, SWITCH_SEQ)
        assert np.isfinite(df["cum_gc_skew"]).all()
        assert df["cum_gc_skew"].iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_window_with_no_g_or_c_scores_zero_not_nan(self, tmp_path):
        df = _windows(tmp_path, "AT" * 1500)
        assert (df["gc_skew"] == 0.0).all()
        assert np.isfinite(df["cum_gc_skew"]).all()


class TestPrediction:
    def test_finds_the_designed_switch_points(self, tmp_path):
        # Origin = global minimum of the cumulative curve, terminus = its
        # maximum. The running sum rises across the G-rich arm and falls back
        # across the C-rich one, so the maximum lands on the leading->lagging
        # join at ARM_BP and the minimum on the lagging->leading join, which for
        # this sequence is the wrap between the last base and position 1.
        _, result = _predict(tmp_path, SWITCH_SEQ)
        assert _circular_delta(result["Origin (bp)"], 1, SEQ_BP) <= WIN
        assert _circular_delta(result["Terminus (bp)"], ARM_BP, SEQ_BP) <= WIN

    def test_designed_switch_is_called_confident(self, tmp_path):
        _, result = _predict(tmp_path, SWITCH_SEQ)
        assert result["Prediction confident"] is True
        assert result["Separation (fraction of genome)"] == pytest.approx(0.5, abs=0.02)

    def test_reports_the_documented_keys(self, tmp_path):
        _, result = _predict(tmp_path, SWITCH_SEQ)
        assert set(result) == {
            "Origin (bp)", "Terminus (bp)",
            "Origin window index", "Terminus window index",
            "Windows", "Separation (fraction of genome)",
            "Cumulative skew amplitude", "Replichore skew t-statistic",
            "Replichore skew p-value", "Bootstrap surrogates",
            "Prediction confident", "Prediction method",
        }
        assert result["Prediction method"] == GC_SKEW_METHOD

    def test_prediction_is_independent_of_coverage_depth(self, tmp_path):
        # Same sequence, 40x the reads. Skew is a property of the reference.
        deep = write_coverage_table(
            tmp_path / "deep.coverage.tsv", SWITCH_SEQ, cov=1000
        )
        df_deep = preprocess(read_coverage_table(deep), win=WIN, step=STEP, frag=150)
        _, shallow = _predict(tmp_path, SWITCH_SEQ)
        deep_result = predict_ori_ter_from_skew(df_deep, win=WIN, step=STEP)
        assert deep_result["Origin (bp)"] == shallow["Origin (bp)"]
        assert deep_result["Terminus (bp)"] == shallow["Terminus (bp)"]

    def test_empty_frame_is_rejected_by_name(self, tmp_path):
        df = _windows(tmp_path, SWITCH_SEQ).iloc[0:0]
        with pytest.raises(ValueError, match="empty window frame"):
            predict_ori_ter_from_skew(df, win=WIN, step=STEP)


class TestRotationInvariance:
    """Subtracting the mean before cumulating is what buys this.

    A circularly permuted copy of a genome is the same genome, so it must
    predict the same locus. Without the mean subtraction the running sum ends at
    n*mean(skew) instead of 0, the wraparound contributes a linear ramp, and the
    extrema slide with the choice of coordinate 1.
    """

    @pytest.mark.parametrize("shift", [600, 1800, 3000, 4700])
    def test_permuted_reference_predicts_the_same_locus(self, tmp_path, shift):
        rotated = SWITCH_SEQ[shift:] + SWITCH_SEQ[:shift]

        _, base = _predict(tmp_path, SWITCH_SEQ, name="orig")
        _, moved = _predict(tmp_path, rotated, name="rot")

        for key in ("Origin (bp)", "Terminus (bp)"):
            mapped = moved[key] + shift
            delta = _circular_delta(mapped, base[key], SEQ_BP)
            assert delta <= WIN, (
                f"{key}: permuted reference maps to {mapped % SEQ_BP}, "
                f"original says {base[key]} ({delta} bp apart)"
            )


class TestConfidenceGate:
    """The gate flags a prediction; it never hides one.

    Every case below must still report real coordinates, so a rejected call can
    be diagnosed from the JSON rather than reduced to a boolean.
    """

    def test_zero_skew_sequence_is_not_confident(self, tmp_path):
        _, result = _predict(tmp_path, "AT" * 1500)
        assert result["Prediction confident"] is False
        assert result["Cumulative skew amplitude"] == 0.0

    def test_zero_skew_sequence_does_not_warn_or_raise(self, tmp_path):
        with np.errstate(all="raise"):
            _, result = _predict(tmp_path, "AT" * 1500)
        assert np.isfinite(result["Replichore skew t-statistic"])

    def test_too_few_windows_is_not_confident(self, tmp_path):
        # A handful of windows, but a perfectly clean skew switch. There is no
        # minimum-window gate any more: the bootstrap has to reject this on its
        # own, which is the point of dropping that constant.
        short = LEADING * 100 + LAGGING * 100
        _, result = _predict(tmp_path, short, win=600, step=120)
        assert result["Windows"] < 10
        assert result["Prediction confident"] is False
        assert result["Replichore skew p-value"] > 0.01

    def test_adjacent_extrema_are_not_confident(self, tmp_path):
        # A single G-rich patch in an otherwise skew-free genome puts the two
        # extrema close together rather than antipodally, failing the 35-65%
        # separation band.
        seq = "AT" * 1400 + "GGGGCA" * 40 + "AT" * 1400
        _, result = _predict(tmp_path, seq)
        assert result["Separation (fraction of genome)"] < 0.35
        assert result["Prediction confident"] is False

    def test_rejected_prediction_still_reports_a_p_value(self, tmp_path):
        # Not just the coordinates -- the evidence too, so the rejection can be
        # read rather than taken on trust.
        _, result = _predict(tmp_path, "AT" * 1500)
        assert result["Replichore skew p-value"] == pytest.approx(1.0)
        assert result["Prediction confident"] is False

    def test_rejected_prediction_still_reports_coordinates(self, tmp_path):
        _, result = _predict(tmp_path, "AT" * 1500)
        assert isinstance(result["Origin (bp)"], int)
        assert isinstance(result["Terminus (bp)"], int)


class TestBootstrap:
    """The circular block bootstrap behind the p-value.

    A plain t-test would be indefensible here: skew is spatially autocorrelated,
    so the t statistic's magnitude is inflated by an unknown factor. The
    bootstrap builds the null by resampling BLOCKS -- preserving local structure
    while destroying the global two-arm pattern -- and re-runs the whole
    procedure, extrema search included, on every surrogate so that choosing the
    breakpoints from the data is paid for rather than ignored.
    """

    def test_p_value_is_a_probability(self, tmp_path):
        _, result = _predict(tmp_path, SWITCH_SEQ)
        p = result["Replichore skew p-value"]
        assert 0.0 < p <= 1.0

    def test_p_value_floors_at_one_over_b_plus_one(self, tmp_path):
        # A clean switch exhausts every surrogate, so p reads back exactly its
        # floor. This is an upper bound, not a measurement -- which is why
        # "Bootstrap surrogates" is reported alongside it.
        df = _windows(tmp_path, SWITCH_SEQ)
        result = predict_ori_ter_from_skew(
            df, win=WIN, step=STEP, n_surrogates=200
        )
        assert result["Bootstrap surrogates"] == 200
        assert result["Replichore skew p-value"] == round(1 / 201, 5)

    def test_more_surrogates_lowers_the_floor(self, tmp_path):
        df = _windows(tmp_path, SWITCH_SEQ)
        coarse = predict_ori_ter_from_skew(df, win=WIN, step=STEP, n_surrogates=100)
        fine = predict_ori_ter_from_skew(df, win=WIN, step=STEP, n_surrogates=500)
        assert fine["Replichore skew p-value"] < coarse["Replichore skew p-value"]

    def test_is_deterministic_for_a_fixed_seed(self, tmp_path):
        # Goldens depend on this.
        df = _windows(tmp_path, SWITCH_SEQ)
        runs = [predict_ori_ter_from_skew(df, win=WIN, step=STEP) for _ in range(2)]
        assert runs[0] == runs[1]

    def test_seed_changes_only_the_p_value(self, tmp_path):
        df = _windows(tmp_path, SWITCH_SEQ)
        a = predict_ori_ter_from_skew(df, win=WIN, step=STEP, seed=1)
        b = predict_ori_ter_from_skew(df, win=WIN, step=STEP, seed=2)
        for key in ("Origin (bp)", "Terminus (bp)",
                    "Replichore skew t-statistic", "Windows"):
            assert a[key] == b[key], key

    @pytest.mark.parametrize("block", [5, 10, 20])
    def test_verdict_is_insensitive_to_block_length(self, tmp_path, block):
        # Block length is a modelling choice, so the conclusion must not ride on
        # it -- across every length that still leaves enough blocks (>=12 at
        # this sequence's 239 windows), the verdict is the same.
        df = _windows(tmp_path, SWITCH_SEQ)
        result = predict_ori_ter_from_skew(df, win=WIN, step=STEP, block=block)
        assert result["Prediction confident"] is True

    def test_power_is_lost_when_blocks_get_too_few(self, tmp_path):
        """What DOES matter is the number of blocks, not their length.

        Pinning this because it is the reason SKEW_TARGET_BLOCKS exists and the
        reason the default block length is adaptive rather than a constant: with
        only a handful of blocks, reshuffling them reassembles a two-arm pattern
        often enough that even a perfectly clean switch stops being significant.
        """
        df = _windows(tmp_path, SWITCH_SEQ)
        n = len(df)
        many = predict_ori_ter_from_skew(df, win=WIN, step=STEP, block=n // 24)
        few = predict_ori_ter_from_skew(df, win=WIN, step=STEP, block=n // 4)
        assert many["Replichore skew p-value"] < few["Replichore skew p-value"]
        assert many["Prediction confident"] is True
        assert few["Prediction confident"] is False

    def test_block_length_is_clamped_to_the_sequence(self, tmp_path):
        # block=10_000 on a 59-window frame must not produce a degenerate
        # resample or divide by zero; it is clamped to n // 4.
        df = _windows(tmp_path, SWITCH_SEQ)
        result = predict_ori_ter_from_skew(df, win=WIN, step=STEP, block=10_000)
        assert 0.0 < result["Replichore skew p-value"] <= 1.0

    def test_too_short_to_bootstrap_reports_no_surrogates(self, tmp_path):
        # Under 8 windows there is nothing to resample. Rather than invent a
        # null, it declines: p = 1.0 and a surrogate count of 0 that says so.
        short = LEADING * 100 + LAGGING * 100
        _, result = _predict(tmp_path, short, win=600, step=120)
        assert result["Bootstrap surrogates"] == 0
        assert result["Replichore skew p-value"] == pytest.approx(1.0)


class TestArtifacts:
    def test_json_is_written_and_parses(self, tmp_path):
        out = str(tmp_path / "out")
        _, result = _predict(tmp_path, SWITCH_SEQ)
        path = write_gc_skew_results(result, out, "chrS")

        assert os.path.dirname(path) == os.path.join(out, "GC_skew")
        assert os.path.basename(path).endswith("chrS_gc_skew_results.json")
        with open(path) as fh:
            assert json.load(fh) == result

    def test_filename_does_not_depend_on_a_trailing_slash(self, tmp_path):
        # samplename is the output prefix's last path component + genome_id, via
        # sample_prefix(), which RSTRIPS the separator. It used to strip()
        # instead, so CNery's default out_dir of "CNV_out/" split to the empty
        # string and this file came out named for the sequence alone -- beside a
        # CNV_csv/CNV_outchrS_CNV.csv from the same run, because run_HMM already
        # rstripped. breseq reads the JSON by name, so the name must not depend
        # on how the caller spelled -o. Same rule as *_otr_results.json.
        _, result = _predict(tmp_path, SWITCH_SEQ)
        with_slash = write_gc_skew_results(
            result, str(tmp_path / "out") + "/", "chrS")
        without = write_gc_skew_results(result, str(tmp_path / "out"), "chrS")

        assert os.path.basename(with_slash) == "outchrS_gc_skew_results.json"
        assert os.path.basename(with_slash) == os.path.basename(without)

    def test_json_is_strict_rfc_json(self, tmp_path):
        # Unlike the OTR results file, which emits a bare NaN.
        out = str(tmp_path / "out")
        _, result = _predict(tmp_path, SWITCH_SEQ)
        path = write_gc_skew_results(result, out, "chrS")
        with open(path) as fh:
            json.loads(fh.read(), parse_constant=_no_constants)

    def test_writer_creates_its_own_directory(self, tmp_path):
        out = str(tmp_path / "never_made")
        _, result = _predict(tmp_path, SWITCH_SEQ)
        write_gc_skew_results(result, out, "chrS")
        assert os.path.isdir(os.path.join(out, "GC_skew"))

    def test_plot_is_written(self, tmp_path):
        out = str(tmp_path / "out")
        df, result = _predict(tmp_path, SWITCH_SEQ)
        path = plot_gc_skew(df, out, result)
        assert os.path.exists(path)
        assert path.endswith("chrS_GC_skew.pdf")

    def test_plot_is_written_for_a_rejected_prediction(self, tmp_path):
        out = str(tmp_path / "out")
        df, result = _predict(tmp_path, "AT" * 1500)
        assert result["Prediction confident"] is False
        assert os.path.exists(plot_gc_skew(df, out, result))


def _no_constants(name):
    raise AssertionError(f"non-RFC JSON constant in output: {name}")
