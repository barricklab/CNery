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
ARM_UNITS = 500
ARM_BP = ARM_UNITS * len(LEADING)          # 3000 bp per arm
SWITCH_SEQ = LEADING * ARM_UNITS + LAGGING * ARM_UNITS
SEQ_BP = len(SWITCH_SEQ)                   # 6000 bp -> 59 windows, over min_windows


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
        # 10 windows, well under min_windows, but a clean skew switch: rejected
        # on window count alone.
        short = LEADING * 100 + LAGGING * 100
        _, result = _predict(tmp_path, short, win=600, step=120)
        assert result["Windows"] < 50
        assert result["Prediction confident"] is False

    def test_adjacent_extrema_are_not_confident(self, tmp_path):
        # A single G-rich patch in an otherwise skew-free genome puts the two
        # extrema close together rather than antipodally, failing the 35-65%
        # separation band.
        seq = "AT" * 1400 + "GGGGCA" * 40 + "AT" * 1400
        _, result = _predict(tmp_path, seq)
        assert result["Separation (fraction of genome)"] < 0.35
        assert result["Prediction confident"] is False

    def test_rejected_prediction_still_reports_coordinates(self, tmp_path):
        _, result = _predict(tmp_path, "AT" * 1500)
        assert isinstance(result["Origin (bp)"], int)
        assert isinstance(result["Terminus (bp)"], int)


class TestArtifacts:
    def test_json_is_written_and_parses(self, tmp_path):
        out = str(tmp_path / "out")
        _, result = _predict(tmp_path, SWITCH_SEQ)
        path = write_gc_skew_results(result, out, "chrS")

        assert os.path.dirname(path) == os.path.join(out, "GC_skew")
        assert os.path.basename(path).endswith("chrS_gc_skew_results.json")
        with open(path) as fh:
            assert json.load(fh) == result

    def test_filename_follows_the_otr_results_convention(self, tmp_path):
        # samplename is the output prefix's last path component + genome_id, so
        # the trailing slash CNery's default out_dir carries leaves the prefix
        # empty and the file named for the sequence alone. Same rule as
        # apply_otr_correction's *_otr_results.json.
        _, result = _predict(tmp_path, SWITCH_SEQ)
        path = write_gc_skew_results(result, str(tmp_path / "out") + "/", "chrS")
        assert os.path.basename(path) == "chrS_gc_skew_results.json"

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
