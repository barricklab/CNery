"""End-to-end runs through main(), driving the real argparse surface.

Nothing exercised the command line before, which mattered less when the CLI was a thin
wrapper over a fixed pair of BAM/FASTA paths. Inputs are now the whole interface: which
files get picked up, what they are called, and what a bad invocation leaves behind.
"""

import json
import os
import sys

import pytest

from CNery.get_CNV import main

COLUMNS = [
    "position", "ref_base",
    "unique_top_cov", "unique_bot_cov",
    "redundant_top_cov", "redundant_bot_cov",
]
TOTAL_ONLY_COLUMNS = ["position", "ref_base", "unique_cov", "redundant_cov", "total_cov"]

# Enough windows at -w 100 -s 50 for the HMM and the OTR fit to have something to chew on.
SEQ = ("ACGTACGGCTAA" * 250)


def _render(header, rows, n, d):
    lines = [d.join(header)]
    lines += [d.join(str(v) for v in row) for row in rows]
    lines.append(d.join(("#", "number_of_positions", str(n))))
    return "\n".join(lines) + "\n"


def _write_table(path, seq=SEQ, cov=25, d="\t"):
    rows = [(i + 1, base, cov, cov, 0, 0) for i, base in enumerate(seq)]
    path.write_text(_render(COLUMNS, rows, len(seq), d))
    return path


def _write_total_only(path, seq=SEQ, cov=25, d=","):
    """The same coverage in the shape `bam2cov --total-only` writes."""
    rows = [(i + 1, base, cov * 2, 0, cov * 2) for i, base in enumerate(seq)]
    path.write_text(_render(TOTAL_ONLY_COLUMNS, rows, len(seq), d))
    return path


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["CNery"] + argv)
    return main()


def _csv_names(out):
    return sorted(os.listdir(os.path.join(out, "CNV_csv")))


class TestPositionalInputs:
    def test_single_file(self, tmp_path, monkeypatch):
        table = _write_table(tmp_path / "chrA.coverage.tsv")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(table), "-o", out, "-w", "100", "-s", "50"])

        assert any("chrA" in n for n in _csv_names(out))
        assert os.listdir(os.path.join(out, "CNV_plt"))

    def test_several_files_are_analyzed_together(self, tmp_path, monkeypatch):
        a = _write_table(tmp_path / "chrA.coverage.tsv")
        b = _write_table(tmp_path / "chrB.coverage.tsv")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(a), str(b), "-o", out, "-w", "100", "-s", "50"])

        names = _csv_names(out)
        assert any("chrA" in n for n in names)
        assert any("chrB" in n for n in names)
        # Both GC files are POOLED across the two references, not one per
        # reference -- that is what this counts. Two files, not one, because the
        # correction has two passes: the fit on raw coverage, and the refit after
        # OTR that removes the GC trend the position-dependent tent puts back.
        # Both are fitted once across every table in the run.
        gc_files = sorted(os.listdir(os.path.join(out, "GC_bias")))
        assert len(gc_files) == 2, gc_files
        assert sum("GC_passes" in n for n in gc_files) == 1
        # The giveaway that they are pooled: one file naming BOTH references.
        assert all("chrA_and_chrB" in n for n in gc_files), gc_files

    def test_directory_is_expanded(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "chrA.coverage.tsv")
        _write_table(cov / "chrB.coverage.tsv")
        (cov / "reference.fasta").write_text(">chrA\nACGT\n")   # must be ignored
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "-o", out, "-w", "100", "-s", "50"])

        names = _csv_names(out)
        assert any("chrA" in n for n in names)
        assert any("chrB" in n for n in names)

    def test_defaults_to_the_current_folder(self, tmp_path, monkeypatch):
        _write_table(tmp_path / "chrA.coverage.tsv")
        monkeypatch.chdir(tmp_path)
        out = str(tmp_path / "out")
        _run(monkeypatch, ["-o", out, "-w", "100", "-s", "50"])

        assert any("chrA" in n for n in _csv_names(out))


class TestFormats:
    """All four shapes of the same table run end to end without being declared."""

    def test_csv_folder(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "chrA.coverage.csv", d=",")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "-o", out, "-w", "100", "-s", "50"])

        assert any("chrA" in n for n in _csv_names(out))

    def test_total_only_csv_folder(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_total_only(cov / "chrA.coverage.csv")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "-o", out, "-w", "100", "-s", "50"])

        assert any("chrA" in n for n in _csv_names(out))

    def test_csv_and_tsv_together(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "chrA.coverage.csv", d=",")
        _write_table(cov / "chrB.coverage.tsv", d="\t")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "-o", out, "-w", "100", "-s", "50"])

        names = _csv_names(out)
        assert any("chrA" in n for n in names)
        assert any("chrB" in n for n in names)

    def test_total_only_matches_strand_split_end_to_end(self, tmp_path, monkeypatch):
        # The two spellings of one dataset must produce the same calls, not merely both
        # produce output.
        strand = tmp_path / "strand"
        total = tmp_path / "total"
        strand.mkdir()
        total.mkdir()
        _write_table(strand / "chrA.coverage.tsv", d="\t")
        _write_total_only(total / "chrA.coverage.csv")

        outs = {}
        for name, folder in (("s", strand), ("t", total)):
            out = tmp_path / f"out_{name}"
            _run(monkeypatch, [str(folder), "-o", str(out), "-w", "100", "-s", "50"])
            produced = [n for n in _csv_names(str(out)) if n.endswith("_CNV.csv")]
            outs[name] = (out / "CNV_csv" / produced[0]).read_text()

        assert outs["s"] == outs["t"]


class TestFileEndingFlag:
    def test_override_matches_a_different_name(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "chrA.cov.txt")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "--file-ending", "cov.txt",
                           "-o", out, "-w", "100", "-s", "50"])

        assert any("chrA" in n for n in _csv_names(out))

    def test_override_replaces_the_default(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "chrA.cov.txt")
        _write_table(cov / "chrB.coverage.tsv")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "--file-ending", "cov.txt",
                           "-o", out, "-w", "100", "-s", "50"])

        names = _csv_names(out)
        assert any("chrA" in n for n in names)
        assert not any("chrB" in n for n in names)

    def test_flag_repeats_to_accept_several(self, tmp_path, monkeypatch):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "chrA.cov.txt")
        _write_table(cov / "chrB.coverage.tsv")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov),
                           "--file-ending", "cov.txt",
                           "--file-ending", "coverage.tsv",
                           "-o", out, "-w", "100", "-s", "50"])

        names = _csv_names(out)
        assert any("chrA" in n for n in names)
        assert any("chrB" in n for n in names)

    def test_positional_after_the_flag_is_not_swallowed(self, tmp_path, monkeypatch):
        # The reason --file-ending appends instead of taking nargs="+": a greedy flag
        # would eat the directory argument that follows it.
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "chrA.cov.txt")
        out = str(tmp_path / "out")
        _run(monkeypatch, ["--file-ending", "cov.txt", str(cov),
                           "-o", out, "-w", "100", "-s", "50"])

        assert any("chrA" in n for n in _csv_names(out))


class TestBadInvocations:
    def test_missing_path_creates_no_output(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        with pytest.raises(FileNotFoundError):
            _run(monkeypatch, [str(tmp_path / "nope"), "-o", str(out)])
        assert not out.exists()

    def test_folder_without_tables_creates_no_output(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        out = tmp_path / "out"
        with pytest.raises(FileNotFoundError):
            _run(monkeypatch, [str(empty), "-o", str(out)])
        assert not out.exists()

    def test_duplicate_ids_create_no_output(self, tmp_path, monkeypatch):
        a = tmp_path / "runA"
        b = tmp_path / "runB"
        a.mkdir()
        b.mkdir()
        _write_table(a / "chrA.coverage.tsv")
        _write_table(b / "chrA.coverage.tsv")
        out = tmp_path / "out"
        with pytest.raises(ValueError):
            _run(monkeypatch, [str(a), str(b), "-o", str(out)])
        assert not out.exists()

    def test_removed_flags_are_rejected(self, tmp_path, monkeypatch):
        table = _write_table(tmp_path / "chrA.coverage.tsv")
        for flag in ("-ref", "-i", "-reg", "--frag_size"):
            with pytest.raises(SystemExit):
                _run(monkeypatch, [str(table), flag, "whatever"])


class TestFlagSpellings:
    """The documented spelling of every flag must actually parse.

    README once showed `--region`, which the parser did not accept -- it only had
    `-reg` -- so the example failed with 'unrecognized arguments'. Nothing caught
    that, because nothing exercised the flags.
    """

    @pytest.mark.parametrize("flag", [
        "--file-ending", "--region", "-o", "--output", "-w", "--window",
        "-s", "--step-size", "-f", "--frag-size",
        "-z", "--deletion-coverage-fraction", "--bias",
    ])
    def test_flag_is_accepted(self, flag, tmp_path, monkeypatch):
        parser_args = {
            "--file-ending": "coverage.tsv", "--region": "100-2000",
            "--bias": "none",
            # A fraction of baseline now, so the generic "100" would ask the
            # zero state to expect 100x the single-copy level.
            "-z": "0.05", "--deletion-coverage-fraction": "0.05",
        }
        value = parser_args.get(flag, "1000" if flag in ("-w", "--window") else None)
        if value is None:
            value = {"-o": str(tmp_path / "out"), "--output": str(tmp_path / "out")}.get(
                flag, "100"
            )
        table = _write_table(tmp_path / "chrA.coverage.tsv")
        out = str(tmp_path / "out")
        # Only asserting the parser accepts it; a run is the cheapest way to be sure.
        _run(monkeypatch, [str(table), flag, value, "-o", out, "-w", "100", "-s", "50"])


class TestRegion:
    """--region crops the plot only, and a bad one must fail loudly."""

    def test_valid_region_runs(self, tmp_path, monkeypatch):
        table = _write_table(tmp_path / "chrA.coverage.tsv")
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(table), "--region", "500-2000",
                           "-o", out, "-w", "100", "-s", "50"])
        assert any("chrA" in n for n in _csv_names(out))

    def test_csv_still_holds_every_window(self, tmp_path, monkeypatch):
        # The region restricts the PLOT; calls and CSVs stay genome-wide.
        table = _write_table(tmp_path / "chrA.coverage.tsv")
        outs = {}
        for label, argv in (("full", []), ("cropped", ["--region", "500-2000"])):
            out = tmp_path / label
            _run(monkeypatch, [str(table)] + argv
                 + ["-o", str(out), "-w", "100", "-s", "50"])
            produced = [n for n in _csv_names(str(out)) if n.endswith("_CNV.csv")]
            outs[label] = len((out / "CNV_csv" / produced[0]).read_text().splitlines())
        assert outs["full"] == outs["cropped"]

    @pytest.mark.parametrize("bad", [
        "1-2-3",            # too many parts
        "abc-def",          # not numbers
        "100",              # no separator
        "2000-500",         # backwards
        ":100-200",         # colon but no sequence ID
        "chrA:1-2-3",       # qualified, still malformed
    ])
    def test_invalid_region_exits_nonzero(self, bad, tmp_path, monkeypatch):
        # It used to `return` a message from main(), exiting 0 having done nothing --
        # which reads as success.
        table = _write_table(tmp_path / "chrA.coverage.tsv")
        with pytest.raises(SystemExit) as excinfo:
            _run(monkeypatch, [str(table), "--region", bad,
                               "-o", str(tmp_path / "out"), "-w", "100", "-s", "50"])
        assert excinfo.value.code != 0

    def test_open_intervals(self, tmp_path, monkeypatch):
        table = _write_table(tmp_path / "chrA.coverage.tsv")
        for i, region in enumerate(("500-", "-2000", "chrA:500-", "chrA:-2000")):
            _run(monkeypatch, [str(table), "--region", region,
                               "-o", str(tmp_path / f"out{i}"), "-w", "100", "-s", "50"])

    def test_bad_region_creates_no_output(self, tmp_path, monkeypatch):
        # Validated alongside the inputs, before any directory is made.
        table = _write_table(tmp_path / "chrA.coverage.tsv")
        out = tmp_path / "out"
        with pytest.raises(SystemExit):
            _run(monkeypatch, [str(table), "--region", "nonsense", "-o", str(out)])
        assert not out.exists()


class TestRegionWithSeveralSequences:
    """A region names the sequence it crops; the rest are plotted whole.

    One coordinate range applied to every reference is what used to crash a
    chromosome-plus-plasmid run: the range falls outside the shorter sequence, the
    plot slice comes out empty, and its median is NaN.
    """

    @staticmethod
    def _two(tmp_path):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "bigChrom.coverage.tsv", seq=SEQ)
        _write_table(cov / "tinyPlasmid.coverage.tsv", seq="ACGTACGGCTAA" * 30)
        return cov

    def test_region_selects_which_sequences_are_plotted(self, tmp_path, monkeypatch):
        cov = self._two(tmp_path)
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "--region", "bigChrom:800-2000",
                           "-o", out, "-w", "100", "-s", "50"])

        plots = sorted(os.listdir(os.path.join(out, "CNV_plt")))
        assert any("bigChrom" in n for n in plots)
        assert not any("tinyPlasmid" in n for n in plots), (
            "a sequence not named in --region must not be plotted"
        )

    def test_unplotted_sequences_are_still_called(self, tmp_path, monkeypatch):
        # --region scopes plotting only; the analysis still covers every sequence.
        cov = self._two(tmp_path)
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "--region", "bigChrom:800-2000",
                           "-o", out, "-w", "100", "-s", "50"])

        csvs = sorted(os.listdir(os.path.join(out, "CNV_csv")))
        assert any("bigChrom" in n for n in csvs)
        assert any("tinyPlasmid" in n for n in csvs)

    def test_several_regions_plot_several_sequences(self, tmp_path, monkeypatch):
        cov = self._two(tmp_path)
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov),
                           "--region", "bigChrom:800-2000",
                           "--region", "tinyPlasmid:100-300",
                           "-o", out, "-w", "100", "-s", "50"])

        plots = sorted(os.listdir(os.path.join(out, "CNV_plt")))
        assert any("bigChrom" in n for n in plots)
        assert any("tinyPlasmid" in n for n in plots)

    def test_repeated_region_for_one_sequence_is_rejected(self, tmp_path, monkeypatch):
        # One plot per sequence, so it cannot carry two ranges.
        cov = self._two(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            _run(monkeypatch, [str(cov),
                               "--region", "bigChrom:100-200",
                               "--region", "bigChrom:800-2000",
                               "-o", str(tmp_path / "out"), "-w", "100", "-s", "50"])
        assert excinfo.value.code != 0

    def test_bare_region_is_rejected_when_ambiguous(self, tmp_path, monkeypatch):
        cov = self._two(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            _run(monkeypatch, [str(cov), "--region", "800-2000",
                               "-o", str(tmp_path / "out"), "-w", "100", "-s", "50"])
        assert excinfo.value.code != 0

    def test_unknown_sequence_is_rejected(self, tmp_path, monkeypatch):
        cov = self._two(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            _run(monkeypatch, [str(cov), "--region", "nosuchseq:800-2000",
                               "-o", str(tmp_path / "out"), "-w", "100", "-s", "50"])
        assert excinfo.value.code != 0

    def test_region_outside_the_named_sequence_does_not_crash(self, tmp_path, monkeypatch):
        # find_nearest() clamps, so an out-of-range region slices to nothing; plot_copy
        # falls back to the whole sequence rather than dividing by a NaN median.
        cov = self._two(tmp_path)
        out = str(tmp_path / "out")
        _run(monkeypatch, [str(cov), "--region", "tinyPlasmid:900000-999000",
                           "-o", out, "-w", "100", "-s", "50"])
        assert any("tinyPlasmid" in n for n in os.listdir(os.path.join(out, "CNV_plt")))


def _reject(value):
    raise AssertionError(f"not strict JSON: {value}")


def _otr_json(out, seq_id):
    """The OTR record breseq reads for one sequence, parsed strictly.

    parse_constant refuses NaN/Infinity: breseq reads this with nlohmann, which
    is strict RFC JSON, and a single bare NaN costs it the whole file.
    """
    path = os.path.join(out, "OTR_corr",
                        f"{os.path.basename(out)}{seq_id}_otr_results.json")
    with open(path) as fh:
        return json.loads(fh.read(), parse_constant=_reject)


class TestDegenerateCoverage:
    """A reference with nothing to measure is a RESULT, not a crash.

    Each of these took the whole run down before, with a bare library exception
    naming no file: a header-only table raised IndexError from preprocess, an
    all-zero one ValueError from np.interp inside fit_gc_bias, and a healthy
    chromosome beside a zero-coverage plasmid ZeroDivisionError from solve_pr --
    the realistic one, since a plasmid that got no reads is an ordinary event.

    What every case must produce is the OTR results JSON. breseq reads it by
    name, and a missing file costs it exactly what an unparseable one does.
    """

    def _run_ok(self, monkeypatch, table_dir, out):
        _run(monkeypatch, [str(table_dir), "-o", out, "-w", "100", "-s", "100"])

    def test_header_only_table(self, tmp_path, monkeypatch):
        d = tmp_path / "in"
        d.mkdir()
        _write_table(d / "empty.coverage.tsv", seq="")
        out = str(tmp_path / "out")

        self._run_ok(monkeypatch, d, out)

        data = _otr_json(out, "empty")
        assert data["Origin-to-Terminus/Bias Ratio"] == "Not detected"
        # breseq does not type-check these two, so a null is as damaging as a
        # missing file. There is no coordinate to report, but there is an int.
        assert isinstance(data["Origin window"], int)
        assert isinstance(data["Terminus window"], int)
        assert data["No usable coverage reason"] == "the coverage table has no position rows"

    def test_all_zero_coverage(self, tmp_path, monkeypatch):
        d = tmp_path / "in"
        d.mkdir()
        _write_table(d / "flat.coverage.tsv", cov=0)
        out = str(tmp_path / "out")

        self._run_ok(monkeypatch, d, out)

        data = _otr_json(out, "flat")
        assert data["Origin-to-Terminus/Bias Ratio"] == "Not detected"
        assert data["No usable coverage reason"] == "every window has zero coverage"

    def test_every_window_is_a_repeat(self, tmp_path, monkeypatch):
        # NOT degenerate: an all-repeat replicon has real coverage and gets real
        # calls. Only fit_gc_bias could not survive it, and it now declines to an
        # identity curve instead of raising.
        d = tmp_path / "in"
        d.mkdir()
        rows = [(i + 1, b, 0, 0, 12, 12) for i, b in enumerate(SEQ)]
        (d / "rep.coverage.tsv").write_text(_render(COLUMNS, rows, len(SEQ), "\t"))
        out = str(tmp_path / "out")

        self._run_ok(monkeypatch, d, out)

        assert _otr_json(out, "rep")["No usable coverage reason"] is None
        assert any("rep" in n for n in _csv_names(out))

    def test_zero_plasmid_does_not_take_down_the_chromosome(self, tmp_path, monkeypatch):
        """The case that will actually happen: a plasmid that got no reads."""
        d = tmp_path / "in"
        d.mkdir()
        _write_table(d / "chrom.coverage.tsv")
        _write_table(d / "plasmid.coverage.tsv", cov=0)
        out = str(tmp_path / "out")

        self._run_ok(monkeypatch, d, out)

        # Both sequences get a file, and the healthy one is unaffected.
        assert _otr_json(out, "plasmid")["No usable coverage reason"] is not None
        assert _otr_json(out, "chrom")["No usable coverage reason"] is None
        assert any("chrom" in n and n.endswith("_CNV.csv") for n in _csv_names(out))
        assert any("plasmid" in n and n.endswith("_CNV.csv") for n in _csv_names(out))

    def test_a_dead_plasmid_does_not_null_the_chromosomes_copy_number(
            self, tmp_path, monkeypatch):
        # relative_copy_numbers ranks by win_end.max(); a zero-window frame gives
        # NaN, and max() with a NaN key returns the FIRST key rather than raising
        # -- so an empty table sorting first used to anchor the run on a NaN and
        # write "Relative copy number": null into every healthy sequence too.
        d = tmp_path / "in"
        d.mkdir()
        _write_table(d / "aaa_dead.coverage.tsv", seq="")   # sorts first
        _write_table(d / "zzz_live.coverage.tsv")
        out = str(tmp_path / "out")

        self._run_ok(monkeypatch, d, out)

        assert _otr_json(out, "zzz_live")["Relative copy number"] == pytest.approx(1.0)

    def test_break_points_csv_keeps_its_three_columns(self, tmp_path, monkeypatch):
        # breseq asserts exactly three columns and the assert is fatal, so an
        # empty file is not an acceptable stand-in for no segments.
        d = tmp_path / "in"
        d.mkdir()
        _write_table(d / "empty.coverage.tsv", seq="")
        out = str(tmp_path / "out")

        self._run_ok(monkeypatch, d, out)

        path = os.path.join(out, "CNV_csv",
                            f"{os.path.basename(out)}empty_break_pts.csv")
        with open(path) as fh:
            header = fh.readline().strip().split(",")
        assert header == ["Startpos", "State", "Segment_Size"]

    @pytest.mark.parametrize("bias", ["all", "gc", "otr", "none"])
    def test_every_bias_mode_writes_the_json(self, bias, tmp_path, monkeypatch):
        """--bias gc and --bias none used to write NO OTR record at all.

        Both return from correct_one() before any OTR stage runs, so two of the
        four modes gave breseq nothing on a completely healthy table.
        """
        d = tmp_path / "in"
        d.mkdir()
        _write_table(d / "chrA.coverage.tsv")
        out = str(tmp_path / "out")

        _run(monkeypatch, [str(d), "-o", out, "-w", "100", "-s", "100",
                           "--bias", bias])

        data = _otr_json(out, "chrA")
        assert isinstance(data["Origin window"], int)
        assert isinstance(data["Terminus window"], int)
        assert "Origin-to-Terminus/Bias Ratio" in data
