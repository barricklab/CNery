"""End-to-end runs through main(), driving the real argparse surface.

Nothing exercised the command line before, which mattered less when the CLI was a thin
wrapper over a fixed pair of BAM/FASTA paths. Inputs are now the whole interface: which
files get picked up, what they are called, and what a bad invocation leaves behind.
"""

import os
import sys

import pytest

from CNery.get_CNV import main

HEADER = (
    "position\tref_base\tunique_top_cov\tunique_bot_cov"
    "\tredundant_top_cov\tredundant_bot_cov"
)

# Enough windows at -w 100 -s 50 for the HMM and the OTR fit to have something to chew on.
SEQ = ("ACGTACGGCTAA" * 250)


def _write_table(path, seq=SEQ, cov=25):
    rows = "\n".join(
        f"{i + 1}\t{base}\t{cov}\t{cov}\t0\t0" for i, base in enumerate(seq)
    )
    path.write_text(HEADER + "\n" + rows + "\n#\tnumber_of_positions\t%d\n" % len(seq))
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
        # One pooled GC fit across both, not one per reference.
        assert len(os.listdir(os.path.join(out, "GC_bias"))) == 1

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
        for flag in ("-ref", "-i"):
            with pytest.raises(SystemExit):
                _run(monkeypatch, [str(table), flag, "whatever"])
