"""Tests for --coverage-dir: running from pre-generated coverage tables, without a BAM."""

import os

import pytest

from CNery.core import (
    COVERAGE_TABLE_SUFFIX,
    coverage_table_path,
    parse_fasta_records,
    process_multi_genome,
)

HEADER = (
    "position\tref_base\tunique_top_cov\tunique_bot_cov"
    "\tredundant_top_cov\tredundant_bot_cov"
)


def _write_table(path, seq, cov=25):
    rows = "\n".join(
        f"{i + 1}\t{base}\t{cov}\t{cov}\t0\t0" for i, base in enumerate(seq)
    )
    footer = f"#\tnumber_of_positions\t{len(seq)}\n"
    path.write_text(HEADER + "\n" + rows + "\n" + footer)


@pytest.fixture
def two_reference_run(tmp_path):
    """A chromosome + plasmid, with coverage tables but deliberately NO BAM."""
    seqs = {"REL606": "ACGT" * 300, "pPlasmid": "GGCC" * 200}

    fasta = tmp_path / "reference.fasta"
    fasta.write_text("".join(f">{name}\n{s}\n" for name, s in seqs.items()))

    cov_dir = tmp_path / "coverage"
    cov_dir.mkdir()
    for name, s in seqs.items():
        _write_table(cov_dir / f"{name}{COVERAGE_TABLE_SUFFIX}", s)

    out = tmp_path / "out"
    (out / "GC_bias").mkdir(parents=True)
    return {"fasta": str(fasta), "cov_dir": str(cov_dir), "out": str(out), "seqs": seqs}


class TestCoverageTablePath:
    def test_named_by_seq_id_only(self):
        # No coordinates, and crucially no colon: illegal on Windows, shown as "/" by Finder.
        assert coverage_table_path("/cov", "REL606") == "/cov/REL606.coverage.tsv"

    def test_contains_no_colon(self):
        assert ":" not in os.path.basename(coverage_table_path("/cov", "REL606"))


class TestParseFastaRecords:
    def test_seq_id_stops_at_first_whitespace(self, tmp_path):
        # breseq and samtools identify sequences by the first token; the rest is a
        # description. Keeping it would break both the bam2cov region and the table name.
        fa = tmp_path / "desc.fasta"
        fa.write_text(">REL606 Escherichia coli B str. REL606, complete genome\nACGT\n")
        assert parse_fasta_records(str(fa)) == [("REL606", 4)]

    def test_bare_ids_unchanged(self, tmp_path):
        fa = tmp_path / "bare.fasta"
        fa.write_text(">chr1\nACGT\n>pPlasmid\nGG\n")
        assert parse_fasta_records(str(fa)) == [("chr1", 4), ("pPlasmid", 2)]


class TestProcessFromCoverageDir:
    def test_runs_without_any_bam(self, two_reference_run):
        info = two_reference_run
        bam = os.path.join(os.path.dirname(info["fasta"]), "reference.bam")
        assert not os.path.exists(bam)      # the point: no BAM anywhere

        result = process_multi_genome(
            bamfile=bam,
            fastafile=info["fasta"],
            output_prefix=info["out"],
            win=100, step=50, frag=150,
            coverage_dir=info["cov_dir"],
        )

        assert set(result) == set(info["seqs"])
        for seq_id, df in result.items():
            assert len(df) > 0
            assert (df["genome_id"] == seq_id).all()

    def test_never_invokes_breseq(self, two_reference_run, monkeypatch):
        info = two_reference_run

        def explode(*args, **kwargs):
            raise AssertionError("breseq must not be run when --coverage-dir is given")

        monkeypatch.setattr("subprocess.run", explode)

        process_multi_genome(
            bamfile="/nonexistent.bam",
            fastafile=info["fasta"],
            output_prefix=info["out"],
            win=100, step=50, frag=150,
            coverage_dir=info["cov_dir"],
        )

    def test_missing_table_names_the_offender(self, two_reference_run):
        info = two_reference_run
        os.remove(os.path.join(info["cov_dir"], "pPlasmid" + COVERAGE_TABLE_SUFFIX))

        with pytest.raises(FileNotFoundError) as excinfo:
            process_multi_genome(
                bamfile="/nonexistent.bam",
                fastafile=info["fasta"],
                output_prefix=info["out"],
                win=100, step=50, frag=150,
                coverage_dir=info["cov_dir"],
            )
        message = str(excinfo.value)
        assert "pPlasmid.coverage.tsv" in message
        assert "REL606" not in message.replace("REL606.coverage.tsv", "")

    def test_failure_is_reported_before_any_work(self, two_reference_run):
        # Validation must happen up front, not part-way through the reference loop, so a
        # missing table cannot leave half the references processed.
        info = two_reference_run
        os.remove(os.path.join(info["cov_dir"], "REL606" + COVERAGE_TABLE_SUFFIX))
        gc_plots = os.path.join(info["out"], "GC_bias")

        with pytest.raises(FileNotFoundError):
            process_multi_genome(
                bamfile="/nonexistent.bam",
                fastafile=info["fasta"],
                output_prefix=info["out"],
                win=100, step=50, frag=150,
                coverage_dir=info["cov_dir"],
            )
        assert os.listdir(gc_plots) == []

    def test_coverage_dir_wins_over_breseq_dir(self, two_reference_run, tmp_path):
        # An explicit --coverage-dir must not be shadowed by a stale 08_ directory.
        info = two_reference_run
        stale = tmp_path / "run" / "08_mutation_identification"
        stale.mkdir(parents=True)
        for name, s in info["seqs"].items():
            _write_table(stale / f"{name}{COVERAGE_TABLE_SUFFIX}", s, cov=999)

        result = process_multi_genome(
            bamfile="/nonexistent.bam",
            fastafile=info["fasta"],
            output_prefix=info["out"],
            win=100, step=50, frag=150,
            breseq_dir=str(tmp_path / "run"),
            coverage_dir=info["cov_dir"],
        )
        # cov=25 per strand -> 50 total, not the 1998 the stale tables would give.
        assert result["REL606"]["read_count_cov"].median() == 50
