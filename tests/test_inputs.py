"""Turning command-line inputs into coverage tables.

CNery takes coverage tables and nothing else: no FASTA, no BAM, and it never runs breseq.
Inputs are files and/or folders, folders are expanded by file ending, and the sequence ID
of each table comes from its file name.
"""

import os

import pytest

from CNery.core import (
    COVERAGE_TABLE_SUFFIX,
    DEFAULT_FILE_ENDINGS,
    coverage_table_path,
    genome_id_from_path,
    process_multi_genome,
    resolve_coverage_inputs,
)

COLUMNS = [
    "position", "ref_base",
    "unique_top_cov", "unique_bot_cov",
    "redundant_top_cov", "redundant_bot_cov",
]


def _write_table(path, seq, cov=25, d="\t"):
    """A coverage table, delimited by `d` throughout -- header, data and footer."""
    lines = [d.join(COLUMNS)]
    lines += [
        d.join(str(v) for v in (i + 1, base, cov, cov, 0, 0))
        for i, base in enumerate(seq)
    ]
    lines.append(d.join(("#", "number_of_positions", str(len(seq)))))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def two_reference_run(tmp_path):
    """A chromosome + plasmid as coverage tables. No FASTA, no BAM -- none is read."""
    seqs = {"REL606": "ACGT" * 300, "pPlasmid": "GGCC" * 200}

    cov_dir = tmp_path / "coverage"
    cov_dir.mkdir()
    for name, s in seqs.items():
        _write_table(cov_dir / f"{name}{COVERAGE_TABLE_SUFFIX}", s)

    out = tmp_path / "out"
    (out / "GC_bias").mkdir(parents=True)
    return {"cov_dir": str(cov_dir), "out": str(out), "seqs": seqs}


class TestCoverageTablePath:
    def test_named_by_seq_id_only(self):
        # No coordinates, and crucially no colon: illegal on Windows, shown as "/" by Finder.
        assert coverage_table_path("/cov", "REL606") == "/cov/REL606.coverage.tsv"

    def test_contains_no_colon(self):
        assert ":" not in os.path.basename(coverage_table_path("/cov", "REL606"))

    def test_suffix_tracks_the_default_ending(self):
        assert COVERAGE_TABLE_SUFFIX == ".coverage.tsv"
        assert "coverage.tsv" in DEFAULT_FILE_ENDINGS


class TestGenomeIdFromPath:
    def test_default_ending_is_stripped(self):
        assert genome_id_from_path("/cov/REL606.coverage.tsv") == "REL606"

    def test_interior_dots_are_kept(self):
        # Only the ending goes, not everything after the first dot. Sequence IDs
        # routinely contain dots, and truncating there would merge distinct references.
        assert genome_id_from_path("/cov/my.sample.1.coverage.tsv") == "my.sample.1"

    def test_accession_style_id_survives(self):
        assert genome_id_from_path("/cov/NC_012967.1.coverage.tsv") == "NC_012967.1"

    def test_custom_ending(self):
        assert genome_id_from_path("/cov/REL606.cov.txt", ["cov.txt"]) == "REL606"

    def test_custom_ending_leaves_the_rest_alone(self):
        # --file-ending tsv strips only ".tsv", so ".coverage" stays part of the id.
        assert genome_id_from_path("/cov/REL606.coverage.tsv", ["tsv"]) == "REL606.coverage"

    def test_leading_dot_on_the_ending_is_tolerated(self):
        assert genome_id_from_path("/cov/REL606.coverage.tsv", [".coverage.tsv"]) == "REL606"

    def test_first_matching_ending_wins(self):
        got = genome_id_from_path("/cov/REL606.coverage.tsv", ["coverage.tsv", "tsv"])
        assert got == "REL606"

    def test_unmatched_name_drops_only_its_extension(self):
        # A file named directly on the command line need not match any ending.
        assert genome_id_from_path("/cov/weird.name.txt", ["coverage.tsv"]) == "weird.name"

    def test_name_that_is_only_the_ending(self):
        # No "." in front of the ending to remove, so fall back to the extension rule.
        assert genome_id_from_path("/cov/coverage.tsv") == "coverage"

    def test_csv_ending_is_stripped_too(self):
        assert genome_id_from_path("/cov/REL606.coverage.csv") == "REL606"

    def test_csv_interior_dots_are_kept(self):
        assert genome_id_from_path("/cov/my.sample.1.coverage.csv") == "my.sample.1"


class TestBothFormatsAreDefaults:
    """CSV and TSV are found without being asked for; they differ only in delimiter."""

    def test_csv_is_picked_up_by_default(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "chrA.coverage.csv", "ACGT" * 10, d=",")
        assert list(resolve_coverage_inputs([str(d)])) == ["chrA"]

    def test_a_folder_may_mix_the_two(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "chrA.coverage.csv", "ACGT" * 10, d=",")
        _write_table(d / "chrB.coverage.tsv", "GGCC" * 10, d="\t")
        assert set(resolve_coverage_inputs([str(d)])) == {"chrA", "chrB"}

    def test_same_sequence_in_both_formats_is_ambiguous(self, tmp_path):
        # Both resolve to the same sequence ID. They could disagree -- different BAMs,
        # different windowing -- so refuse rather than silently picking one.
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "REL606.coverage.csv", "ACGT" * 10, d=",")
        _write_table(d / "REL606.coverage.tsv", "GGCC" * 10, d="\t")

        with pytest.raises(ValueError) as excinfo:
            resolve_coverage_inputs([str(d)])
        message = str(excinfo.value)
        assert "REL606" in message
        assert "REL606.coverage.csv" in message
        assert "REL606.coverage.tsv" in message


class TestResolveCoverageInputs:
    def test_directory_is_expanded(self, two_reference_run):
        got = resolve_coverage_inputs([two_reference_run["cov_dir"]])
        assert set(got) == {"REL606", "pPlasmid"}

    def test_explicit_files_are_loaded(self, two_reference_run):
        cov = two_reference_run["cov_dir"]
        got = resolve_coverage_inputs([
            os.path.join(cov, "REL606.coverage.tsv"),
            os.path.join(cov, "pPlasmid.coverage.tsv"),
        ])
        assert list(got) == ["REL606", "pPlasmid"]

    def test_files_and_directories_can_be_mixed(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "chrA.coverage.tsv", "ACGT" * 10)
        loose = _write_table(tmp_path / "chrB.coverage.tsv", "GGCC" * 10)

        got = resolve_coverage_inputs([str(loose), str(d)])
        assert set(got) == {"chrA", "chrB"}

    def test_directory_listing_is_sorted(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        for name in ("zeta", "alpha", "mu"):
            _write_table(d / f"{name}.coverage.tsv", "ACGT" * 10)
        assert list(resolve_coverage_inputs([str(d)])) == ["alpha", "mu", "zeta"]

    def test_explicit_file_need_not_match_the_ending(self, tmp_path):
        odd = _write_table(tmp_path / "sample.txt", "ACGT" * 10)
        assert list(resolve_coverage_inputs([str(odd)])) == ["sample"]

    def test_non_matching_files_in_a_directory_are_ignored(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "chrA.coverage.tsv", "ACGT" * 10)
        (d / "notes.txt").write_text("not a coverage table")
        (d / "reference.fasta").write_text(">chrA\nACGT\n")
        assert list(resolve_coverage_inputs([str(d)])) == ["chrA"]

    def test_subdirectories_are_not_searched(self, tmp_path):
        # Non-recursive on purpose: a stale copy one level down would otherwise
        # collide with the live table and break a working command.
        d = tmp_path / "dir"
        (d / "old").mkdir(parents=True)
        _write_table(d / "chrA.coverage.tsv", "ACGT" * 10)
        _write_table(d / "old" / "chrA.coverage.tsv", "TTTT" * 10)
        assert list(resolve_coverage_inputs([str(d)])) == ["chrA"]


class TestFileEndingOverride:
    def test_custom_ending_matches(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "chrA.cov.txt", "ACGT" * 10)
        assert list(resolve_coverage_inputs([str(d)], ["cov.txt"])) == ["chrA"]

    def test_custom_ending_replaces_the_default(self, tmp_path):
        # The point of "replaces, not extends": the default must stop matching.
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "chrA.cov.txt", "ACGT" * 10)
        _write_table(d / "chrB.coverage.tsv", "GGCC" * 10)
        assert list(resolve_coverage_inputs([str(d)], ["cov.txt"])) == ["chrA"]

    def test_several_endings_are_all_matched(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "chrA.cov.txt", "ACGT" * 10)
        _write_table(d / "chrB.coverage.tsv", "GGCC" * 10)
        got = resolve_coverage_inputs([str(d)], ["cov.txt", "coverage.tsv"])
        assert set(got) == {"chrA", "chrB"}

    def test_none_means_the_default(self, two_reference_run):
        # argparse leaves file_ending as None when the flag is unused; that value is
        # passed straight through.
        got = resolve_coverage_inputs([two_reference_run["cov_dir"]], None)
        assert set(got) == {"REL606", "pPlasmid"}


class TestInputErrors:
    def test_missing_path_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_coverage_inputs([str(tmp_path / "nope")])
        assert "nope" in str(excinfo.value)

    def test_directory_with_no_tables_names_the_endings_tried(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        (d / "readme.txt").write_text("nothing here")
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_coverage_inputs([str(d)])
        message = str(excinfo.value)
        assert str(d) in message
        for ending in DEFAULT_FILE_ENDINGS:
            assert ending in message

    def test_duplicate_genome_id_names_both_paths(self, tmp_path):
        a = tmp_path / "runA"
        b = tmp_path / "runB"
        a.mkdir()
        b.mkdir()
        _write_table(a / "REL606.coverage.tsv", "ACGT" * 10)
        _write_table(b / "REL606.coverage.tsv", "GGCC" * 10)

        with pytest.raises(ValueError) as excinfo:
            resolve_coverage_inputs([str(a), str(b)])
        message = str(excinfo.value)
        assert "REL606" in message
        assert str(a / "REL606.coverage.tsv") in message
        assert str(b / "REL606.coverage.tsv") in message

    def test_duplicate_across_differing_endings(self, tmp_path):
        # Two endings can strip to the same id even from different file names.
        d = tmp_path / "dir"
        d.mkdir()
        _write_table(d / "chrA.cov.txt", "ACGT" * 10)
        _write_table(d / "chrA.coverage.tsv", "GGCC" * 10)
        with pytest.raises(ValueError) as excinfo:
            resolve_coverage_inputs([str(d)], ["cov.txt", "coverage.tsv"])
        assert "chrA" in str(excinfo.value)

    def test_nothing_resolved_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_coverage_inputs([], None)


class TestProcessFromCoverageTables:
    def test_runs_without_any_bam_or_fasta(self, two_reference_run):
        info = two_reference_run
        # The point: nothing but the tables exists anywhere.
        for absent in ("reference.bam", "reference.fasta"):
            assert not os.path.exists(os.path.join(info["cov_dir"], absent))

        result = process_multi_genome(
            resolve_coverage_inputs([info["cov_dir"]]),
            output_prefix=info["out"],
            win=100, step=50, frag=150,
        )

        assert set(result) == set(info["seqs"])
        for seq_id, df in result.items():
            assert len(df) > 0
            assert (df["genome_id"] == seq_id).all()

    def test_genome_id_comes_from_the_file_name(self, tmp_path):
        cov = tmp_path / "coverage"
        cov.mkdir()
        _write_table(cov / "my.sample.1.coverage.tsv", "ACGT" * 300)
        out = tmp_path / "out"
        (out / "GC_bias").mkdir(parents=True)

        result = process_multi_genome(
            resolve_coverage_inputs([str(cov)]),
            output_prefix=str(out),
            win=100, step=50, frag=150,
        )
        assert list(result) == ["my.sample.1"]

    def test_pooled_gc_fit_covers_every_input(self, two_reference_run):
        info = two_reference_run
        process_multi_genome(
            resolve_coverage_inputs([info["cov_dir"]]),
            output_prefix=info["out"],
            win=100, step=50, frag=150,
        )
        # One pooled diagnostic plot, named for both references.
        produced = os.listdir(os.path.join(info["out"], "GC_bias"))
        assert len(produced) == 1
        assert "REL606" in produced[0] and "pPlasmid" in produced[0]

    def test_bad_schema_is_reported_before_any_work(self, tmp_path):
        cov = tmp_path / "coverage"
        cov.mkdir()
        (cov / "chrA.coverage.tsv").write_text("a\tb\n1\t2\n")
        out = tmp_path / "out"
        (out / "GC_bias").mkdir(parents=True)

        with pytest.raises(ValueError):
            process_multi_genome(
                resolve_coverage_inputs([str(cov)]),
                output_prefix=str(out),
                win=100, step=50, frag=150,
            )
        assert os.listdir(os.path.join(out, "GC_bias")) == []
