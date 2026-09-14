"""Reference groups: several coverage tables declared to be one molecule.

breseq's -c gives a genome as a draft assembly, and CNery has to be told which
coverage tables are contigs of it -- otherwise each contig refits its own
baseline and is called CN 1 whatever its real copy number. The declaration is a
table breseq writes, naming FILES exactly; this module is about reading it and
mapping it onto the run.
"""

import os

import pytest

from CNery.core import (
    REFERENCE_GROUP_COLUMNS,
    coverage_inputs_from_group_table,
    read_group_table,
    resolve_reference_groups,
)

COLUMNS = [
    "position", "ref_base",
    "unique_top_cov", "unique_bot_cov",
    "redundant_top_cov", "redundant_bot_cov",
]


def _write_table(path, seq="ACGT" * 10, cov=25, d="\t"):
    lines = [d.join(COLUMNS)]
    lines += [
        d.join(str(v) for v in (i + 1, base, cov, cov, 0, 0))
        for i, base in enumerate(seq)
    ]
    lines.append(d.join(("#", "number_of_positions", str(len(seq)))))
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_groups(path, rows, header=("file", "group"), d="\t", footer=None):
    """A reference group table. `rows` are tuples matching `header`."""
    lines = [d.join(header)]
    lines += [d.join("" if v is None else str(v) for v in row) for row in rows]
    if footer:
        lines.append(footer)
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def assembly(tmp_path):
    """Two contigs of one draft assembly, plus a plasmid that is its own reference."""
    cov = tmp_path / "coverage"
    cov.mkdir()
    for name in ("contig_1", "contig_2", "pKAN"):
        _write_table(cov / f"{name}.coverage.tsv")
    inputs = {
        name: str(cov / f"{name}.coverage.tsv")
        for name in ("contig_1", "contig_2", "pKAN")
    }
    return {"dir": cov, "inputs": inputs, "tmp": tmp_path}


def _resolve(assembly, table):
    return resolve_reference_groups(
        assembly["inputs"], read_group_table(str(table)), str(table)
    )


class TestReadGroupTable:
    """Parsing. Format is detected, not declared -- as for a coverage table."""

    def test_tsv_and_csv_both_parse(self, assembly):
        rows = [("contig_1.coverage.tsv", "asm"), ("contig_2.coverage.tsv", "asm")]
        tsv = _write_groups(assembly["dir"] / "g.tsv", rows, d="\t")
        csv = _write_groups(assembly["dir"] / "g.csv", rows, d=",")
        assert read_group_table(str(tsv)) == read_group_table(str(csv))

    def test_comment_lines_are_stripped(self, assembly):
        # breseq appends a provenance footer to its coverage tables; the group
        # table is allowed the same, by the same `#` prefix rule.
        table = _write_groups(
            assembly["dir"] / "g.tsv",
            [("contig_1.coverage.tsv", "asm")],
            footer="#\tbreseq\t0.40.0",
        )
        assert len(read_group_table(str(table))) == 1

    def test_extra_columns_are_ignored(self, assembly):
        # Adding a column must never break a reader -- the same rule the OTR JSON
        # lives under, so breseq can grow the table without a coordinated release.
        table = _write_groups(
            assembly["dir"] / "g.tsv",
            [("contig_1.coverage.tsv", "asm", 412887, "asm.fasta")],
            header=("file", "group", "length", "source"),
        )
        assert read_group_table(str(table)) == [
            (str(assembly["dir"] / "contig_1.coverage.tsv"), "asm")
        ]

    def test_column_order_is_free(self, assembly):
        table = _write_groups(
            assembly["dir"] / "g.tsv",
            [("asm", "contig_1.coverage.tsv")],
            header=("group", "file"),
        )
        assert read_group_table(str(table))[0][1] == "asm"

    def test_relative_paths_resolve_against_the_tables_own_directory(self, assembly):
        # Written beside the coverage tables, read from anywhere. breseq writes
        # basenames; the file must not stop working when read from elsewhere.
        table = _write_groups(
            assembly["dir"] / "g.tsv", [("contig_1.coverage.tsv", "asm")]
        )
        path, _ = read_group_table(str(table))[0]
        assert path == str(assembly["dir"] / "contig_1.coverage.tsv")
        assert os.path.isfile(path)

    @pytest.mark.parametrize("blank", ["", "NA", "n/a", "-", ".", "none"])
    def test_blank_group_means_ungrouped(self, assembly, blank):
        table = _write_groups(
            assembly["dir"] / "g.tsv", [("pKAN.coverage.tsv", blank)]
        )
        assert read_group_table(str(table))[0][1] is None

    def test_header_only_table_is_valid(self, assembly):
        # This is what lets breseq pass --group-table unconditionally instead of
        # deciding whether a run has any draft assemblies in it.
        table = _write_groups(assembly["dir"] / "g.tsv", [])
        assert read_group_table(str(table)) == []


class TestReadGroupTableErrors:
    def test_missing_file_is_reported_by_name(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            read_group_table(str(tmp_path / "nope.tsv"))
        assert str(tmp_path / "nope.tsv") in str(excinfo.value)

    def test_a_directory_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            read_group_table(str(tmp_path))
        assert "directory" in str(excinfo.value)

    @pytest.mark.parametrize(
        "header", [("file", "kind"), ("name", "group"), ("a", "b")]
    )
    def test_missing_required_column_names_path_and_columns(self, tmp_path, header):
        table = _write_groups(tmp_path / "g.tsv", [("x", "y")], header=header)
        with pytest.raises(ValueError) as excinfo:
            read_group_table(str(table))
        message = str(excinfo.value)
        assert str(table) in message
        for column in REFERENCE_GROUP_COLUMNS:
            assert column in message
        # The reader says what it actually found, as normalize_coverage_columns does.
        for column in header:
            assert column in message

    def test_empty_file_cell_is_an_error(self, tmp_path):
        table = _write_groups(tmp_path / "g.tsv", [("", "asm")])
        with pytest.raises(ValueError) as excinfo:
            read_group_table(str(table))
        assert "empty" in str(excinfo.value)


class TestResolveReferenceGroups:
    def test_members_are_grouped_and_the_plasmid_is_its_own(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1.coverage.tsv", "asm"),
            ("contig_2.coverage.tsv", "asm"),
            ("pKAN.coverage.tsv", ""),
        ])
        groups, members = _resolve(assembly, table)
        assert groups == {"contig_1": "asm", "contig_2": "asm", "pKAN": "pKAN"}
        assert members == {"asm": ["contig_1", "contig_2"], "pKAN": ["pKAN"]}

    def test_the_mapping_is_total_over_the_inputs(self, assembly):
        # Every consumer reads groups[genome_id] unconditionally, so a partial
        # mapping would be a KeyError deep inside a stage.
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1.coverage.tsv", "asm"),
            ("contig_2.coverage.tsv", "asm"),
            ("pKAN.coverage.tsv", ""),
        ])
        groups, members = _resolve(assembly, table)
        assert set(groups) == set(assembly["inputs"])
        assert sorted(g for ms in members.values() for g in ms) == sorted(groups)

    def test_no_table_makes_every_sequence_a_group_of_one(self, assembly):
        groups, members = resolve_reference_groups(assembly["inputs"])
        assert groups == {n: n for n in assembly["inputs"]}
        assert all(len(ms) == 1 for ms in members.values())

    def test_an_all_ungrouped_table_is_the_same_as_no_table(self, assembly):
        # The invariant the whole feature rests on, at this layer: passing the flag
        # with nothing grouped must be indistinguishable from not passing it.
        table = _write_groups(assembly["dir"] / "g.tsv", [
            (f"{name}.coverage.tsv", "") for name in assembly["inputs"]
        ])
        assert _resolve(assembly, table) == resolve_reference_groups(
            assembly["inputs"]
        )

    def test_members_keep_input_order(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_2.coverage.tsv", "asm"),
            ("pKAN.coverage.tsv", "asm"),
            ("contig_1.coverage.tsv", "asm"),
        ])
        _, members = _resolve(assembly, table)
        # Input order, not table order: every pooled statistic concatenates in
        # this order, so it has to be the run's order or results depend on how
        # breseq happened to sort its references.
        assert members["asm"] == ["contig_1", "contig_2", "pKAN"]


class TestResolveReferenceGroupsErrors:
    """The table must correspond to the run exactly, in both directions."""

    def test_a_row_for_a_file_the_run_did_not_read(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [
            (f"{name}.coverage.tsv", "asm") for name in assembly["inputs"]
        ] + [("ghost.coverage.tsv", "asm")])
        with pytest.raises(ValueError) as excinfo:
            _resolve(assembly, table)
        assert "ghost.coverage.tsv" in str(excinfo.value)

    def test_an_input_with_no_row(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1.coverage.tsv", "asm"),
            ("contig_2.coverage.tsv", "asm"),
        ])
        with pytest.raises(ValueError) as excinfo:
            _resolve(assembly, table)
        assert "pKAN.coverage.tsv" in str(excinfo.value)

    def test_both_directions_are_reported_together(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1.coverage.tsv", "asm"),
            ("contig_2.coverage.tsv", "asm"),
            ("ghost.coverage.tsv", "asm"),
        ])
        with pytest.raises(ValueError) as excinfo:
            _resolve(assembly, table)
        message = str(excinfo.value)
        assert "ghost.coverage.tsv" in message
        assert "pKAN.coverage.tsv" in message

    def test_one_file_on_two_rows(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1.coverage.tsv", "asm"),
            ("contig_1.coverage.tsv", "other"),
            ("contig_2.coverage.tsv", "asm"),
            ("pKAN.coverage.tsv", ""),
        ])
        with pytest.raises(ValueError) as excinfo:
            _resolve(assembly, table)
        assert "contig_1.coverage.tsv" in str(excinfo.value)

    def test_a_stem_does_not_match_a_file(self, assembly):
        # No stem matching and no ending inference, deliberately: a near-miss
        # between breseq's sequence ID and the name CNery derives from a file is
        # exactly the failure this handoff invites, and matching it loosely would
        # turn that into a silently ungrouped run.
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1", "asm"), ("contig_2", "asm"), ("pKAN", ""),
        ])
        with pytest.raises(ValueError) as excinfo:
            _resolve(assembly, table)
        assert "contig_1" in str(excinfo.value)

    def test_a_group_may_not_be_named_after_a_sequence(self, assembly):
        # A singleton's group key IS its genome_id, so this would silently merge.
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1.coverage.tsv", "pKAN"),
            ("contig_2.coverage.tsv", "pKAN"),
            ("pKAN.coverage.tsv", ""),
        ])
        with pytest.raises(ValueError) as excinfo:
            _resolve(assembly, table)
        assert "pKAN" in str(excinfo.value)


class TestGroupTableSuppliesTheInputs:
    """--group-table with no positional arguments: the table names the files.

    This is what keeps breseq's command line to one line on a draft assembly --
    repeating all 137 paths as arguments says nothing the table does not.
    """

    def test_inputs_come_from_the_file_column(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1.coverage.tsv", "asm"),
            ("contig_2.coverage.tsv", "asm"),
            ("pKAN.coverage.tsv", ""),
        ])
        got = coverage_inputs_from_group_table(
            read_group_table(str(table)), table_path=str(table)
        )
        assert got == assembly["inputs"]

    def test_a_named_file_that_is_not_there(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [("ghost.coverage.tsv", "")])
        with pytest.raises(FileNotFoundError) as excinfo:
            coverage_inputs_from_group_table(
                read_group_table(str(table)), table_path=str(table)
            )
        assert "ghost.coverage.tsv" in str(excinfo.value)

    def test_two_files_resolving_to_one_sequence_id(self, assembly):
        other = assembly["tmp"] / "elsewhere"
        other.mkdir()
        _write_table(other / "contig_1.coverage.tsv")
        table = _write_groups(assembly["dir"] / "g.tsv", [
            ("contig_1.coverage.tsv", "asm"),
            (str(other / "contig_1.coverage.tsv"), "asm"),
        ])
        with pytest.raises(ValueError) as excinfo:
            coverage_inputs_from_group_table(
                read_group_table(str(table)), table_path=str(table)
            )
        assert "contig_1" in str(excinfo.value)

    def test_an_empty_table_names_nothing(self, assembly):
        table = _write_groups(assembly["dir"] / "g.tsv", [])
        with pytest.raises(FileNotFoundError):
            coverage_inputs_from_group_table(
                read_group_table(str(table)), table_path=str(table)
            )


class TestRelativeCopyNumberIsPerGroup:
    """A draft assembly is one reference, so it gets one measured level."""

    @staticmethod
    def _frame(n, cov, end):
        import numpy as np
        import pandas as pd
        return pd.DataFrame({
            "gc_corr_norm_cov": np.full(n, float(cov)),
            "win_end": np.linspace(0.0, float(end), n),
            "is_deletion": np.zeros(n, dtype=bool),
            "is_redundant": np.zeros(n, dtype=bool),
        })

    def test_no_groups_reproduces_the_per_sequence_numbers(self):
        from CNery.core import relative_copy_numbers
        frames = {
            "chrom": self._frame(300, 1.0, 3_000_000),
            "p1": self._frame(20, 2.5, 100_000),
        }
        assert relative_copy_numbers(frames) == relative_copy_numbers(
            frames, {"chrom": "chrom", "p1": "p1"}
        )

    def test_every_member_publishes_the_group_value(self):
        from CNery.core import relative_copy_numbers
        frames = {
            "chrom": self._frame(300, 1.0, 3_000_000),
            "c1": self._frame(20, 2.0, 100_000),
            "c2": self._frame(20, 2.0, 100_000),
        }
        got = relative_copy_numbers(
            frames, {"chrom": "chrom", "c1": "asm", "c2": "asm"}
        )
        assert got["c1"] == got["c2"] == pytest.approx(2.0)
        assert got["chrom"] == pytest.approx(1.0)

    def test_the_median_is_pooled_not_a_median_of_medians(self):
        # Unequal window counts are the whole point: 400 windows at 1.0 and 4 at
        # 9.0 pool to 1.0, where a median of the two medians would read 5.0.
        from CNery.core import relative_copy_numbers
        frames = {
            "anchor": self._frame(1000, 1.0, 9_000_000),
            "big": self._frame(400, 1.0, 400_000),
            "small": self._frame(4, 9.0, 4_000),
        }
        got = relative_copy_numbers(
            frames, {"anchor": "anchor", "big": "asm", "small": "asm"}
        )
        assert got["big"] == pytest.approx(1.0)

    def test_the_anchor_is_the_longest_group_not_the_longest_sequence(self):
        # Two 2 Mb contigs of one assembly outweigh a 3 Mb finished chromosome,
        # so the assembly anchors at 1.0 and the chromosome reads its own level.
        from CNery.core import relative_copy_numbers
        frames = {
            "chrom": self._frame(300, 2.0, 3_000_000),
            "c1": self._frame(200, 1.0, 2_000_000),
            "c2": self._frame(200, 1.0, 2_000_000),
        }
        got = relative_copy_numbers(
            frames, {"chrom": "chrom", "c1": "asm", "c2": "asm"}
        )
        assert got["c1"] == pytest.approx(1.0)
        assert got["chrom"] == pytest.approx(2.0)

    def test_an_empty_member_does_not_poison_its_group(self):
        from CNery.core import relative_copy_numbers
        frames = {
            "chrom": self._frame(300, 1.0, 3_000_000),
            "c1": self._frame(20, 2.0, 100_000),
            "dead": self._frame(0, 1.0, 1),
        }
        got = relative_copy_numbers(
            frames, {"chrom": "chrom", "c1": "asm", "dead": "asm"}
        )
        assert got["c1"] == pytest.approx(2.0)
