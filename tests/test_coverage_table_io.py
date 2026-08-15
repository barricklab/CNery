"""Reading a coverage table off disk: footer stripping and the schema check.

These tests used to reach _read_coverage_tab() through bam2cov_to_df(), with breseq's
subprocess mocked out. Nothing here ever needed breseq -- the footer is a property of the
file, not of how it was produced -- so they now write a table and read it directly.
"""

import pandas as pd
import pytest

from CNery.core import _read_coverage_tab, read_coverage_table

HEADER = (
    "position\tref_base\tunique_top_cov\tunique_bot_cov"
    "\tredundant_top_cov\tredundant_bot_cov\n"
)
# breseq's default footer: coverage_output.cpp writes these four unconditionally.
FOOTER = (
    "#\tregion_unique_average_cov\t50\n"
    "#\tregion_repeat_average_cov\t0\n"
    "#\tregion_average_cov\t50\n"
    "#\tnumber_of_positions\t20\n"
)
# --show-average prepends one more line (m_show_average).
FOOTER_SHOW_AVERAGE = "#\treference_unique_average_cov\t51\n" + FOOTER
# Per-read-group output appends three lines per group (m_per_read_group).
FOOTER_TWO_READ_GROUPS = FOOTER + "".join(
    f"#\t{p}_{stat}\t50\n"
    for p in ("rg1", "rg2")
    for stat in ("region_unique_average_cov", "region_repeat_average_cov", "region_average_cov")
)


def _write_table(path, n=20, cov=50, footer=FOOTER, header=HEADER):
    rows = "\n".join(
        f"{i+1}\t{'ACGT'[i % 4]}\t{cov//2}\t{cov//2}\t0\t0"
        for i in range(n)
    )
    path.write_text(header + rows + "\n" + footer)
    return str(path)


class TestFooterHandling:
    """The trailing '#' block is variable-length, so it must be matched by prefix.

    coverage_output.cpp writes four lines by default, one more under --show-average, and three
    per read group. breseq's own 08_mutation_identification/*.coverage.tab has none at all.
    A fixed skipfooter=4 silently deleted four real data rows from footerless input.
    """

    def test_no_footer_keeps_every_row(self, tmp_path):
        # Regression: this is the shape of 08_mutation_identification/*.coverage.tab.
        df = _read_coverage_tab(_write_table(tmp_path / "nofoot.tsv", footer=""))
        assert len(df) == 20
        assert df.index[-1] == 20      # last position survives

    def test_default_footer_is_stripped(self, tmp_path):
        df = _read_coverage_tab(_write_table(tmp_path / "foot4.tsv", footer=FOOTER))
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_show_average_footer_is_stripped(self, tmp_path):
        # Five lines, not four -- a fixed count cannot handle this.
        df = _read_coverage_tab(
            _write_table(tmp_path / "foot5.tsv", footer=FOOTER_SHOW_AVERAGE)
        )
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_per_read_group_footer_is_stripped(self, tmp_path):
        # 4 + 3 per read group = 10 lines here.
        df = _read_coverage_tab(
            _write_table(tmp_path / "footrg.tsv", footer=FOOTER_TWO_READ_GROUPS)
        )
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_no_footer_data_values_are_intact(self, tmp_path):
        # Guard the payload, not just the row count.
        df = _read_coverage_tab(_write_table(tmp_path / "vals.tsv", footer=""))
        assert (df["unique_top_cov"] == 25).all()
        assert (df["unique_bot_cov"] == 25).all()
        assert list(df.index) == list(range(1, 21))


class TestSchemaCheck:
    """read_coverage_table() rejects a wrong-schema file at the boundary.

    _read_coverage_tab() takes column 0 as the position index whatever it holds, so a
    mismatched table parses cleanly and only fails later, as a bare KeyError from inside
    preprocess(). Any file can now be named on the command line, so that failure has to
    be caught where the file is read.
    """

    def test_valid_table_passes_through(self, tmp_path):
        df = read_coverage_table(_write_table(tmp_path / "ok.tsv"))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

    def test_missing_ref_base_is_named(self, tmp_path):
        # The 08_mutation_identification/*.coverage.tab schema: no ref_base, and GC
        # content is computed from ref_base, so CNery cannot use it.
        path = tmp_path / "noref.tsv"
        path.write_text(
            "position\tunique_top_cov\tunique_bot_cov\tredundant_top_cov\tredundant_bot_cov\n"
            + "\n".join(f"{i+1}\t25\t25\t0\t0" for i in range(20))
            + "\n"
        )
        with pytest.raises(ValueError) as excinfo:
            read_coverage_table(str(path))
        assert "ref_base" in str(excinfo.value)
        assert str(path) in str(excinfo.value)

    def test_unrelated_tsv_is_rejected(self, tmp_path):
        path = tmp_path / "notes.tsv"
        path.write_text("a\tb\n1\t2\n")
        with pytest.raises(ValueError) as excinfo:
            read_coverage_table(str(path))
        message = str(excinfo.value)
        for col in ("ref_base", "unique_top_cov", "redundant_bot_cov"):
            assert col in message
