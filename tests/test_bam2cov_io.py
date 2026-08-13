import subprocess
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from CNery.core import bam2cov_to_df

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


def _tab_content(n=20, cov=50, footer=FOOTER):
    rows = "\n".join(
        f"{i+1}\t{'ACGT'[i % 4]}\t{cov//2}\t{cov//2}\t0\t0"
        for i in range(n)
    )
    return HEADER + rows + "\n" + footer


def _mock_writing_tab(tab_path, n=20, cov=50, footer=FOOTER):
    def side_effect(*args, **kwargs):
        tab_path.write_text(_tab_content(n, cov, footer))
        return MagicMock(returncode=0, stdout="", stderr="")
    return side_effect


def _read_with_footer(tmp_path, single_fasta, footer, n=20, name="cov"):
    """Run bam2cov_to_df against a mocked breseq that writes a table with `footer`."""
    bam = tmp_path / "s.bam"
    bam.touch()
    prefix = str(tmp_path / name)
    tab = Path(prefix + ".tab")
    with patch("subprocess.run", side_effect=_mock_writing_tab(tab, n=n, footer=footer)):
        return bam2cov_to_df(str(bam), single_fasta, prefix)


class TestFooterHandling:
    """The trailing '#' block is variable-length, so it must be matched by prefix.

    coverage_output.cpp writes four lines by default, one more under --show-average, and three
    per read group. breseq's own 08_mutation_identification/*.coverage.tab has none at all.
    A fixed skipfooter=4 silently deleted four real data rows from footerless input.
    """

    def test_no_footer_keeps_every_row(self, tmp_path, single_fasta):
        # Regression: this is the shape of 08_mutation_identification/*.coverage.tab.
        df = _read_with_footer(tmp_path, single_fasta, footer="", name="nofoot")
        assert len(df) == 20
        assert df.index[-1] == 20      # last position survives

    def test_default_footer_is_stripped(self, tmp_path, single_fasta):
        df = _read_with_footer(tmp_path, single_fasta, footer=FOOTER, name="foot4")
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_show_average_footer_is_stripped(self, tmp_path, single_fasta):
        # Five lines, not four -- a fixed count cannot handle this.
        df = _read_with_footer(
            tmp_path, single_fasta, footer=FOOTER_SHOW_AVERAGE, name="foot5"
        )
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_per_read_group_footer_is_stripped(self, tmp_path, single_fasta):
        # 4 + 3 per read group = 10 lines here.
        df = _read_with_footer(
            tmp_path, single_fasta, footer=FOOTER_TWO_READ_GROUPS, name="footrg"
        )
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_no_footer_data_values_are_intact(self, tmp_path, single_fasta):
        # Guard the payload, not just the row count.
        df = _read_with_footer(tmp_path, single_fasta, footer="", name="vals")
        assert (df["unique_top_cov"] == 25).all()
        assert (df["unique_bot_cov"] == 25).all()
        assert list(df.index) == list(range(1, 21))


def test_returns_dataframe(tmp_path, single_fasta):
    bam = tmp_path / "s.bam"
    bam.touch()
    prefix = str(tmp_path / "cov")
    tab = Path(prefix + ".tab")
    with patch("subprocess.run", side_effect=_mock_writing_tab(tab)):
        df = bam2cov_to_df(str(bam), single_fasta, prefix)
    assert isinstance(df, pd.DataFrame)


def test_missing_tab_file_raises(tmp_path, single_fasta):
    bam = tmp_path / "s.bam"
    bam.touch()
    prefix = str(tmp_path / "cov_missing")
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        with pytest.raises(Exception):
            bam2cov_to_df(str(bam), single_fasta, prefix)


def test_command_contains_bam_and_fasta(tmp_path, single_fasta):
    bam = tmp_path / "s.bam"
    bam.touch()
    prefix = str(tmp_path / "cov_cmd")
    tab = Path(prefix + ".tab")
    with patch("subprocess.run", side_effect=_mock_writing_tab(tab)) as mock_run:
        try:
            bam2cov_to_df(str(bam), single_fasta, prefix)
        except Exception:
            pass
        called_args = str(mock_run.call_args)
        assert str(bam) in called_args or single_fasta in called_args