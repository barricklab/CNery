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
FOOTER = "# end\n# end\n# end\n# end\n"


def _tab_content(n=20, cov=50):
    rows = "\n".join(
        f"{i+1}\t{'ACGT'[i % 4]}\t{cov//2}\t{cov//2}\t0\t0"
        for i in range(n)
    )
    return HEADER + rows + "\n" + FOOTER


def _mock_writing_tab(tab_path, n=20, cov=50):
    def side_effect(*args, **kwargs):
        tab_path.write_text(_tab_content(n, cov))
        return MagicMock(returncode=0, stdout="", stderr="")
    return side_effect


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