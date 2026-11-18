"""Reading a coverage table off disk: delimiter, footer, and schema.

breseq bam2cov writes four shapes of the same data -- TSV or CSV, strand-split or
--total-only -- and CNery accepts all four without being told which it is given. The
delimiter comes from the file's own header row and the schema from its column names, so
these tests pin both detections as well as the variable-length footer stripping.
"""

import numpy as np
import pandas as pd
import pytest

from CNery.core import (
    _detect_delimiter,
    _read_coverage_table,
    normalize_coverage_columns,
    preprocess,
    read_coverage_table,
)

STRAND_COLUMNS = [
    "position", "ref_base",
    "unique_top_cov", "unique_bot_cov",
    "redundant_top_cov", "redundant_bot_cov",
]
TOTAL_ONLY_COLUMNS = ["position", "ref_base", "unique_cov", "redundant_cov", "total_cov"]

# breseq's default footer: coverage_output.cpp writes these four unconditionally.
FOOTER_ROWS = [
    ("region_unique_average_cov", "50"),
    ("region_repeat_average_cov", "0"),
    ("region_average_cov", "50"),
    ("number_of_positions", "20"),
]
# --show-average prepends one more line (m_show_average).
SHOW_AVERAGE_ROWS = [("reference_unique_average_cov", "51")] + FOOTER_ROWS
# Per-read-group output appends three lines per group (m_per_read_group).
TWO_READ_GROUP_ROWS = FOOTER_ROWS + [
    (f"{p}_{stat}", "50")
    for p in ("RG-0", "RG-1")
    for stat in ("region_unique_average_cov", "region_repeat_average_cov", "region_average_cov")
]

DELIMITERS = [("\t", "tsv"), (",", "csv")]


def _render(header, rows, footer_rows, d):
    """One table as text, using `d` between every field -- header, data and footer alike.

    That is exactly what bam2cov does: it picks the separator once from the output format
    and reuses it everywhere, so a CSV footer reads "#,region_average_cov,50".
    """
    lines = [d.join(header)]
    lines += [d.join(str(v) for v in row) for row in rows]
    lines += [d.join(("#",) + tuple(row)) for row in footer_rows]
    return "\n".join(lines) + "\n"


def _strand_rows(n=20, cov=25, redundant=0):
    return [
        (i + 1, "ACGT"[i % 4], cov, cov, redundant, redundant)
        for i in range(n)
    ]


def _write_strand(path, n=20, cov=25, redundant=0, footer_rows=FOOTER_ROWS, d="\t"):
    path.write_text(_render(STRAND_COLUMNS, _strand_rows(n, cov, redundant), footer_rows, d))
    return str(path)


def _write_total_only(path, n=20, cov=25, redundant=0, footer_rows=FOOTER_ROWS, d=","):
    """The same coverage, in the shape `bam2cov --total-only` writes.

    unique_cov / redundant_cov are the strand sums breseq computes itself, and total_cov
    is their sum -- which is why the narrower table costs CNery nothing.
    """
    rows = [
        (i + 1, "ACGT"[i % 4], cov * 2, redundant * 2, cov * 2 + redundant * 2)
        for i in range(n)
    ]
    path.write_text(_render(TOTAL_ONLY_COLUMNS, rows, footer_rows, d))
    return str(path)


class TestDelimiterDetection:
    """The format is a property of the bytes, not of the file name."""

    def test_tab_detected(self, tmp_path):
        assert _detect_delimiter(_write_strand(tmp_path / "a.tsv", d="\t")) == "\t"

    def test_comma_detected(self, tmp_path):
        assert _detect_delimiter(_write_strand(tmp_path / "a.csv", d=",")) == ","

    def test_extension_does_not_decide(self, tmp_path):
        # A comma-delimited table saved as ".tsv" still reads correctly -- the point of
        # sniffing content rather than trusting the name.
        df = _read_coverage_table(_write_strand(tmp_path / "misnamed.tsv", d=","))
        assert len(df) == 20
        assert list(df.columns[:1]) == ["ref_base"]

    def test_leading_comment_lines_are_skipped_when_detecting(self, tmp_path):
        path = tmp_path / "lead.csv"
        body = _render(STRAND_COLUMNS, _strand_rows(), FOOTER_ROWS, ",")
        path.write_text("#,a note\n" + body)
        assert _detect_delimiter(str(path)) == ","

    def test_ambiguous_header_is_rejected(self, tmp_path):
        path = tmp_path / "weird.tsv"
        path.write_text("just_one_column\n1\n2\n")
        with pytest.raises(ValueError) as excinfo:
            _detect_delimiter(str(path))
        assert str(path) in str(excinfo.value)

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("")
        with pytest.raises(ValueError):
            _detect_delimiter(str(path))


@pytest.mark.parametrize("d,label", DELIMITERS)
class TestFooterHandling:
    """The trailing '#' block is variable-length, so it must be matched by prefix.

    coverage_output.cpp writes four lines by default, one more under --show-average, and three
    per read group. breseq's own 08_mutation_identification/*.coverage.tab has none at all.
    A fixed skipfooter=4 silently deleted four real data rows from footerless input.

    Run for both delimiters: the footer is written with the same separator as the data, so
    CSV needs the same proof TSV does.
    """

    def test_no_footer_keeps_every_row(self, tmp_path, d, label):
        # Regression: this is the shape of 08_mutation_identification/*.coverage.tab.
        df = _read_coverage_table(
            _write_strand(tmp_path / f"nofoot.{label}", footer_rows=[], d=d)
        )
        assert len(df) == 20
        assert df.index[-1] == 20      # last position survives

    def test_default_footer_is_stripped(self, tmp_path, d, label):
        df = _read_coverage_table(
            _write_strand(tmp_path / f"foot4.{label}", footer_rows=FOOTER_ROWS, d=d)
        )
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_show_average_footer_is_stripped(self, tmp_path, d, label):
        # Five lines, not four -- a fixed count cannot handle this.
        df = _read_coverage_table(
            _write_strand(tmp_path / f"foot5.{label}", footer_rows=SHOW_AVERAGE_ROWS, d=d)
        )
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_per_read_group_footer_is_stripped(self, tmp_path, d, label):
        # 4 + 3 per read group = 10 lines here.
        df = _read_coverage_table(
            _write_strand(tmp_path / f"footrg.{label}", footer_rows=TWO_READ_GROUP_ROWS, d=d)
        )
        assert len(df) == 20
        assert df.index[-1] == 20

    def test_data_values_are_intact(self, tmp_path, d, label):
        # Guard the payload, not just the row count.
        df = _read_coverage_table(
            _write_strand(tmp_path / f"vals.{label}", footer_rows=[], d=d)
        )
        assert (df["unique_top_cov"] == 25).all()
        assert (df["unique_bot_cov"] == 25).all()
        assert list(df.index) == list(range(1, 21))


class TestSchemaDetection:
    """Either bam2cov coverage schema is accepted, and anything else is refused by name.

    _read_coverage_table() takes column 0 as the position index whatever it holds, so a
    mismatched table parses cleanly and only fails later, as a bare KeyError from inside
    preprocess(). Any file can be named on the command line, so that has to be caught
    where the file is read.
    """

    def test_strand_split_is_accepted(self, tmp_path):
        df = read_coverage_table(_write_strand(tmp_path / "ok.tsv"))
        assert isinstance(df, pd.DataFrame)
        assert {"unique_cov", "redundant"} <= set(df.columns)

    def test_total_only_is_accepted(self, tmp_path):
        df = read_coverage_table(_write_total_only(tmp_path / "ok.csv"))
        assert {"unique_cov", "redundant"} <= set(df.columns)

    def test_total_cov_is_not_required(self, tmp_path):
        # total_cov is unique_cov + redundant_cov by construction, so CNery ignores it.
        path = tmp_path / "notot.csv"
        rows = [(i + 1, "ACGT"[i % 4], 50, 0) for i in range(20)]
        path.write_text(_render(
            ["position", "ref_base", "unique_cov", "redundant_cov"], rows, FOOTER_ROWS, ","
        ))
        df = read_coverage_table(str(path))
        assert (df["unique_cov"] == 50).all()
        assert (df["redundant"] == 0).all()

    def test_normalization_is_idempotent(self, tmp_path):
        # preprocess() normalizes again on a frame that already came through
        # read_coverage_table(); that must be a no-op, not a double-count.
        once = read_coverage_table(_write_strand(tmp_path / "idem.tsv", cov=25))
        twice = normalize_coverage_columns(once)
        pd.testing.assert_series_equal(once["unique_cov"], twice["unique_cov"])
        pd.testing.assert_series_equal(once["redundant"], twice["redundant"])

    def test_missing_ref_base_is_named(self, tmp_path):
        # The 08_mutation_identification/*.coverage.tab schema: no ref_base, and GC
        # content is computed from ref_base, so CNery cannot use it.
        path = tmp_path / "noref.tsv"
        rows = [(i + 1, 25, 25, 0, 0) for i in range(20)]
        path.write_text(_render(
            ["position", "unique_top_cov", "unique_bot_cov",
             "redundant_top_cov", "redundant_bot_cov"], rows, [], "\t"
        ))
        with pytest.raises(ValueError) as excinfo:
            read_coverage_table(str(path))
        assert "ref_base" in str(excinfo.value)
        assert str(path) in str(excinfo.value)

    def test_unrelated_table_names_both_accepted_schemas(self, tmp_path):
        path = tmp_path / "notes.csv"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError) as excinfo:
            read_coverage_table(str(path))
        message = str(excinfo.value)
        assert "unique_top_cov" in message      # the strand-split shape
        assert "unique_cov" in message          # the --total-only shape
        assert "total-only" in message


class TestTotalOnlyIsLossless:
    """The same coverage, written both ways, must give the same windowed result.

    This is what makes --total-only a real input route rather than a quietly degraded
    one. breseq sums the strands in C++ exactly as preprocess() would in Python, so
    nothing CNery uses is lost -- including pct_redundant, which drives repeat censoring.
    """

    @staticmethod
    def _both(tmp_path, n=400, cov=25, redundant=0):
        strand = read_coverage_table(
            _write_strand(tmp_path / "s.tsv", n=n, cov=cov, redundant=redundant, d="\t")
        )
        total = read_coverage_table(
            _write_total_only(tmp_path / "t.csv", n=n, cov=cov, redundant=redundant, d=",")
        )
        kw = dict(win=100, step=50, frag=150)
        return preprocess(strand, **kw), preprocess(total, **kw)

    def test_identical_window_coverage(self, tmp_path):
        a, b = self._both(tmp_path)
        assert len(a) == len(b)
        np.testing.assert_allclose(
            a["read_count_cov"].to_numpy(), b["read_count_cov"].to_numpy()
        )

    def test_identical_gc(self, tmp_path):
        a, b = self._both(tmp_path)
        np.testing.assert_allclose(
            a["gc_percent"].to_numpy(), b["gc_percent"].to_numpy()
        )

    def test_repeat_censoring_survives(self, tmp_path):
        # The one thing --total-only could plausibly have cost: redundant coverage is
        # still reported separately, so pct_redundant is still computable.
        a, b = self._both(tmp_path, redundant=5)
        assert (a["pct_redundant"] > 0).all()
        np.testing.assert_allclose(
            a["pct_redundant"].to_numpy(), b["pct_redundant"].to_numpy()
        )

    def test_no_redundant_coverage_reads_as_zero(self, tmp_path):
        a, b = self._both(tmp_path, redundant=0)
        assert (a["pct_redundant"] == 0).all()
        assert (b["pct_redundant"] == 0).all()


class TestTableWithNoDataRows:
    """A valid header and no positions is a REFERENCE WITH NO READS, not a broken file.

    bam2cov writes one for a sequence nothing mapped to, and every check
    read_coverage_table() makes passes: the columns are all there, only the rows
    are missing. preprocess() then took df.index[0] to find the first coordinate
    and raised a bare IndexError four frames below the input boundary -- taking
    down every other reference in the same invocation with it.
    """

    def test_it_is_accepted_by_the_reader(self, tmp_path):
        path = _write_strand(tmp_path / "none.tsv", n=0)
        df = read_coverage_table(path)
        assert len(df) == 0

    def test_windowing_it_yields_no_windows(self, tmp_path):
        path = _write_strand(tmp_path / "none.tsv", n=0)
        out = preprocess(read_coverage_table(path), win=100, step=100)
        assert len(out) == 0

    def test_the_empty_frame_still_carries_the_column_contract(self, tmp_path):
        # Each stage reads the previous stage's columns BY NAME, so an empty
        # frame missing them fails later and less clearly than one that has them.
        path = _write_strand(tmp_path / "none.tsv", n=0)
        out = preprocess(read_coverage_table(path), win=100, step=100)
        for col in ("win_st", "win_end", "win_len", "gc_percent",
                    "read_count_cov", "pct_redundant", "window_num",
                    "norm_raw_cov", "gc_skew", "cum_gc_skew"):
            assert col in out.columns

    def test_window_coordinates_stay_integers(self, tmp_path):
        # Built from empty LISTS, which pandas types as `object`. Left that way,
        # pd.concat with a populated frame in process_multi_genome promotes the
        # POOLED column -- so one empty plasmid would silently turn a healthy
        # chromosome's win_st into floats, all the way out to its CNV.csv.
        path = _write_strand(tmp_path / "none.tsv", n=0)
        empty = preprocess(read_coverage_table(path), win=100, step=100)
        full = preprocess(
            read_coverage_table(_write_strand(tmp_path / "some.tsv", n=500)),
            win=100, step=100,
        )
        pooled = pd.concat([empty, full], ignore_index=True)
        for col in ("win_st", "win_end", "win_len", "window_num"):
            assert pd.api.types.is_integer_dtype(pooled[col]), col
