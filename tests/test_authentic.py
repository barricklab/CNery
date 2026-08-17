"""Consistency tests against authentic breseq coverage tables.

Marked ``authentic``. These run by default; use ``pytest -m synthetic`` for the fast offline
inner loop. On a cold cache they download ~152 MB from GitHub Releases, then cache it (set
CNERY_TESTDATA_DIR to relocate). See DEVELOPER.

Six datasets, chosen to differ where it matters rather than to pile up more of the same:

    ltee_ara_p1_50k_shift   1 seq,  10 cols,  4-line footer, permuted reference, OTR fires
    ltee_ara_m3_32k_2rg     1 seq,  26 cols, 10-line footer, 2 read groups
    ltee_ara_m3_38k         1 seq,  18 cols,  7-line footer, 1 read group
    ltee_ara_p5_75k_exp     1 seq,   5 cols,  4-line footer, --total-only, OTR fires strongly
    cwbi_ssym_ht04          3 seq,   5 cols,  4-line footer, --total-only, CSV, plasmids
    adp1_mgd06_lb           1 seq,   5 cols,  4-line footer, --total-only, 3.59 Mb Acinetobacter

The footer lengths are the point of that column: 4, 10 and 7, so these are real regression
coverage for the prefix-based footer stripping that replaced ``skipfooter=4`` -- which would
have silently deleted 6 and 3 data rows from the 32k and 38k tables.

The last three rows each cover something nothing else does:

* ``ltee_ara_p5_75k_exp`` -- OTR correction FIRING strongly. Exponential phase, so replication
  forks are active and the gradient is real at 1.95x peak to trough. See TestOriginTerminus.
* ``cwbi_ssym_ht04`` -- the only MULTI-SEQUENCE dataset and the only CSV one. A chromosome plus
  two plasmids, so it is the only cover for process_multi_genome's pooled GC fit and its shared
  global median, which is what keeps the plasmids at 2.82x and 1.88x the chromosome rather than
  flattening each to 1.0. Also the only high-copy state on clean coverage: CN 33 at 59,501.
* ``adp1_mgd06_lb`` -- the first genome that is not REL606. Sequence lengths used to be a module
  constant and the window count a literal 9258; both now come from Sequence.

Tests that read a frame or an output file are parametrized per SEQUENCE, not per dataset,
because CNery's OTR and copy-number stages are per-sequence and so is their output.

These assertions record what the code does TODAY. That is deliberate: a tripwire for unintended
change, not a claim the current output is correct.

After an intended behavior change, refresh the goldens and review the diff::

    pytest -m authentic --regenerate-goldens
    git diff tests/data/expected/

Golden tests report as SKIPPED on that run, not passed -- a run that rewrote its own
expectations has verified nothing.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from CNery.core import (
    GC_SKEW_METHOD,
    STRAND_SPLIT_COLUMNS,
    TOTAL_ONLY_COLUMNS,
    _detect_delimiter,
    _read_coverage_table,
    _skew_block_length,
    apply_otr_correction,
    fit_otr_bias,
    genome_id_from_path,
    plot_gc_skew,
    predict_ori_ter_from_skew,
    preprocess,
    process_multi_genome,
    read_coverage_table,
    relative_copy_numbers,
    resolve_coverage_inputs,
    run_HMM,
    write_gc_skew_results,
)
from data._fetch import load_registry

pytestmark = pytest.mark.authentic

# Goldens are only meaningful at fixed windowing.
WIN, STEP, FRAG = 1000, 500, 150

EXPECTED_DIR = os.path.join(os.path.dirname(__file__), "data", "expected")


@dataclass(frozen=True)
class Sequence:
    """One reference sequence within a dataset.

    Everything here is a property of the sequence rather than of the run, which
    is why it cannot live on Spec: cwbi_ssym_ht04 is a chromosome plus two
    plasmids, and they differ on nearly every field below.
    """

    seq_id: str
    length: int
    relative_cn: float = 1.0     # copies relative to the run's LONGEST sequence
    expected_cn: int = 1         # modal copy number run_HMM actually calls
    otr_detected: bool = False   # does otr_fit find a replication gradient here?
    otr_tightens: bool = True    # ...and is the gradient strong enough to help?
    otr_source: str = None       # "coverage fit" or "GC skew" -- WHICH estimate won
    skew_confident: bool = True  # does the cumulative GC skew clear its own gate?
    has_deletions: bool = True  # any window at or below 10% of the median
    has_repeats: bool = True    # any window carrying redundant coverage

    @property
    def windows(self):
        """Windows preprocess() emits: only full-width ones, at stride STEP."""
        return (self.length - WIN) // STEP + 1


@dataclass(frozen=True)
class Spec:
    sequences: tuple              # of Sequence, in resolve_coverage_inputs order
    read_groups: int              # 0 = no --per-read-group columns at all
    footer_lines: int             # 4 aggregate + 3 per read group
    data_columns: int             # excluding the position index
    schema: str = "strand_split"  # or "total_only" (bam2cov --total-only)
    ending: str = "coverage.tsv"  # the file ending, and so the delimiter

    @property
    def seq_ids(self):
        return [q.seq_id for q in self.sequences]

    def sequence(self, seq_id):
        return next(q for q in self.sequences if q.seq_id == seq_id)


REL606 = 4_629_812

DATASETS = {
    # The only sequence where the coverage fit BEATS the GC skew. Both arms are
    # live and the likelihood ratio picks the coverage (p = 0.004): the two agree
    # on the origin to ~1% of the genome but put the terminus 10.6% apart, and a
    # tent hinged at the skew's terminus mis-slopes the whole curve -- 87% of the
    # coverage fit's advantage comes from OUTSIDE the disputed stretch. Worth
    # knowing that the skew is nonetheless the better landmark estimate here:
    # mapped back through the 2,314,906 bp rotation its terminus lands at 1.526 Mb
    # against REL606's dif at ~1.55 Mb, while the coverage fit's is ~490 kb off.
    # The observed coverage trough is at neither, sitting between them -- a
    # single-kink tent cannot represent a terminus REGION.
    "ltee_ara_p1_50k_shift": Spec(
        (Sequence("REL606_2314906bp_shift", REL606, otr_detected=True,
                  otr_source="coverage fit"),), 0, 4, 9),
    # Confident skew, but the coverage says the skew's ORIGIN is the low point
    # (anchor ratio 0.907), so the prediction is rejected rather than relabelled.
    "ltee_ara_m3_32k_2rg": Spec((Sequence("REL606", REL606),), 2, 10, 25),
    # Stationary phase, so there is no replication gradient in the coverage and
    # the free fit cannot clear its own gate. The skew arm still fires, because
    # the reference says an origin is there and the coverage does not contradict
    # it -- and the applied ramp (1.069) is honestly indistinguishable from noise,
    # with the coverage's own fixed-breakpoint p at 0.243. otr_tightens=False
    # records what that buys: 85.6% -> 85.3%, a wash. This is the accepted cost of
    # OTR_SKEW_MAX_P = None, and it is bounded by the amplitude being FITTED from
    # the coverage rather than imported: a flat sample yields a flat tent.
    "ltee_ara_m3_38k": Spec(
        (Sequence("REL606", REL606, otr_detected=True, otr_tightens=False,
                  otr_source="GC skew"),), 1, 7, 17),
    # The strong case, and the one that validates the whole arbitration. Both arms
    # fire and agree to 1.8% of the genome, so the likelihood ratio cannot separate
    # them (p = 0.82) and defers to the skew -- whose origin lands on REL606's oriC
    # at ~3.886 Mb, where the coverage fit was 82 kb short.
    "ltee_ara_p5_75k_exp": Spec(
        (Sequence("REL606", REL606, otr_detected=True, otr_source="GC skew"),),
        0, 4, 4, schema="total_only"),
    # The only multi-sequence dataset, and the only CSV one. The plasmids are the
    # reason it is here: process_multi_genome normalises every sequence against ONE
    # pooled median, so a multi-copy plasmid lands at a multiple of the chromosome
    # instead of being flattened to 1.0. Neither plasmid carries a deletion or a
    # replication origin. They are also heavily repeat-censored -- 111 and 114 clean
    # windows against run_HMM's min_called_windows=100 -- so a change that tightens
    # censoring drops them onto the uncensored fallback and looks like a golden move.
    # Also the only sequence reaching a high copy-number state on CLEAN coverage:
    # a 2 kb block at 59,501 sits at ~33x the pooled median with pct_redundant 0.0,
    # so n_states comes out 39. Of the 79 chromosome windows above 10x, 76 ARE
    # redundant and correctly censored -- these three are not.
    "cwbi_ssym_ht04": Spec(
        (Sequence("chromosome", 3_354_690, otr_detected=True, otr_source="GC skew"),
         # This one changed twice. The chromosome's COVERAGE fit is a marginal
         # 1.217 that fails the significance gate outright (p = 0.28), and it used
         # to be applied anyway -- a wash that moved windows within 20% of
         # single-copy from 91.8% to 91.5%. The GC-skew arm now supplies the
         # breakpoints instead, at ratio 1.169, and the correction finally helps
         # (91.8% -> 92.5%), which is why otr_tightens is no longer False here.
         # Neither plasmid clears the GC-skew gate, which is the expected answer:
         # a plasmid has no bidirectional replication origin, so there is no sign
         # change for the cumulative curve to turn on. Both are rejected by the
         # bootstrap (p = 0.36 and 0.44) rather than by their size -- plasmid_1
         # even lands inside the 35-65% separation band at 0.444, so separation
         # alone would pass it, and both get an adequate 21 and 16 blocks, so the
         # rejection is on evidence rather than on having too little data.
         Sequence("plasmid_1", 116_754, relative_cn=2.9531, has_deletions=False,
                  skew_confident=False),
         Sequence("plasmid_2", 82_656, relative_cn=1.8980, has_deletions=False,
                  skew_confident=False)),
        0, 4, 4, schema="total_only", ending="coverage.csv"),
    # Like m3_38k, corrected on the skew's coordinates rather than the coverage's
    # -- the free fit's optimum is a degenerate spike 1.1% of the genome wide,
    # nowhere near the 35-65% band. LB culture, so some fork activity is at least
    # plausible, and the correction does tighten (94.1% -> 95.0%); but the
    # coverage's own evidence at those breakpoints is p = 0.615, so read the ramp
    # (1.067) as sequence-justified rather than coverage-justified.
    "adp1_mgd06_lb": Spec(
        (Sequence("ADP1-ISx", 3_592_307, otr_detected=True, otr_source="GC skew"),),
        0, 4, 4, schema="total_only"),
}

ALL = list(DATASETS)
WITH_READ_GROUPS = [n for n, s in DATASETS.items() if s.read_groups]

#: (dataset, seq_id) for every sequence in every dataset. Tests that read a frame
#: or an output file are parametrized over this rather than over datasets, because
#: CNery's OTR and copy-number stages are per-sequence and so is their output.
ALL_SEQUENCES = [(n, q.seq_id) for n, s in DATASETS.items() for q in s.sequences]


def per_sequence(cls):
    """Parametrize a class over every (dataset, sequence) pair.

    `dataset_dir` is parametrized alongside `seq` because the session-scoped
    `pipeline` hangs off it. The three cwbi_ssym_ht04 sequences share one
    `dataset_dir` param value, so pytest caches the pipeline run and it happens
    once per dataset rather than once per sequence.
    """
    return pytest.mark.parametrize(
        "dataset_dir,seq",
        [(n, (n, q)) for n, q in ALL_SEQUENCES],
        indirect=True,
        ids=[f"{n}:{q}" for n, q in ALL_SEQUENCES],
    )(cls)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def dataset_dir(request):
    """Extracted directory for the dataset named by the parametrized `request.param`."""
    from conftest import _dataset_or_skip

    return request.param, _dataset_or_skip(request.param)


def _tables(path, name):
    """{seq_id: table path} for a dataset, by the same route CNery uses.

    Not coverage_table_path(), whose suffix defaults to ".coverage.tsv" -- that is
    wrong for a CSV dataset, and it cannot find more than one sequence anyway.
    """
    return resolve_coverage_inputs([str(path)])


@pytest.fixture(scope="session")
def raw_tables(dataset_dir):
    """{seq_id: raw parsed frame} for every sequence in the dataset."""
    name, path = dataset_dir
    return name, {seq_id: _read_coverage_table(p)
                  for seq_id, p in _tables(path, name).items()}


#: Pipeline results by dataset name. Keyed explicitly rather than leaning on pytest's
#: per-param caching of a session-scoped fixture: `dataset_dir` is parametrized jointly
#: with `seq` (see per_sequence), and under that pairing pytest rebuilds the fixture for
#: nearly every test. That took the authentic tier from 45s to 9m44s -- ~130 rebuilds at
#: ~4.5s each -- and it is invisible in the results, so it is worth pinning here.
_PIPELINES = {}


def _run_pipeline(name, path, out):
    """process_multi_genome once, then OTR + HMM per sequence, mirroring get_CNV.main()."""
    for sub in ("CNV_plt", "CNV_csv", "GC_bias", "OTR_corr", "GC_skew"):
        (out / sub).mkdir()

    per_genome = process_multi_genome(
        resolve_coverage_inputs([str(path)]),
        output_prefix=str(out),
        win=WIN, step=STEP, frag=FRAG,
    )

    # Mirrors get_CNV.main(): one number per sequence, computed across ALL of them
    # because apply_otr_correction() runs per sequence and cannot see the others.
    relative_cn = relative_copy_numbers(per_genome)

    frames = {}
    for seq_id, df_gc in per_genome.items():
        # df_gc already carries is_deletion/is_redundant from the
        # mask_coverage_windows() call inside process_multi_genome()'s GC stage,
        # so fit_otr_bias() reuses them.
        # Ahead of the OTR fit, exactly as get_CNV.main() places it. This
        # ORDERING IS LOAD-BEARING now: it reads only ref_base, so it is
        # available whatever the coverage does, and fit_otr_bias() takes it as
        # the second breakpoint candidate. Computing it afterwards -- as this
        # harness used to -- silently runs the coverage-only arm and stops
        # mirroring main().
        skew = predict_ori_ter_from_skew(df_gc, win=WIN, step=STEP)
        write_gc_skew_results(skew, str(out), seq_id)
        plot_gc_skew(df_gc, str(out), skew)

        df_otr, ori, ter = apply_otr_correction(
            fit_otr_bias(df_gc, str(out), skew_result=skew), str(out),
            relative_copy_number=relative_cn[seq_id],
        )

        frames[seq_id] = {"gc": df_gc, "otr": df_otr,
                          "cnv": run_HMM(df_otr, str(out)), "skew": skew}

    return {"name": name, "spec": DATASETS[name], "out": out, "frames": frames}


@pytest.fixture(scope="session")
def pipeline(dataset_dir, tmp_path_factory):
    """The full pipeline for one dataset, run at most once per session."""
    name, path = dataset_dir
    if name not in _PIPELINES:
        _PIPELINES[name] = _run_pipeline(name, path, tmp_path_factory.mktemp(f"out_{name}"))
    return _PIPELINES[name]


@pytest.fixture
def seq(request, pipeline):
    """One sequence's frames plus its Sequence spec.

    Parametrized indirectly over ALL_SEQUENCES; `pipeline` stays session-scoped so
    the expensive run happens once per dataset, not once per sequence.
    """
    _, seq_id = request.param
    return {
        "name": pipeline["name"],
        "spec": pipeline["spec"],
        "seq": pipeline["spec"].sequence(seq_id),
        "seq_id": seq_id,
        "out": pipeline["out"],
        **pipeline["frames"][seq_id],
    }


def _produced(out, seq_id, subdir, suffix):
    """The path CNery writes for one sequence: <out basename><seq_id><suffix>."""
    prefix = os.path.basename(str(out))
    return os.path.join(str(out), subdir, f"{prefix}{seq_id}{suffix}")


def _coverage_columns(name):
    """The coverage columns this dataset's schema is required to carry.

    bam2cov writes either strand-split counts or, under --total-only, the sums
    it has already taken. The raw frames are read with _read_coverage_table,
    which deliberately does NOT normalize, so the expectation has to follow the
    schema rather than assume one of them.
    """
    if DATASETS[name].schema == "total_only":
        return TOTAL_ONLY_COLUMNS
    return STRAND_SPLIT_COLUMNS


def _golden(name, seq_id, suffix):
    """Goldens carry the sequence id, because CNery writes one file per sequence."""
    return os.path.join(EXPECTED_DIR, f"{name}_{seq_id}{suffix}")


# --------------------------------------------------------------------------- fetch


@pytest.mark.parametrize("dataset_dir", ALL, indirect=True)
class TestFetch:
    """Proves the published assets are what the tests actually run on.

    Without this a green run only shows that *some* bytes were on disk; a stale or
    hand-placed cache would look identical to a working download.
    """

    def _cached(self, name):
        import pooch

        from data._fetch import CACHE_ENV_VAR

        entry = load_registry()[name]
        cache = os.environ.get(CACHE_ENV_VAR) or str(pooch.os_cache("cnery"))
        return os.path.join(cache, entry["file"]), entry

    def test_archive_landed_in_the_cache(self, dataset_dir):
        path, _ = self._cached(dataset_dir[0])
        assert os.path.isfile(path)

    def test_cached_archive_matches_the_registry_hash(self, dataset_dir):
        # pooch verifies on download; re-checking catches later corruption of a warm cache,
        # which no download-time check can see.
        import hashlib

        path, entry = self._cached(dataset_dir[0])
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        assert digest.hexdigest() == entry["sha256"]

    def test_archive_size_matches_the_registry(self, dataset_dir):
        path, entry = self._cached(dataset_dir[0])
        assert os.path.getsize(path) == entry["bytes"]


# --------------------------------------------------------------------------- input integrity


@pytest.mark.parametrize("dataset_dir", ALL, indirect=True)
class TestInputIntegrity:
    def test_no_bam_is_shipped(self, dataset_dir):
        _, path = dataset_dir
        assert not os.path.exists(os.path.join(str(path), "reference.bam"))

    def test_every_position_survives_parsing(self, raw_tables):
        # The direct regression for skipfooter=4: these tables have 4-, 7- and 10-line
        # footers, so any fixed-count skip loses real data from at least two of them.
        name, frames = raw_tables
        for sequence in DATASETS[name].sequences:
            df = frames[sequence.seq_id]
            assert len(df) == sequence.length
            assert df.index[0] == 1
            assert df.index[-1] == sequence.length

    def test_position_is_the_index(self, raw_tables):
        for df in raw_tables[1].values():
            assert df.index.name == "position"

    def test_footer_did_not_leak_into_the_data(self, raw_tables):
        name, frames = raw_tables
        for df in frames.values():
            assert not df["ref_base"].astype(str).str.startswith("#").any()
            assert df[_coverage_columns(name)[0]].dtype.kind in "iu"

    def test_columns_cnery_depends_on(self, raw_tables):
        name, frames = raw_tables
        for df in frames.values():
            for col in ("ref_base",) + _coverage_columns(name):
                assert col in df.columns

    def test_column_count_matches_spec(self, raw_tables):
        name, frames = raw_tables
        for df in frames.values():
            assert df.shape[1] == DATASETS[name].data_columns

    def test_seq_ids_are_recoverable_from_the_file_names(self, dataset_dir):
        # Sequence IDs come from the file name rather than a FASTA header, so the
        # naming rule has to hold against real file names -- including
        # "REL606_2314906bp_shift", which the "strip only the ending" rule must leave
        # intact rather than cutting at a dot, and the .csv endings of cwbi_ssym_ht04.
        name, path = dataset_dir
        for seq_id, table in _tables(path, name).items():
            assert os.path.isfile(table)
            assert genome_id_from_path(table) == seq_id

    def test_resolving_the_folder_finds_every_declared_sequence(self, dataset_dir):
        # Stronger than "exactly one table": the resolved set must be the whole
        # declared list, in resolve_coverage_inputs' sorted-by-filename order.
        name, path = dataset_dir
        assert list(resolve_coverage_inputs([str(path)])) == DATASETS[name].seq_ids

    def test_delimiter_is_detected_from_the_bytes(self, dataset_dir):
        # cwbi_ssym_ht04 is the only real CSV in the collection; every other CSV test
        # in the suite runs on a synthetic table of a few thousand bases.
        name, path = dataset_dir
        want = "," if DATASETS[name].ending.endswith(".csv") else "\t"
        for table in _tables(path, name).values():
            assert _detect_delimiter(table) == want


# --------------------------------------------------------------------------- read groups


@pytest.mark.parametrize("dataset_dir", WITH_READ_GROUPS, indirect=True)
class TestPerReadGroupColumns:
    """--per-read-group appends columns; CNery must read past them untouched."""

    def test_read_group_columns_are_present(self, raw_tables):
        name, frames = raw_tables
        for df in frames.values():
            for g in range(DATASETS[name].read_groups):
                assert f"RG-{g}_unique_top_cov" in df.columns

    def test_aggregate_columns_keep_their_names(self, raw_tables):
        # breseq appends the repeats after the aggregates precisely so CNery keeps working;
        # if that ever changed to interleaving, this is what would catch it.
        for df in raw_tables[1].values():
            assert list(df.columns)[:5] == [
                "ref_base", "unique_top_cov", "unique_bot_cov",
                "redundant_top_cov", "redundant_bot_cov",
            ]

    def test_calls_ignore_the_extra_columns(self, raw_tables, pipeline):
        # Windowing the aggregate-only subset must give exactly what the full table gives.
        name, frames = raw_tables
        for seq_id, df in frames.items():
            trimmed = df[["ref_base", "unique_top_cov", "unique_bot_cov",
                          "redundant_top_cov", "redundant_bot_cov"]]
            got = preprocess(trimmed, win=WIN, step=STEP, frag=FRAG)
            want = pipeline["frames"][seq_id]["gc"]
            assert len(got) == len(want)
            np.testing.assert_allclose(
                got["read_count_cov"].to_numpy(),
                want["read_count_cov"].to_numpy(),
            )


# --------------------------------------------------------------------------- pipeline shape


@per_sequence
class TestPipelineShape:
    def test_row_count_preserved_end_to_end(self, seq):
        assert len(seq["cnv"]) == len(seq["gc"])

    def test_genome_id_survives(self, seq):
        assert (seq["cnv"]["genome_id"] == seq["seq_id"]).all()

    def test_no_nan_in_calls(self, seq):
        assert not seq["cnv"]["prob_copy_number"].isna().any()

    def test_corrected_coverage_is_finite_and_non_negative(self, seq):
        values = seq["cnv"]["otr_gc_corr_norm_cov"].to_numpy()
        assert np.isfinite(values).all()
        assert (values >= 0).all()

    def test_window_count_is_stable(self, seq):
        # preprocess() keeps every repeat-overlapping window (flagging pct_redundant
        # instead of dropping it) and emits only full-width windows, so the count is
        # exactly the number of `win`-wide windows that fit at stride `step`.
        assert len(seq["gc"]) == seq["seq"].windows

    def test_repeat_windows_are_retained_and_flagged(self, seq):
        # Windows over repeat content must be present in the frame, not dropped, and
        # carry a nonzero pct_redundant. ADP1-ISx has had its IS elements removed by
        # construction, so it has far fewer than the LTEE clones -- but not none.
        df = seq["gc"]
        assert "pct_redundant" in df.columns
        assert df["is_redundant"].equals(df["pct_redundant"] > 0)
        if seq["seq"].has_repeats:
            assert (df["pct_redundant"] > 0).any()

    def test_gc_correction_does_not_resurrect_deletions(self, seq):
        if not seq["seq"].has_deletions:
            pytest.skip("no deletions on this sequence (neither CWBI plasmid has one)")
        df = seq["gc"]
        deleted = df["read_count_cov"] <= df["read_count_cov"].median() * 0.1
        assert deleted.any(), "sequence should contain real deletions"
        assert (df.loc[deleted, "gc_corr_norm_cov"] == 0.0).all()


# --------------------------------------------------------------------------- OTR


@per_sequence
class TestOriginTerminus:
    """Whether OTR correction fires, per sequence, WHICH estimate supplied the
    breakpoints, and that it does real work when it does.

    There are now two candidates and three gates between them.

    The COVERAGE fit must land 35-65% of the genome apart AND beat a circular
    block bootstrap at p <= 0.01. That second condition is new and is the whole
    point: the old `bias_threshold` was vacuous (the label swap in otr_fit()
    guarantees y_ori >= y_ter, so "ratio > 1.0" reduced to "y_ter > 0"), so
    nothing ever tested the tent against a null and a flat genome whose best-fit
    tent happened to be antipodal had its coverage divided by noise. CWBI's
    chromosome was exactly that case at a marginal 1.217.

    The GC-SKEW fit is admitted whenever the skew's own bootstrap called the
    prediction confident and the coverage does not contradict its ORIENTATION.
    It needs no coverage significance of its own (OTR_SKEW_MAX_P is None), which
    is a deliberate decision with a visible cost -- see ltee_ara_m3_38k and
    adp1_mgd06_lb in DATASETS above.

    When BOTH are live a bootstrap likelihood-ratio test decides, and only then:
    the two models are nested, differing by exactly the two breakpoint
    parameters. On this corpus that happens on two sequences and splits them,
    which is why `otr_source` is recorded per sequence and not assumed.

    Note what this costs the suite: TestGCSkewOriginTerminus is NO LONGER an
    independent cross-check on these coordinates, because the skew now feeds the
    OTR fit. What it still independently establishes is that the skew estimate is
    a function of the reference sequence alone -- which is what makes it usable
    on stationary-phase samples where the coverage carries no gradient, and also
    what makes ramp injection there a real possibility rather than a hypothetical.

    These assertions fail if detection starts OR stops happening anywhere, or if
    the winning estimate changes -- all three are real changes worth noticing.
    """

    def test_detection_matches_the_spec(self, seq):
        # Read what the run PRODUCED, not the golden -- asserting that the golden says what
        # the golden says would be circular and would pass no matter what the code did.
        with open(_produced(seq["out"], seq["seq_id"], "OTR_corr", "_otr_results.json")) as fh:
            ratio = json.load(fh)["Origin-to-Termius/Bias Ratio"]
        if seq["seq"].otr_detected:
            assert ratio != "Not detected", "expected OTR correction to fire"
            assert float(ratio) > 1.0, f"origin should out-cover terminus, got {ratio}"
        else:
            assert ratio == "Not detected"

    def test_correction_source_matches_the_spec(self, seq):
        """WHICH estimate supplied the breakpoints, not merely whether one did.

        Recorded per sequence because the two arms answer for different reasons.
        ltee_ara_p5_75k_exp is the case worth understanding: BOTH fire and agree
        to 1.8% of the genome, so the likelihood ratio cannot separate them
        (p = 0.82) and defers to the GC skew as the better-supported estimate of
        the same thing -- REL606's oriC is at ~3.886 Mb, where the skew lands,
        and the coverage fit was 82 kb short of it.

        ltee_ara_p1_50k_shift is the other end: both fire, but they place the
        terminus 10.6% of the genome apart and the coverage fit explains
        materially more (p = 0.004), so it keeps its own breakpoints.
        """
        with open(_produced(seq["out"], seq["seq_id"], "OTR_corr", "_otr_results.json")) as fh:
            data = json.load(fh)
        expected = seq["seq"].otr_source or "not corrected"
        assert data["Breakpoint source"] == expected
        # "Correction type" is how a reader of the file alone tells them apart.
        if expected == "GC skew":
            assert data["Correction type"] == GC_SKEW_METHOD
        else:
            assert data["Correction type"] == "Ori-ter coordinates fit by coverage"

    def test_skew_sourced_coordinates_agree_with_the_skew_file(self, seq):
        """The two JSONs now share a coordinate and must not be able to disagree."""
        if seq["seq"].otr_source != "GC skew":
            pytest.skip("breakpoints did not come from the GC skew here")
        with open(_produced(seq["out"], seq["seq_id"], "OTR_corr", "_otr_results.json")) as fh:
            otr = json.load(fh)
        with open(_produced(seq["out"], seq["seq_id"], "GC_skew", "_gc_skew_results.json")) as fh:
            skew = json.load(fh)
        assert otr["Origin window"] == skew["Origin (bp)"]
        assert otr["Terminus window"] == skew["Terminus (bp)"]

    def test_coverage_passes_through_when_no_bias_is_found(self, seq):
        # With no bias detected, otr_correction must leave the GC-corrected values alone.
        if seq["seq"].otr_detected:
            pytest.skip("OTR fires here; see test_correction_tightens_coverage")
        df = seq["otr"]
        np.testing.assert_allclose(
            df["otr_gc_corr_norm_cov"].to_numpy(),
            df["gc_corr_norm_cov"].to_numpy(),
            rtol=1e-9,
        )

    def test_correction_tightens_coverage(self, seq):
        """The correction must pull coverage TOWARD the single-copy level.

        Reporting a ratio is not evidence that anything was corrected. On
        ltee_ara_p5_75k_exp the fraction of windows within 20% of single-copy goes from
        53% to 95%.

        This is also the regression test for the ori/ter label swap: otr_fit's objective is
        symmetric under exchanging the two breakpoints, so a fit that comes back mirrored
        divides by an INVERTED ramp and spreads coverage out instead of tightening it.
        """
        if not seq["seq"].otr_detected:
            pytest.skip("no OTR correction applied on this sequence")
        df = seq["otr"]
        near = lambda v: float(((v > 0.8) & (v < 1.2)).mean())
        before = near(df["gc_corr_norm_cov"].to_numpy())
        after = near(df["otr_gc_corr_norm_cov"].to_numpy())

        # An inverted ramp is not subtle -- it divides by the reciprocal of the real
        # gradient, so it costs tens of points. This bound catches that on every
        # sequence, including ones whose gradient is too weak to gain anything.
        assert after > before - 0.05, (
            f"OTR correction spread coverage out: {before:.1%} -> {after:.1%}. "
            "An inverted ori/ter labelling would do exactly this."
        )
        if seq["seq"].otr_tightens:
            assert after > before, f"expected tightening, got {before:.1%} -> {after:.1%}"
            assert after > 0.85

    def test_results_json_is_strict_json(self, seq):
        """No bare NaN or Infinity -- breseq's parser is strict and rejects the file.

        breseq reads this with nlohmann/json, which has no allow_nan, so ONE
        non-finite value makes the whole file unparseable: it warns and reports no
        ori-ter bias. That was true of every "Not detected" file CNery wrote, because
        the origin/terminus coverages are NaN there. json.dump emits bare NaN by
        default, so this guards the fix.
        """
        def reject(token):
            raise AssertionError(f"non-finite {token!r} in the results JSON")

        with open(_produced(seq["out"], seq["seq_id"], "OTR_corr", "_otr_results.json")) as fh:
            json.loads(fh.read(), parse_constant=reject)

    def test_otr_json_matches_golden(self, seq, regenerate_goldens):
        from conftest import golden_compare

        def compare(got_path, want_path):
            with open(got_path) as fh:
                got = json.load(fh)
            with open(want_path) as fh:
                want = json.load(fh)
            assert got["Origin-to-Termius/Bias Ratio"] == want["Origin-to-Termius/Bias Ratio"]
            assert got["Origin window"] == want["Origin window"]
            assert got["Terminus window"] == want["Terminus window"]
            # A float, so compared with tolerance -- LOWESS and scipy.optimize drift
            # across library versions.
            assert got["Relative copy number"] == pytest.approx(
                want["Relative copy number"], rel=1e-6)

        golden_compare(
            _produced(seq["out"], seq["seq_id"], "OTR_corr", "_otr_results.json"),
            _golden(seq["name"], seq["seq_id"], "_otr_results.json"),
            regenerate_goldens, compare,
        )


# --------------------------------------------------------------------------- GC skew


# Coordinate rotation baked into the p1_50k_shift reference, and named in its
# sequence ID (REL606_2314906bp_shift): adding it to a coordinate in that
# reference gives the same locus in the unrotated REL606 references.
SHIFT_BP = 2_314_906
SHIFTED = "ltee_ara_p1_50k_shift"

#: Every dataset whose reference is REL606 in its published orientation. Three
#: different clones across two growth phases, which is what makes
#: test_same_reference_agrees_exactly a real test rather than a tautology.
UNSHIFTED = ["ltee_ara_m3_38k", "ltee_ara_m3_32k_2rg", "ltee_ara_p5_75k_exp"]


def _circular_delta(a, b, length=REL606):
    d = abs(a - b) % length
    return min(d, length - d)


@pytest.fixture(scope="session")
def skew_by_sequence():
    """{(dataset, seq_id): prediction} for every sequence in every dataset.

    Built from preprocess() alone rather than from the `pipeline` fixture: the
    cross-dataset comparisons below are about the prediction being a function of
    the reference and nothing else, so they should not run through the
    bias-correction stages at all.
    """
    from conftest import _dataset_or_skip

    out = {}
    for name, spec in DATASETS.items():
        path = _dataset_or_skip(name)
        for seq_id, table in _tables(path, name).items():
            df = preprocess(read_coverage_table(table),
                            win=WIN, step=STEP, frag=FRAG)
            out[(name, seq_id)] = predict_ori_ter_from_skew(df, win=WIN, step=STEP)
    return out


@per_sequence
class TestGCSkewArtifacts:
    def test_json_and_plot_are_written(self, seq):
        for suffix in ("_gc_skew_results.json", "_GC_skew.pdf"):
            assert os.path.exists(
                _produced(seq["out"], seq["seq_id"], "GC_skew", suffix)
            ), suffix

    def test_confidence_matches_the_spec(self, seq):
        # Recorded per sequence rather than blanket-asserted, for the same reason
        # otr_detected is: a plasmid has no replication origin, so there is no
        # sign change for the cumulative curve to turn on.
        assert seq["skew"]["Prediction confident"] is seq["seq"].skew_confident

    def test_confident_predictions_are_near_antipodal(self, seq):
        if not seq["seq"].skew_confident:
            pytest.skip("no confident skew prediction on this sequence")
        assert 0.35 <= seq["skew"]["Separation (fraction of genome)"] <= 0.65

    def test_p_value_agrees_with_the_verdict(self, seq):
        # Every chromosome exhausts the bootstrap and reads back its floor of
        # 1/(B+1); neither plasmid comes close (0.36 and 0.44). Asserted as a
        # band rather than exactly, since p is a Monte Carlo estimate.
        p = seq["skew"]["Replichore skew p-value"]
        floor = 1 / (seq["skew"]["Bootstrap surrogates"] + 1)
        if seq["seq"].skew_confident:
            assert p == pytest.approx(floor, rel=0.01), "expected p at its floor"
        else:
            assert p > 0.1, "a sequence with no origin should be nowhere near significant"

    def test_plasmids_are_rejected_on_evidence_not_on_size(self, seq):
        # The bootstrap replaced a minimum-window gate, so it matters that the
        # plasmids fail for the right reason. Both get an adequate number of
        # blocks (21 and 16) and still land at p > 0.1 -- they are rejected
        # because there is no replichore structure, not because they are short.
        if seq["seq"].skew_confident:
            pytest.skip("this sequence has a confident prediction")
        n = seq["skew"]["Windows"]
        assert n / _skew_block_length(n) >= 10, "too few blocks to conclude anything"

    def test_skew_json_matches_golden(self, seq, regenerate_goldens):
        from conftest import golden_compare

        def compare(got_path, want_path):
            with open(got_path) as fh:
                got = json.load(fh)
            with open(want_path) as fh:
                want = json.load(fh)
            # Integer coordinates and the verdict only. The floats (separation,
            # amplitude, t) are left out deliberately -- they are summary
            # statistics that drift with library versions, and the structural
            # facts are what a reviewer needs to see change.
            for key in ("Origin (bp)", "Terminus (bp)",
                        "Origin window index", "Terminus window index",
                        "Windows", "Bootstrap surrogates", "Prediction confident"):
                assert got[key] == want[key], key

        golden_compare(
            _produced(seq["out"], seq["seq_id"], "GC_skew", "_gc_skew_results.json"),
            _golden(seq["name"], seq["seq_id"], "_gc_skew_results.json"),
            regenerate_goldens, compare,
        )


class TestGCSkewOriginTerminus:
    """The prediction is a function of the reference sequence and nothing else.

    That the SAMPLE cannot matter is true by construction -- the estimate reads
    only `ref_base` -- and the synthetic tier pins it directly at two depths
    (test_gc_skew.py::test_prediction_is_independent_of_coverage_depth). What is
    worth checking on real tables is narrower: that the sequence CNery recovers
    is the same one no matter how the table around it is SHAPED. The three
    unrotated REL606 datasets differ in schema (strand-split vs --total-only),
    in read-group column count (0, 1, 2) and in footer length (4, 7, 10), so a
    parsing change that shifted ref_base by a row would split them.

    The rotated dataset tests the other property: the same sequence cut at a
    different coordinate must still predict the same locus, which is what
    subtracting the mean before cumulating buys.
    """

    def test_same_reference_agrees_exactly(self, skew_by_sequence):
        results = [skew_by_sequence[(n, "REL606")] for n in UNSHIFTED]
        first = results[0]
        for other, name in zip(results[1:], UNSHIFTED[1:]):
            assert other["Origin (bp)"] == first["Origin (bp)"], name
            assert other["Terminus (bp)"] == first["Terminus (bp)"], name

    @pytest.mark.parametrize("key", ["Origin (bp)", "Terminus (bp)"])
    def test_permuted_reference_predicts_the_same_locus(self, skew_by_sequence, key):
        mapped = skew_by_sequence[(SHIFTED, "REL606_2314906bp_shift")][key] + SHIFT_BP
        reference = skew_by_sequence[(UNSHIFTED[0], "REL606")][key]
        delta = _circular_delta(mapped, reference)
        assert delta <= WIN, (
            f"{key}: rotated reference maps to {mapped % REL606}, "
            f"unrotated says {reference} ({delta} bp apart)"
        )



# --------------------------------------------------------------------------- copy number


@per_sequence
class TestCopyNumber:
    def test_segments_match_golden(self, seq, regenerate_goldens):
        from conftest import golden_compare

        def compare(got_path, want_path):
            pd.testing.assert_frame_equal(pd.read_csv(got_path), pd.read_csv(want_path))

        golden_compare(
            _produced(seq["out"], seq["seq_id"], "CNV_csv", "_break_pts.csv"),
            _golden(seq["name"], seq["seq_id"], "_break_pts.csv"),
            regenerate_goldens, compare,
        )

    def test_modal_copy_number_matches_the_spec(self, seq):
        calls = seq["cnv"]["prob_copy_number"]
        assert calls.mode().iloc[0] == seq["seq"].expected_cn
        assert (calls == seq["seq"].expected_cn).mean() > 0.8

    def test_relative_copy_number_is_reported(self, seq):
        """Every sequence publishes its copies relative to the longest one.

        This is the number run_HMM cannot give: it refits the single-copy level from
        whichever sequence it is handed -- fitted mu 100.9, 300.0 and 194.6 for CWBI's
        three -- so a plasmid sitting wholly above the chromosome is called copy number
        1. The multiple is computed by the pooled normalisation and would otherwise be
        discarded. Deliberately non-integral: 2.95 is a measurement.
        """
        with open(_produced(seq["out"], seq["seq_id"], "OTR_corr", "_otr_results.json")) as fh:
            got = json.load(fh)["Relative copy number"]
        assert isinstance(got, float)
        assert got == pytest.approx(seq["seq"].relative_cn, rel=0.02)

    def test_longest_sequence_is_the_anchor(self, seq):
        """The longest sequence reads exactly 1.0, so the value is 'copies relative to
        the chromosome' rather than 'relative to a pooled median it happens to dominate'."""
        longest = max(seq["spec"].sequences, key=lambda q: q.length)
        if seq["seq_id"] != longest.seq_id:
            pytest.skip("not the longest sequence in this dataset")
        with open(_produced(seq["out"], seq["seq_id"], "OTR_corr", "_otr_results.json")) as fh:
            assert json.load(fh)["Relative copy number"] == pytest.approx(1.0)

    def test_no_segment_starts_on_a_repeat_window(self, seq):
        """Repeat windows are censored from the observation sequence, so no segment
        boundary can be placed on one -- a pile-up must not be able to open a segment."""
        df = seq["cnv"]
        redundant_starts = set(df.loc[df["is_redundant"], "win_st"])
        produced = pd.read_csv(
            _produced(seq["out"], seq["seq_id"], "CNV_csv", "_break_pts.csv"))
        # The first segment starts at 0, which is a coordinate rather than a window.
        offenders = sorted(set(produced["Startpos"][1:]) & redundant_starts)
        assert not offenders, f"segments start on repeat windows: {offenders}"

    def test_repeat_windows_inherit_a_call(self, seq):
        if not seq["seq"].has_repeats:
            pytest.skip("no repeat content on this sequence")
        df = seq["cnv"]
        assert df["is_redundant"].any()
        assert not df.loc[df["is_redundant"], "prob_copy_number"].isna().any()

    def test_deletions_are_called(self, seq):
        if not seq["seq"].has_deletions:
            pytest.skip("no deletions on this sequence (neither CWBI plasmid has one)")
        assert 0 in set(seq["cnv"]["prob_copy_number"].unique()), "expected called deletions"
