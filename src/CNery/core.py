#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import ndimage
import matplotlib as mplt
import matplotlib.patches as mpatches
from scipy.stats import geom, nbinom
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import  gammaln
from scipy.optimize import minimize
from itertools import cycle, islice


# breseq bam2cov writes either CSV or TSV; both are picked up without being asked for,
# since the two differ only in their delimiter and CNery detects that from the file itself.
# ".coverage.tab" is the same TSV content under the legacy extension that the deprecated
# --table (-t) flag still writes, so it is accepted too.
DEFAULT_FILE_ENDINGS = ("coverage.csv", "coverage.tsv", "coverage.tab")

COVERAGE_TABLE_SUFFIX = ".coverage.tsv"

# Every coverage table carries these two, whichever coverage schema it uses. `ref_base` is
# the one that matters most: it holds the reference sequence, which is where GC content
# comes from -- CNery never reads a FASTA.
BASE_COVERAGE_COLUMNS = ("ref_base",)

# The two coverage schemas bam2cov emits. Plain output splits each count by strand;
# --total-only sums the strands in C++ and writes the pair CNery actually wants, plus a
# `total_cov` that is just their sum. Both reduce to the same two canonical columns, so
# --total-only loses nothing CNery uses -- see normalize_coverage_columns().
STRAND_SPLIT_COLUMNS = (
    "unique_top_cov",
    "unique_bot_cov",
    "redundant_top_cov",
    "redundant_bot_cov",
)
TOTAL_ONLY_COLUMNS = ("unique_cov", "redundant_cov")


def sample_prefix(output):
    """The `<sample>` every output filename is prefixed with, from the output dir.

    rstrip, not strip: the default output directory is "CNV_out/" WITH a trailing
    slash (get_CNV.main()), and `"CNV_out/".split("/")[-1]` is the empty string.
    So the JSON writers used to drop the prefix entirely under the default -o
    while run_HMM's CSVs kept it -- OTR_corr/chrA_otr_results.json beside
    CNV_csv/CNV_outchrA_CNV.csv, from one invocation. A file breseq cannot find
    is worth no more than one that was never written, and the name should not
    depend on how the caller happened to spell -o.
    """
    return str(output).rstrip("/").split("/")[-1]


def coverage_table_path(coverage_dir, seq_id, suffix=COVERAGE_TABLE_SUFFIX):
    """Path of the coverage table for `seq_id` inside `coverage_dir`.

    Named for the sequence ID alone -- no coordinates. breseq's default no-region output
    names files after the full region ("REL606:1-4629812.tsv"), which is not what CNery
    looks for and, more importantly, contains a colon: illegal in filenames on Windows and
    displayed as "/" by the macOS Finder.
    """
    return os.path.join(coverage_dir, f"{seq_id}{suffix}")


def normalize_file_endings(file_endings=None):
    """The endings to match, as a list, with any leading dots removed.

    None means "the default", so a caller can pass argparse's unset value straight
    through: supplying --file-ending REPLACES the defaults rather than adding to them.
    """
    if file_endings is None:
        file_endings = list(DEFAULT_FILE_ENDINGS)
    if isinstance(file_endings, str):
        file_endings = [file_endings]
    return [str(e).lstrip(".") for e in file_endings if str(e).strip()]


def genome_id_from_path(path, file_endings=None):
    """Sequence ID for a coverage table: its basename minus the matched file ending.

    Only the ending and the single "." in front of it are removed -- NOT everything
    after the first dot -- so "my.sample.1.coverage.tsv" gives "my.sample.1". Sequence
    IDs routinely contain dots, and truncating at the first one would silently merge
    distinct references into one genome_id.

    Endings are tried in the order given, so an earlier one wins over a later one that
    also matches. A file named explicitly on the command line need not match any ending
    at all; then only its final extension is dropped.
    """
    name = os.path.basename(path)
    for ending in normalize_file_endings(file_endings):
        suffix = "." + ending
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def matches_file_ending(name, file_endings=None):
    """Whether `name` ends in one of `file_endings`, preceded by a "."."""
    return any(
        name.endswith("." + ending)
        for ending in normalize_file_endings(file_endings)
    )


def resolve_coverage_inputs(paths, file_endings=None):
    """Expand command-line inputs into an ordered {genome_id: path} mapping.

    `paths` may mix files and directories. A file is taken as given, whatever it is
    named; a directory contributes the files inside it -- top level only, no recursion --
    whose names end in one of `file_endings`. Directory listings are sorted so a run is
    reproducible regardless of filesystem order.

    Every problem is reported before any table is read, so a bad invocation cannot leave
    half a run's worth of output behind. Directories are not searched recursively on
    purpose: a stale copy of a table one level down would otherwise collide with the live
    one and turn a working command into a duplicate-ID error.
    """
    endings = normalize_file_endings(file_endings)
    tried = ", ".join(endings) if endings else "(none)"

    missing = []
    empty_dirs = []
    resolved = {}
    origin = {}

    for path in paths:
        if not os.path.exists(path):
            missing.append(path)
            continue

        if os.path.isdir(path):
            found = [
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if matches_file_ending(name, endings)
                and os.path.isfile(os.path.join(path, name))
            ]
            if not found:
                empty_dirs.append(path)
                continue
        elif os.path.isfile(path):
            found = [path]
        else:
            missing.append(path)
            continue

        for table in found:
            genome_id = genome_id_from_path(table, endings)
            if genome_id in resolved:
                raise ValueError(
                    f"Duplicate sequence ID {genome_id!r} from two inputs:\n"
                    f"  {origin[genome_id]}\n"
                    f"  {table}\n"
                    "Sequence IDs come from the file name, so two tables cannot share "
                    "one. Rename a file or run them separately."
                )
            resolved[genome_id] = table
            origin[genome_id] = table

    if missing:
        raise FileNotFoundError(
            "No such input path(s): " + ", ".join(missing)
        )
    if empty_dirs:
        raise FileNotFoundError(
            "No coverage tables in: " + ", ".join(empty_dirs)
            + f". Looked for files ending in: {tried}. "
            "Use --file-ending to match a different name."
        )
    if not resolved:
        raise FileNotFoundError(
            f"No coverage tables found in: {', '.join(paths)}. "
            f"Looked for files ending in: {tried}."
        )

    return resolved


def parse_region(text):
    """Parse a --region argument into (seq_id, start, end).

    Accepts "SEQ_ID:START-END", and "START-END" without a sequence ID -- which is
    only meaningful when the run has a single input sequence, checked by the caller,
    which is the one place that knows what was resolved. Either coordinate may be
    omitted for an open interval: "SEQ:100-" or "SEQ:-500". A missing coordinate is
    returned as 0, meaning "the genome end on that side".

    Raises ValueError with a message suitable for showing to the user.
    """
    seq_id = None
    coords = text

    # rsplit: the coordinate half never contains a colon, so this survives a
    # sequence ID that does.
    if ":" in text:
        seq_id, coords = text.rsplit(":", 1)
        if not seq_id:
            raise ValueError(
                f"invalid region {text!r}: no sequence ID before the ':'."
            )

    parts = coords.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"invalid region {text!r}: expected two coordinates separated by a "
            "'-', as in 'REL606:3497890-3955678', '3497890-' or '-3955678'."
        )

    try:
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else 0
    except ValueError:
        raise ValueError(
            f"invalid region {text!r}: both coordinates must be whole numbers."
        )

    if start < 0 or end < 0:
        raise ValueError(f"invalid region {text!r}: coordinates cannot be negative.")
    if start and end and start >= end:
        raise ValueError(
            f"invalid region {text!r}: the start coordinate must come before the end."
        )

    return seq_id, start, end


def _detect_delimiter(path):
    """Tab or comma, decided by the header row rather than by the file name.

    bam2cov's CSV and TSV output differ only in this separator, so the format is a
    property of the bytes and not of the extension. Detecting it from the content means a
    file named directly on the command line works whatever it is called, and a
    comma-delimited table someone saved as ".tsv" still reads correctly.

    csv.Sniffer is deliberately not used: it guesses from a sample and misfires on
    single-column input, where it may pick a character out of the data itself.
    """
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            tabs = line.count("\t")
            commas = line.count(",")
            if tabs > commas:
                return "\t"
            if commas > tabs:
                return ","
            raise ValueError(
                f"Cannot tell whether {path} is tab- or comma-separated: its header row "
                f"has {tabs} tab(s) and {commas} comma(s). Expected a breseq "
                "'bam2cov --format TSV' or '--format CSV' coverage table."
            )

    raise ValueError(f"{path} has no header row -- it is empty, or entirely comments.")


def _read_coverage_table(path):
    """Read a breseq coverage table, dropping its commented summary block.

    The trailing '#' block is variable-length: this replaces a fixed skipfooter=4, which
    silently deleted four real data rows from any footerless input. The block is written
    with the same delimiter as the data, so it is stripped by its '#' prefix in CSV
    exactly as in TSV.
    """
    return pd.read_csv(
        path,
        sep=_detect_delimiter(path),
        header=0,
        index_col=0,
        comment="#",
    )


def normalize_coverage_columns(df, path=None):
    """Reduce either bam2cov coverage schema to the canonical `unique_cov`/`redundant` pair.

    Plain bam2cov output splits each count by strand; `--total-only` sums the strands
    itself and writes `unique_cov`, `redundant_cov` and `total_cov`. Those first two are
    exactly the sums CNery would compute anyway, so the narrower table costs nothing:
    `pct_redundant`, repeat censoring and the window median all behave identically.
    `total_cov` is ignored -- it is `unique_cov + redundant_cov` by construction.

    Idempotent, so it is safe to call both when a table is read and again in preprocess(),
    which lets preprocess() keep working on a raw frame handed to it directly.
    """
    where = f"{path}: " if path else ""

    missing_base = [c for c in BASE_COVERAGE_COLUMNS if c not in df.columns]
    have_total_only = all(c in df.columns for c in TOTAL_ONLY_COLUMNS)
    have_strand_split = all(c in df.columns for c in STRAND_SPLIT_COLUMNS)

    if missing_base or not (have_total_only or have_strand_split):
        raise ValueError(
            f"{where}not a usable coverage table. Missing "
            + ", ".join(missing_base + ([] if (have_total_only or have_strand_split)
                                        else ["coverage columns"]))
            + ". Expected a breseq 'bam2cov' table whose first column is the position, "
            "carrying "
            + ", ".join(BASE_COVERAGE_COLUMNS)
            + " plus either "
            + ", ".join(STRAND_SPLIT_COLUMNS)
            + " or, with --total-only, "
            + ", ".join(TOTAL_ONLY_COLUMNS)
            + ". Found: "
            + (", ".join(map(str, df.columns)) if len(df.columns) else "(no columns)")
            + "."
        )

    df = df.copy()

    if have_total_only:
        # Already strand-summed by breseq; only the name of the redundant column differs.
        df["redundant"] = df["redundant_cov"]
    else:
        df["unique_cov"] = df["unique_top_cov"] + df["unique_bot_cov"]
        df["redundant"] = df["redundant_top_cov"] + df["redundant_bot_cov"]

    return df


def read_coverage_table(path):
    """Read a coverage table and check it carries the columns the pipeline needs.

    Without this check a file with the wrong schema fails much later and much less
    clearly: _read_coverage_table() takes the first column as the position index whatever
    it holds, so a mismatched table parses "successfully" and then raises a bare KeyError
    from inside preprocess(). That mattered little when tables could only come from a
    directory CNery named itself, but any file can now be passed on the command line.
    """
    return normalize_coverage_columns(_read_coverage_table(path), path=path)


def preprocess(df, win=100, step=100, frag=400):

    if (step > win):
        return print(
            f'window size: {win} is smaller than step size: {step}. '
            f'Excluding segments of the genome for analysis.'
        )

    # Accepts either bam2cov coverage schema. Idempotent, so calling it again here is
    # harmless when the frame already came through read_coverage_table(), and it keeps
    # preprocess() usable on a raw frame handed to it directly.
    df_b2c = normalize_coverage_columns(df)

    # A table with a valid header and no position rows is a REFERENCE WITH NO
    # COVERAGE, not a malformed file -- bam2cov writes one for a sequence that
    # got no reads, and read_coverage_table() passes it because every column it
    # checks for is present. Windowing it yields no windows, which is the honest
    # answer; taking df.index[0] to find the first coordinate is what used to
    # raise a bare IndexError from four frames below the input boundary.
    # get_CNV.main() turns the empty frame into a "no usable coverage" result and
    # still writes breseq its JSON. See no_usable_coverage_reason().
    start_coord = int(df_b2c.index[0]) if len(df_b2c.index) else 1
    genome = df_b2c['ref_base']
    genome_len = len(genome)
    # GC% is measured over max(frag, win) bases centred on each window, because
    # GC bias acts at the scale of the sequenced fragment rather than at
    # whatever window size was asked for. `gc_pad` is what that costs on each
    # side of the window.
    gc_pad = max((max(int(frag), win) - win) // 2, 0)

    # The reference is circular, so windows near either end draw their padding
    # from the other end. Carry exactly `gc_pad` bases of wrap-around either
    # side -- no more, and never less: this used to be a fixed +/-25% of the
    # genome, which is both larger than needed on a chromosome and too small
    # whenever the fragment exceeds half the reference, where it silently
    # produced an out-of-range slice.
    if genome_len:
        gc_start = (-gc_pad) % genome_len
        genome_cyc = list(
            islice(cycle(genome), gc_start, gc_start + genome_len + 2 * gc_pad)
        )
    else:
        genome_cyc = []

    # Prefix sums of the G and C counts along the reference, so each window's
    # base composition is two array lookups instead of another pass of the
    # per-window Python string work gc_percent already pays for.
    bases = genome.to_numpy()
    cum_g = np.concatenate(([0], np.cumsum(bases == 'G')))
    cum_c = np.concatenate(([0], np.cumsum(bases == 'C')))

    fragseq = []
    fragment = []
    winseq = []
    seq = []
    gcp_s = []
    gc_skew_s = []
    window = []
    win_end = []
    window_med_cov = []
    pct_redundant_s = []

    df_b2c["cov_type"] = df_b2c["redundant"].apply(lambda x: 'R' if x > 0 else 'U')
    df_gc = pd.DataFrame(columns=["window_num"])

    i = 0
    lst_win = 0

    # sliding window = win and increment size = step
    # summarizes GC% and median coverage
    #
    # Every FULL-WIDTH window is kept, using TOTAL coverage
    # (unique + redundant) for its median -- so a repeat's real sequencing
    # depth is reflected instead of just the reads that happened to map
    # uniquely there -- and the fraction of redundant-covered bases in the
    # window is recorded as `pct_redundant`. Downstream,
    # mask_coverage_windows() turns `pct_redundant` into `is_redundant`,
    # which censors the window from GC/OTR bias-model FITTING, from the
    # origin/terminus peak-trough SEARCH (see otr_fit()) and from the
    # Viterbi observation sequence (see run_HMM()), while it still
    # receives a real bias-corrected coverage value and inherits the copy
    # number of the segment it sits in -- so a repeat neither invents an
    # amplification nor breaks a genuine deletion in two.
    while (i <= (genome_len - 1)) and (lst_win < genome_len):

        win_full_cov = df_b2c["unique_cov"].iloc[i:(i + win)].to_numpy()
        win_redundant_cov = df_b2c["redundant"].iloc[i:(i + win)].to_numpy()
        cov_type = df_b2c["cov_type"].iloc[i:(i + win)].to_numpy()

        winu = len(cov_type)  # bases actually available

        # Only windows backed by the full `win` bases are emitted. A
        # trailing partial window at the genome end would take its median
        # over fewer bases, and its short win_end would set the
        # genome-end coordinate used for the final CN segment and for the
        # terminus fallback in apply_otr_correction() -- so it is dropped.
        if winu < win:
            # ...unless no full window fits at all, i.e. the reference is
            # shorter than `win` (a small plasmid or contig). Then this
            # partial window is all there is, and returning an empty
            # frame would be worse than returning a short one.
            if window or winu == 0:
                break

        # Total coverage per base = unique + redundant, so a window
        # spanning a repeat is not under-counted just because some of its
        # reads mapped ambiguously.
        win_cov = (win_full_cov + win_redundant_cov).astype(float)

        # Fraction of bases in this window overlapping redundant coverage.
        pct_redundant = float(np.mean(cov_type == 'R'))

        # Summarize the window coverage statistics
        window_med_cov.insert(i, float(np.nanmedian(win_cov)))
        pct_redundant_s.insert(i, pct_redundant)
        winseq = genome[i:i + winu]
        seq.insert(i, ''.join(str(element) for element in winseq))

        # GC skew (G-C)/(G+C) over the WINDOW, not over the fragment gc_percent
        # uses. gc_percent is fragment-widened because it models a property of
        # the sequencing chemistry; skew is a property of the genome's
        # replication strand asymmetry, so it has to line up with win_st/win_end.
        # A window holding neither G nor C carries no strand information at all,
        # so it contributes 0 rather than dividing by zero. Non-ACGT characters
        # count as neither, matching the exact-uppercase membership test gc_percent
        # uses below -- a soft-masked (lowercase) reference is ignored by both.
        g_count = int(cum_g[i + winu] - cum_g[i])
        c_count = int(cum_c[i + winu] - cum_c[i])
        gc_total = g_count + c_count
        gc_skew_s.insert(i, ((g_count - c_count) / gc_total) if gc_total else 0.0)

        window.insert(i, i)
        win_end.insert(i, i + winu)
        lst_win = win_end[(len(win_end) - 1)]
        # genome_cyc[gc_pad] is genome[0], so this is where window i starts in it.
        i_off = i + gc_pad

        # One span, always max(frag, win). This was two branches: `frag > win`
        # spanned frag, correctly, but the `frag <= win` branch spanned
        # `2 * win - frag` where its own comment said it used the window length
        # -- so at -w 200 -f 150 it measured GC over 250 bases, neither the
        # window nor the fragment.
        fragseq = genome_cyc[(i_off - gc_pad):((i_off + win) + gc_pad)]
        fragment.insert(i, ''.join(str(element) for element in fragseq))
        gcc = ''.join([nucleotide for nucleotide in fragseq
                       if nucleotide in ['C', 'G']])
        gccp = (len(gcc) / len(fragseq))
        gcp_s.insert(i, gccp)

        i = i + step

    # Save the window median and GC% per fragment overlapping a window
    # to the dataframe
    df_gc["win_st"] = [x + start_coord for x in window]
    df_gc["win_end"] = [x + start_coord for x in win_end]
    df_gc["win_len"] = df_gc["win_end"] - df_gc["win_st"]
    df_gc["gc_percent"] = gcp_s
    df_gc["read_count_cov"] = window_med_cov
    df_gc["pct_redundant"] = pct_redundant_s
    df_gc["window_num"] = np.arange(0, len(window_med_cov), 1)
    df_gc["norm_raw_cov"] = (
        df_gc["read_count_cov"] / df_gc["read_count_cov"].median()
    )

    # Cumulative GC skew (Grigoriev 1998, NAR 26:2286): the running sum of the
    # per-window skew bottoms out over the replication origin and peaks at the
    # terminus. See predict_ori_ter_from_skew().
    #
    # The MEAN IS SUBTRACTED FIRST, and that is load-bearing rather than
    # cosmetic. It forces the running sum to return to exactly zero at the last
    # window, which is what makes argmin/argmax independent of where the
    # reference's coordinate 1 happens to fall: rotating the start by r windows
    # then maps the curve to C'(k) = C((r+k) mod n) - C(r), a constant offset,
    # so both extrema stay on the same genomic locus. Without it the sum ends at
    # n * mean(skew) != 0, the wraparound adds a linear ramp, and a circularly
    # permuted reference of the SAME genome predicts a different origin.
    #
    # Overlapping windows (step < win) count each base win/step times over, but
    # that is a single uniform factor on every window, so it rescales the curve
    # without moving either extremum. No stride weighting is needed.
    skew = np.asarray(gc_skew_s, dtype=float)
    df_gc["gc_skew"] = skew
    df_gc["cum_gc_skew"] = np.cumsum(skew - skew.mean()) if skew.size else skew

    # An empty frame is built from empty LISTS, which pandas types as `object`.
    # Downstream every one of these columns is read with .to_numpy(dtype=float)
    # and several are pd.concat'd with populated frames, so leaving them object
    # would turn "no coverage" into a dtype error somewhere else entirely.
    if df_gc.empty:
        for col in ("win_st", "win_end", "win_len", "window_num"):
            df_gc[col] = df_gc[col].astype(int)
        for col in ("gc_percent", "read_count_cov", "pct_redundant",
                    "norm_raw_cov", "gc_skew", "cum_gc_skew"):
            df_gc[col] = df_gc[col].astype(float)

    return df_gc


def plottable(df):
    """Windows a plot may draw: everything except redundant/repeat coverage.

    A window with redundant coverage carries no information about this locus's
    copy number. CWBI's chromosome has 17 kb at 2.13 Mb with `pct_redundant`
    1.000 where UNIQUE depth is exactly 0 and redundant depth is 18x the
    single-copy level -- the reference holds ~18 near-identical copies and every
    read is placed at all of them. Drawing that at 18x invites the reader to see
    an amplification, sets the y-axis so nothing else is legible, and represents
    a quantity the pipeline never used: no fit sees these windows, and the HMM
    neither observes them nor fits its emission model to them.

    Deletions are NOT excluded. They are censored from fitting too, but they are
    real measurements of a real absence and the CN=0 calls are unreadable
    without them.

    Returns a boolean array, all-True when the column is absent.
    """
    if "is_redundant" in df:
        return ~df["is_redundant"].to_numpy(dtype=bool)
    return np.ones(len(df), dtype=bool)


#: Fragment-size scan. GC bias acts at the scale of the sequenced fragment, so
#: `frag` is a property of the LIBRARY -- something a user may simply not know.
#: These bound a grid of candidates to score against the data.
#:
#: The upper bound is a modelling statement, not a convenience. Above ~1 kb an
#: Illumina fragment is rare, and "GC over 2 kb" stops being a fragment property
#: and becomes a long-range positional average that proxies the replication ramp.
#: Measured: scanning to 2000 against RAW coverage picks 2000 -- the edge -- on
#: two of eight sequences, for exactly that reason.
#: The fragment size used when the user pins none and the scan declines.
DEFAULT_FRAG_SIZE = 400
FRAG_SCAN_MAX = 2500
#: Candidate fragment sizes, filtered by the caller to those the window size
#: leaves distinguishable.
#:
#: Roughly geometric -- step ratios run 1.20 to 1.50 -- because fragment-scale
#: effects are multiplicative: 150 -> 200 changes the averaging window by 33%,
#: 800 -> 850 by 6%, so linear spacing would over-resolve the top and
#: under-resolve the bottom. DELIBERATELY COARSEST AT THE LOW END, where the
#: measured optima are shallow and the argmin jitters, so finer spacing would
#: offer resolution the data cannot support. Round numbers a user can match
#: against a library prep, not artefacts of a float multiply.
FRAG_SCAN_GRID = (100, 150, 200, 250, 300, 400, 500, 600, 800,
                  1000, 1200, 1500, 2000, 2500)
FRAG_SCAN_MIN_CANDIDATES = 4
#: Independent CV splits. The argmin genuinely jitters where the optimum is
#: shallow (measured: four distinct answers across eight splits on
#: ltee_ara_m3_32k_2rg), so the selection is a median over repeats rather than
#: one split's argmin.
FRAG_SCAN_REPEATS = 5
FRAG_SCAN_FOLDS = 5
#: Windows the cross-validation scores. The scan is pooled across every sequence
#: in the run, so this caps a cost that would otherwise grow with the genome.
FRAG_SCAN_MAX_WINDOWS = 12000


def frag_candidates(win, lo=None, hi=FRAG_SCAN_MAX, grid=FRAG_SCAN_GRID):
    """Fragment sizes worth scoring, given the window size.

    THE LOWER BOUND IS `win`, AND THAT IS NOT ARBITRARY. preprocess() measures GC
    over `max(frag, win)` bases, so every candidate at or below the window size
    produces BIT-IDENTICAL gc_percent -- they are not candidates, they are
    duplicates of each other. At -w 1000 a 150 bp "fragment" and a 400 bp one are
    the same computation, which is also why a large -w collapses this grid and
    the caller then declines to scan.
    """
    lo = int(win if lo is None else lo)
    return [int(f) for f in grid if lo <= f <= int(hi)]


def reference_gc_flags(df):
    """Per-base G-or-C flags for a coverage table, for the fragment scan.

    Kept as a bool array rather than the prefix sum: 1 byte per base against 8,
    which is 4.6 MB instead of 37 MB on REL606, and the cumsum is cheap when a
    candidate actually needs it.
    """
    bases = normalize_coverage_columns(df)["ref_base"].to_numpy().astype(str)
    return (bases == "G") | (bases == "C")


def gc_percent_for_frag(gc_flags, win_st0, win, frag):
    """gc_percent for every window at this fragment size.

    Mirrors preprocess() exactly -- max(frag, win) bases CENTRED on the window,
    drawn circularly -- but from prefix sums, so rescanning a dozen candidates
    costs a dozen array passes instead of a dozen preprocess() runs.

    `win_st0` is each window's start as a 0-based offset into the reference,
    i.e. `df["win_st"] - start_coord`.
    """
    flags = np.asarray(gc_flags, dtype=bool)
    n = flags.size
    if n == 0:
        return np.zeros(np.asarray(win_st0).size, dtype=float)

    pad = max((max(int(frag), int(win)) - int(win)) // 2, 0)
    cum = np.concatenate(([0], np.cumsum(flags)))
    total = int(cum[-1])

    lo = np.asarray(win_st0, dtype=np.int64) - pad
    hi = np.asarray(win_st0, dtype=np.int64) + int(win) + pad     # half-open
    span = int(win) + 2 * pad

    def prefix(idx):
        whole, rem = np.divmod(idx, n)
        return whole * total + cum[rem]

    return (prefix(hi) - prefix(lo)) / float(span)


def _frag_cv_errors(gc, cov, folds=FRAG_SCAN_FOLDS, seed=0,
                    cap=FRAG_SCAN_MAX_WINDOWS):
    """Per-fold held-out MSE of coverage predicted from GC.

    Out-of-sample on purpose. The in-sample alternative -- how flat the corrected
    coverage looks against GC -- is CIRCULAR: the correction is a LOWESS of
    coverage on gc_percent(frag), so it flattens that axis by construction
    whatever frag it was given. Measured, every candidate looks best on its own
    axis (adp1 reads 0.63% residual trend at its own 400 and 4.98% at 150; at
    frag 150 it reads 0.62% and 1.88% the other way round). Held-out prediction
    error is the only comparison that is not rigged.
    """
    rng = np.random.default_rng(seed)
    gc = np.asarray(gc, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = gc.size
    if n > cap:
        idx = rng.choice(n, cap, replace=False)
        gc, cov, n = gc[idx], cov[idx], cap
    if n < 10 * folds:
        return None

    fold = rng.integers(0, folds, n)
    span = float(gc.max() - gc.min())
    loess = sm.nonparametric.lowess
    out = []
    for j in range(folds):
        train, test = fold != j, fold == j
        if train.sum() < 20 or test.sum() < 5:
            return None
        fit = loess(cov[train], gc[train], frac=0.3, it=1,
                    delta=0.0005 * span if span > 0 else 0.0,
                    is_sorted=False, missing="none", return_sorted=True)
        pred = np.interp(gc[test], fit[:, 0], fit[:, 1])
        out.append(float(np.mean((cov[test] - pred) ** 2)))
    return out


def frag_scan_target(df):
    """The series the fragment scan is scored against, for one sequence.

    Coverage with the replication ramp divided out, on windows the first pass
    called CN = 1. BOTH exclusions are load-bearing, and the scan gives the wrong
    answer without them:

      - COPY NUMBER. cwbi_ssym_ht04's chromosome carries a CN-34 amplification
        whose variance swamps every GC effect -- its held-out MSE is 2.64 against
        0.01-0.2 elsewhere, and its apparent optimum is 0.04% deep, i.e. noise.
      - POSITION. At large frag, "GC" becomes a long-range average that tracks
        genomic coordinate, so it can predict coverage by proxying the
        replication ramp rather than by modelling fragment chemistry. Scanning
        raw coverage picks the top of the range on two of eight sequences.

    Measured, controlling for both moves every optimum to the interior of the
    grid, and makes the three sequences of one sample -- which share a library
    and so must share a fragment size -- agree where they had not.

    Returns (target, keep) with `keep` the windows worth scoring.
    """
    raw = df["norm_raw_cov"].to_numpy(dtype=float)
    tent = np.ones(len(df), dtype=float)
    if "otr_gc_corr_fact" in df.columns:
        tent = df["otr_gc_corr_fact"].to_numpy(dtype=float)
        finite = np.isfinite(tent) & (tent > 0)
        tent = tent / (np.median(tent[finite]) if finite.any() else 1.0)
        tent = np.where(finite, tent, 1.0)
    target = raw / tent

    keep = np.isfinite(target) & (target > 0)
    for column in ("exclude_from_fit", "is_cn_variant"):
        if column in df.columns:
            keep &= ~df[column].to_numpy(dtype=bool)

    # NORMALISED TO THIS SEQUENCE'S OWN LEVEL, because the scan pools every
    # sequence in the run. norm_raw_cov is normalised against the POOLED median,
    # so CWBI's plasmids sit at 2.95x and 1.90x against a chromosome at 1.0 --
    # and at a large fragment size a short plasmid's GC is nearly constant, which
    # turns GC into a REPLICON LABEL that predicts those level differences
    # perfectly. Measured, that alone made the scan pick the top of the grid.
    if keep.any():
        level = float(np.median(target[keep]))
        if np.isfinite(level) and level > 0:
            target = target / level
    return target, keep


def select_frag_size(per_genome, gc_flags, win, default_frag,
                     repeats=FRAG_SCAN_REPEATS, seed=0):
    """Choose the library's fragment size from the data. Returns (frag, detail).

    POOLED, like both GC fits: GC bias belongs to the sequencing chemistry, so
    one fragment size describes the run rather than a different one reaching each
    reference.

    THE SELECTION IS A MEDIAN OVER INDEPENDENT CV SPLITS, not one split's argmin.
    Where the optimum is shallow the argmin genuinely jitters -- measured, four
    distinct answers across eight splits on ltee_ara_m3_32k_2rg, whose optimum is
    flat to 0.7% between 100 and 400. A single split would report that jitter as
    an answer.

    THE DEFAULT IS KEPT UNLESS THE IMPROVEMENT BEATS ITS OWN UNCERTAINTY. The
    margin is the standard error of the paired per-fold differences, so there is
    no invented threshold: a candidate has to win by more than the noise in the
    measurement that chose it.
    """
    detail = {"Fragment size scanned": False, "Fragment size": int(default_frag)}
    candidates = frag_candidates(win)
    if len(candidates) < FRAG_SCAN_MIN_CANDIDATES:
        detail["Fragment size reason"] = (
            f"window size {int(win)} leaves only {len(candidates)} distinct "
            "candidate(s); GC is measured over max(frag, win)")
        return int(default_frag), detail

    columns, targets = [], []
    for genome_id, df in per_genome.items():
        flags = gc_flags.get(genome_id)
        if flags is None or "win_st" not in df:
            continue
        target, keep = frag_scan_target(df)
        if not keep.any():
            continue
        win_st0 = df["win_st"].to_numpy(dtype=np.int64) - int(df["win_st"].min())
        # win_st is 1-based from the table's own first coordinate; the offset of
        # the first window is what makes it 0-based into the reference.
        columns.append((flags, win_st0[keep], keep))
        targets.append(target[keep])
    if not targets:
        detail["Fragment size reason"] = "no windows survived censoring"
        return int(default_frag), detail

    target = np.concatenate(targets)
    gc_by_frag = {f: np.concatenate([gc_percent_for_frag(flags, w, win, f)
                                     for flags, w, _k in columns])
                  for f in candidates}

    # SELECT ON ONE HALF OF THE WINDOWS, JUDGE ON THE OTHER.
    #
    # Testing the winner against the default on the data that chose it is the
    # classic selection-bias error: the winner's margin is large partly BECAUSE
    # it was picked for being large, so with 14 candidates a pure-noise series
    # produces a "significant" improvement. Measured, exactly that -- coverage
    # with no GC signal at all selected 500 over the default and passed the
    # standard-error test. Splitting makes the margin an honest out-of-sample
    # quantity.
    rng = np.random.default_rng(seed)
    half = rng.random(target.size) < 0.5
    if half.sum() < 100 or (~half).sum() < 100:
        detail["Fragment size reason"] = "too few windows to select and judge apart"
        return int(default_frag), detail

    picks = []
    for r in range(repeats):
        means = {}
        for f in candidates:
            got = _frag_cv_errors(gc_by_frag[f][half], target[half], seed=seed + r)
            if got is None:
                detail["Fragment size reason"] = "too few windows to cross-validate"
                return int(default_frag), detail
            means[f] = float(np.mean(got))
        picks.append(min(means, key=means.get))
    chosen = int(np.median(picks))
    chosen = min(candidates, key=lambda f: (abs(f - chosen), f))

    # Nearest grid point to the default, so "did it beat the default" is asked of
    # something the grid can actually express.
    baseline = min(candidates, key=lambda f: (abs(f - int(default_frag)), f))

    detail.update({
        "Fragment size scanned": True,
        "Fragment size candidates": candidates,
        "Fragment size selected": chosen,
        "Fragment size default": int(default_frag),
        "Fragment size picks per split": picks,
    })
    if chosen == baseline:
        detail["Fragment size reason"] = "the default is the best candidate"
        return int(default_frag), detail

    judged = ~half
    diffs = []
    for r in range(repeats):
        a = _frag_cv_errors(gc_by_frag[baseline][judged], target[judged], seed=seed + r)
        b = _frag_cv_errors(gc_by_frag[chosen][judged], target[judged], seed=seed + r)
        if a is None or b is None:
            detail["Fragment size reason"] = "too few windows to judge the choice"
            return int(default_frag), detail
        diffs.extend(np.asarray(a) - np.asarray(b))       # >0 means chosen wins
    diffs = np.asarray(diffs, dtype=float)
    margin = float(diffs.mean())
    stderr = float(diffs.std(ddof=1) / np.sqrt(diffs.size)) if diffs.size > 1 else 0.0
    detail["Fragment size improvement"] = _round(margin, 6)
    detail["Fragment size improvement se"] = _round(stderr, 6)

    if margin > stderr and margin > 0:
        detail["Fragment size"] = chosen
        detail["Fragment size reason"] = (
            "selected: beats the default on held-out windows by more than its "
            "own standard error")
        return chosen, detail

    detail["Fragment size"] = int(default_frag)
    detail["Fragment size reason"] = (
        "default kept: the best candidate did not beat it on held-out windows by "
        "more than the noise in the measurement")
    return int(default_frag), detail


def _round(value, nd):
    try:
        return round(float(value), nd)
    except (TypeError, ValueError):
        return None


def gc_cor_plots(df, output):
    genome_ids = sorted(df["genome_id"].unique())
    # Every table in the run was empty, so the pooled frame has no rows and no
    # genome to name the file after. There is no diagnostic to draw either.
    if not genome_ids:
        return
    if len(genome_ids) > 1:
        label = "_and_".join(str(g) for g in genome_ids)
    else:
        label = str(genome_ids[0])
    samplename = f"{label}_GC_vs_NormRds"
    saveplt = str(output + "/GC_bias/")

    os.makedirs(saveplt, exist_ok=True)

    plt.figure(figsize=(10, 8))

    # The LOWESS below was fitted without repeat windows, so drawing them beside
    # it would show the curve missing points it was never asked to pass through.
    df = df.loc[plottable(df)]

    uniq = (
        df[['gc_percent', 'gc_corr_fact']]
        .drop_duplicates(subset='gc_percent')
        .sort_values('gc_percent')
    )
    # A quadratic needs three points. Below that -- a one-position reference, or
    # a frame where plottable() dropped everything because every window is a
    # repeat or a deletion -- np.polyfit raises LinAlgError ("SVD did not
    # converge") rather than returning anything. Draw the scatter without the
    # summary curve; the points are the diagnostic, the parabola is a reading aid.
    gc_fit = None
    if len(uniq) >= 3:
        gc_fit = np.poly1d(
            np.polyfit(uniq['gc_percent'], uniq['gc_corr_fact'], 2)
        )

    plt.scatter(
        df['gc_percent'],
        df['norm_raw_cov'],
        color='brown',
        label='Raw normalized reads vs GC',
        s=5
    )
    plt.scatter(
        df['gc_percent'],
        df['gc_corr_norm_cov'],
        color="green",
        label='Corrected normalized reads',
        s=10,
        alpha=0.3
    )
    if gc_fit is not None:
        plt.plot(
            np.sort(df['gc_percent'].unique()),
            gc_fit(np.sort(df['gc_percent'].unique())),
            color='black',
            linewidth=3,
            label='LOWESS fit'
        )

    plt.ylabel('Normalized read coverage')
    plt.xlabel('GC% per window')
    plt.title(f'{samplename}_GCvsNormalizedReads')
    plt.legend(loc='upper right')

    plt_full_path = os.path.join(
        saveplt,
        '%s_GC_vs_NormRds.pdf' % samplename.replace(' ', '_')
    )
    plt.savefig(plt_full_path, format='pdf', bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# mask -> fit -> apply split for GC-bias correction. Same LOWESS math
# throughout, reorganized into three composable stages:
#
#   - Windows are censored (excluded from FITTING) for two DISTINCT reasons:
#       "zero_outlier"  : genuinely near-zero coverage (real deletions)
#       "redundant"     : pct_redundant (from preprocess()) above threshold
#   - Only "zero_outlier" windows are frozen to zero in the corrected
#     output. "redundant" windows are excluded from the LOWESS/OTR fits
#     (so repeats can't bias the curve) but still get a real,
#     model-corrected coverage value -- so the HMM does not call spurious
#     deletions over repeats.
# ---------------------------------------------------------------------------
def mask_coverage_windows(
    df,
    zero_frac=0.1,
    redundant_frac_thresh=0.0,
    censor_col="exclude_from_fit",
    deletion_col="is_deletion",
):
    """
    Flag windows that should be excluded from bias-model FITTING, while
    keeping the two exclusion reasons separate so they can be treated
    differently downstream:

      - `is_deletion` (True/False): near-zero/outlier coverage windows,
        i.e. genuine deletions. These ARE frozen to zero by
        apply_gc_correction()/apply_otr_correction().
      - `is_redundant` (True/False): windows whose `pct_redundant`
        (computed by preprocess(), the fraction of bases in the window
        overlapping redundant/repeat coverage) exceeds
        `redundant_frac_thresh`. These are excluded from fitting but are
        NOT frozen to zero -- they still receive a real bias-corrected
        value so the HMM doesn't call spurious deletions over repeats.
      - `censor_col` (default "exclude_from_fit"): is_deletion OR
        is_redundant -- the single flag fit_gc_bias()/fit_otr_bias() use
        to decide which windows inform the fit.

    Does NOT drop rows -- preserves full window-index continuity for
    downstream HMM / plotting code.
    """
    df = df.copy()

    med = df["read_count_cov"].median()
    zero_mask = (df["read_count_cov"] <= (med * zero_frac)).to_numpy()

    cov = df["norm_raw_cov"].to_numpy(dtype=float)
    gc = df["gc_percent"].to_numpy(dtype=float)
    finite_mask = ~(np.isfinite(cov) & np.isfinite(gc))

    is_deletion = zero_mask | finite_mask

    if "pct_redundant" in df.columns:
        is_redundant = (df["pct_redundant"].to_numpy(dtype=float) > redundant_frac_thresh)
    else:
        is_redundant = np.zeros(len(df), dtype=bool)

    df[deletion_col] = is_deletion
    df["is_redundant"] = is_redundant
    df[censor_col] = is_deletion | is_redundant

    df["censor_reason"] = np.select(
        [is_deletion, is_redundant],
        ["zero_outlier", "redundant"],
        default="clean",
    )
    return df


def no_usable_coverage_reason(df):
    """Why this sequence cannot be analysed at all, or None if it can.

    A reference can reach the pipeline with nothing to measure -- a plasmid that
    got no reads, or a coverage table with a valid header and no position rows.
    Every fit downstream then has an empty design matrix, and what used to happen
    was a bare IndexError from preprocess(), a ValueError from np.interp inside
    fit_gc_bias(), or a ZeroDivisionError from solve_pr() several stages later.
    None of those name the sequence, and all of them take the whole run down --
    including the healthy references sharing the invocation.

    So it is classified ONCE, here, and reported as a result: get_CNV.main()
    writes the sequence its full (degenerate) output set, breseq included, says
    so on stdout, and carries on with the rest.

    WHAT IS DELIBERATELY NOT HERE: "every window is censored as redundant". An
    all-repeat replicon has real coverage and deserves real copy-number calls,
    and otr_fit() and run_HMM() already cope by widening the mask rather than
    failing. The only stage that could not survive it was fit_gc_bias(), which
    now declines to an identity curve -- so that case stays a normal analysis
    with one fit skipped, not a degenerate sequence.
    """
    if len(df) == 0:
        return "the coverage table has no position rows"

    cov = df["read_count_cov"].to_numpy(dtype=float)
    if not np.any(np.isfinite(cov) & (cov > 0)):
        return "every window has zero coverage"

    return None


def otr_ratio(otr_fit_result):
    """The applied origin-to-terminus ratio, or None when no tent was applied.

    Same yori/yter apply_otr_correction() reports, from the same two anchors, so
    the pass-1 figure in the results JSON cannot drift from what pass 1 actually
    wrote.
    """
    if not otr_fit_result or not otr_fit_result.get("bias"):
        return None
    y_fit = otr_fit_result["y_fit"]
    y_ter = float(y_fit[otr_fit_result["t_idx"]])
    return float(y_fit[otr_fit_result["o_idx"]] / y_ter) if y_ter > 0 else None


def pass1_summary(otr_fit_result, df_staged):
    """The first pass's verdict, for the second pass's results JSON.

    A verdict that CHANGED under the CN censor is the thing a reader needs to be
    able to see from the file alone -- and on the corpus the evidence moves a lot
    even where the verdict does not (adp1's coverage fit goes p = 0.32 -> 0.001
    once its CN-3 amplification is out of the fit).

    Shared by get_CNV.main() and tests/test_authentic.py::_run_pipeline so the
    harness cannot report something main() does not.
    """
    detail = (otr_fit_result or {}).get("detail") or {}
    censored = (int(df_staged["is_cn_variant"].sum())
                if "is_cn_variant" in df_staged else 0)
    return {
        "Coverage fit r-squared (pass 1)": detail.get("Coverage fit r-squared"),
        "Coverage fit p-value (pass 1)": detail.get("Coverage fit p-value"),
        "Breakpoint source (pass 1)": detail.get("Breakpoint source"),
        "Origin-to-Terminus/Bias Ratio (pass 1)": otr_ratio(otr_fit_result),
        "Windows censored as CN != 1": censored,
        "Refit on CN=1 windows": bool(censored),
    }


#: Floor on the fraction of a sequence's windows that must survive the CN censor.
#: The second pass excludes everything the first pass's HMM did not call CN = 1,
#: and on a genuinely duplicated replicon -- every window CN = 2 -- that is the
#: whole sequence. Falling back to the first pass's censoring there is the honest
#: answer: it is a sequence whose copy number is not 1, not a sequence with no
#: usable windows.
CN_CENSOR_MIN_KEEP = 0.5


def add_cn_censor(df, cn_col="prob_copy_number", min_keep=CN_CENSOR_MIN_KEEP):
    """Flag windows a previous pass's HMM did not call CN = 1, for the next pass.

    Writes `is_cn_variant` and folds it into `exclude_from_fit`, which is the
    single flag every fit stage already consults -- fit_gc_bias() reads it by
    default and otr_fit() ORs `is_cn_variant` in alongside its two siblings. So
    both refits pick the censor up without either of them knowing which pass it
    is in.

    This is what the two-pass structure exists for. `mask_coverage_windows`
    censors on two crude proxies computed once on UNCORRECTED coverage -- near-zero
    depth and repeat overlap -- and an amplification is caught by neither, so it
    enters the GC LOWESS and the OTR tent at full weight. Measured, that matters:
    cwbi_ssym_ht04's chromosome fits its origin inside its own CN-34
    amplification, and adp1_mgd06_lb inside its CN-3 one.

    Returns (df, applied). `applied` is False when the censor would leave under
    `min_keep` of the windows, in which case nothing is added and the second pass
    censors exactly as the first did.
    """
    df = df.copy()
    n = len(df)
    if cn_col not in df.columns or n == 0:
        df["is_cn_variant"] = np.zeros(n, dtype=bool)
        return df, False

    cn = df[cn_col].to_numpy(dtype=float)
    is_variant = np.isfinite(cn) & (np.rint(cn) != 1)

    base = np.zeros(n, dtype=bool)
    for column in ("is_deletion", "is_redundant"):
        if column in df.columns:
            base |= df[column].to_numpy(dtype=bool)

    keep = (~(base | is_variant)).mean()
    if keep < min_keep:
        df["is_cn_variant"] = np.zeros(n, dtype=bool)
        return df, False

    df["is_cn_variant"] = is_variant
    if "exclude_from_fit" in df.columns:
        df["exclude_from_fit"] = df["exclude_from_fit"].to_numpy(dtype=bool) | is_variant
    else:
        df["exclude_from_fit"] = base | is_variant
    return df, True


def stage_pass1(df, min_keep=CN_CENSOR_MIN_KEEP):
    """Snapshot the first pass's results and build the censor for the second.

    Every column the first pass produced is copied to a `*_pass1` name before the
    second pass overwrites the canonical ones, so the correction-stages figure can
    draw both passes and a reader can tell what the CN censor bought. Then
    add_cn_censor() turns the first pass's calls into `is_cn_variant`.

    Lives in core.py rather than in get_CNV.main() because
    tests/test_authentic.py::_run_pipeline has to mirror main() exactly, and that
    harness silently diverging from it has cost real debugging time twice.

    Returns (df, cn_censor_applied).
    """
    df = df.copy()
    for column in ("gc_corr_fact", "gc_corr_norm_cov",
                   "otr_gc_corr_fact", "otr_gc_corr_norm_cov",
                   "prob_copy_number"):
        # otr_gc_corr_fact is absent under --bias gc/none, where
        # apply_otr_correction() never ran.
        if column in df.columns:
            df[f"{column}_pass1"] = df[column].to_numpy()
    return add_cn_censor(df, min_keep=min_keep)


#: Bootstrap replicates for the GC curve's pointwise uncertainty. This is a
#: standard deviation, not a p-value, so it needs far fewer replicates than the
#: detection gates' 1000 -- the sd of an sd from 100 draws is ~7% of itself,
#: which is well inside what the emission variance cares about.
GC_TAU_SURROGATES = 100
#: GC grid the bootstrap curves are evaluated on. LOWESS is smooth by
#: construction, so 200 points resolve it and keep the cost independent of how
#: many windows the sequence has.
GC_TAU_GRID = 200


def _gc_curve_tau(gc, cov, fit_mask, grid, n_surrogates=GC_TAU_SURROGATES, seed=0):
    """Pointwise relative sd of the fitted GC curve, by resampling windows.

    WHY THIS AND NOT A PERCENTILE. The obvious proxy -- "few windows out here" --
    is wrong for this smoother: LOWESS uses a NEAREST-NEIGHBOUR bandwidth
    (frac=0.3), so every fitted point averages the same ~0.3n windows. At GC 0.62
    on ltee_ara_m3_32k_2rg there are still ~12,700 in the neighbourhood. What
    actually degrades at the tails is that the neighbourhood becomes ONE-SIDED,
    so the local linear fit extrapolates within its own window -- the standard
    boundary effect, which inflates the fitted value's variance while leaving the
    point estimate perfectly smooth. No count-based or percentile-based rule sees
    that; resampling does, and it puts no constant in the code for anyone to tune.

    Returns sd of log(curve) on `grid`, i.e. a RELATIVE sd, which is the form the
    emission model needs: the offset enters multiplicatively.
    """
    rng = np.random.default_rng(seed)
    gc_f, cov_f = gc[fit_mask], cov[fit_mask]
    n = gc_f.size
    if n < 50:
        return np.zeros_like(grid)

    loess = sm.nonparametric.lowess
    # delta accelerates the replicates -- exact fits at points more than delta
    # apart, linear interpolation between. The POINT ESTIMATE keeps delta=0.0
    # above; this approximation is only ever used for the spread.
    span = float(gc_f.max() - gc_f.min())
    delta = 0.001 * span if span > 0 else 0.0

    curves = np.empty((n_surrogates, grid.size), dtype=float)
    for b in range(n_surrogates):
        idx = rng.integers(0, n, n)
        out = loess(cov_f[idx], gc_f[idx], frac=0.3, it=1, delta=delta,
                    is_sorted=False, missing="none", return_sorted=True)
        curves[b] = np.interp(grid, out[:, 0], out[:, 1])

    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.where(curves > 0, curves, np.nan))
    tau = np.nanstd(logs, axis=0)
    return np.where(np.isfinite(tau), tau, 0.0)


def fit_gc_bias(
    df,
    censor_col="exclude_from_fit",
    n_robust_iter=3,
    resid_mad=5.0,
    fit_floor_frac=0.05,
    tau_surrogates=GC_TAU_SURROGATES,
    tau_seed=0,
):
    """
    Fit stage of GC-bias correction: iterative robust LOWESS of
    norm_raw_cov vs gc_percent, using only windows where
    df[censor_col] is False (i.e. neither deletions nor redundant-coverage
    windows).

    Returns
    -------
    dict : {"gc_sorted", "fit_sorted", "floor", "gc_grid", "gc_tau"} -- pass to
    apply_gc_correction(). `gc_tau` is the pointwise RELATIVE standard deviation
    of the fitted curve on `gc_grid`, from resampling the fit windows; it is what
    lets run_HMM discount windows whose correction factor is poorly determined.
    Set `tau_surrogates=0` to skip it.
    """
    cov = df["norm_raw_cov"].to_numpy(dtype=float)
    gc = df["gc_percent"].to_numpy(dtype=float)
    censored = df[censor_col].to_numpy(dtype=bool)

    loess = sm.nonparametric.lowess
    fit_mask = (~censored) & np.isfinite(cov) & np.isfinite(gc)

    # Nothing to fit: every window is censored, or has no finite coverage, or
    # there are no windows at all. Decline with an IDENTITY curve rather than
    # raising -- the same shape otr_fit() returns when it cannot fit a tent, and
    # for the same reason: a stage that cannot measure a bias should contribute
    # nothing, not take the run down. LOWESS on empty arrays returns a (0, 2)
    # array and np.interp then raises "array of sample points is empty".
    #
    # A flat 1.0 makes apply_gc_correction() an exact no-op (gc_corr_norm_cov ==
    # norm_raw_cov, gc_corr_fact == 1.0), which is what "no GC correction was
    # applied" has to mean if the emission offsets the HMM composes from
    # gc_corr_fact are to stay honest.
    if not fit_mask.any():
        return {
            "gc_sorted": np.array([0.0, 1.0]),
            "fit_sorted": np.array([1.0, 1.0]),
            "floor": 1e-6,
            "gc_grid": np.linspace(0.0, 1.0, GC_TAU_GRID),
            "gc_tau": np.zeros(GC_TAU_GRID),
        }

    gc_sorted = fit_sorted = None
    for _ in range(max(1, n_robust_iter)):
        gc_f = gc[fit_mask]
        cov_f = cov[fit_mask]
        sm_out = loess(
            cov_f, gc_f,
            frac=0.3, it=1, delta=0.0,
            is_sorted=False, missing='none', return_sorted=True,
        )
        gc_sorted, fit_sorted = sm_out[:, 0], sm_out[:, 1]

        expected = np.interp(gc_f, gc_sorted, fit_sorted)
        resid = cov_f - expected
        med_resid = np.median(resid)
        mad = np.median(np.abs(resid - med_resid))
        if mad <= 0:
            break
        sigma = 1.4826 * mad
        keep = np.abs(resid - med_resid) <= (resid_mad * sigma)

        new_mask = fit_mask.copy()
        idx = np.where(fit_mask)[0]
        new_mask[idx] = keep

        if new_mask.sum() == fit_mask.sum() or new_mask.sum() < 0.5 * fit_mask.sum():
            fit_mask = new_mask
            break
        fit_mask = new_mask

    fit_ref = np.median(fit_sorted)
    floor = fit_floor_frac * fit_ref if np.isfinite(fit_ref) and fit_ref > 0 else 1e-6

    # How well is this curve determined at each GC? The correction is applied as
    # a DIVISOR, and its error is multiplied by copy number, so a window whose
    # offset is uncertain deserves less weight inside an amplification than one
    # whose offset is nailed down. run_HMM turns this into emission variance; see
    # offset_tau() and log_emission_with_offsets().
    grid = np.linspace(float(np.min(gc_sorted)), float(np.max(gc_sorted)),
                       GC_TAU_GRID)
    tau = _gc_curve_tau(gc, cov, fit_mask, grid, n_surrogates=tau_surrogates,
                        seed=tau_seed)

    return {"gc_sorted": gc_sorted, "fit_sorted": fit_sorted, "floor": floor,
            "gc_grid": grid, "gc_tau": tau}


def refit_gc_bias_pooled(per_genome):
    """Pooled GC refit for the SECOND pass, composed into ONE total GC curve.

    Why a second pass exists at all: `gc_corr_norm_cov` is flat against GC by
    construction, but the OTR tent varies with POSITION, and position is
    correlated with GC -- measured r between GC% and the fitted tent is 0.245 on
    adp1_mgd06_lb, 0.110 on ltee_ara_p5_75k_exp, 0.075 on CWBI's chromosome. So
    dividing by the tent puts a GC trend back into coverage the GC stage had just
    removed.

    It also sees a censor the first pass could not: `add_cn_censor` has by now
    folded the first pass's copy-number calls into `exclude_from_fit`, so every
    window the HMM did not call CN = 1 is out of this fit. The first pass had only
    near-zero depth and repeat overlap to go on, neither of which catches an
    amplification.

    Pooled, like the first pass and for the same reason: GC bias is a property of
    the sequencing chemistry rather than of any one reference, so it is fitted
    once across every table in the run. That is why this cannot live inside the
    per-sequence loop -- every sequence has to be OTR-corrected and called before
    the fit can see the pooled residual.

    COMPOSITION IS EXACT. Both passes are functions of GC alone, so

        raw / (g1(gc) * t(x) * g2(gc))    =>    G(gc) = g1(gc) * g2(gc)

    and the total GC correction is a single curve. `gc_corr_fact` becomes G and
    `gc_corr_norm_cov` becomes raw / G, so the column and its factor agree -- they
    did not before, when this pass updated the factor and left the column at
    raw/g1. That agreement is load-bearing now: raw/G is what the second OTR fit
    is handed as its input. The components stay on the frame as
    `gc_corr_fact_pass1` / `gc_corr_fact_pass2` so the correction is auditable and
    the reported curves can be drawn separately.

    SCOPE. Fitted across every sequence and applied to every sequence, including
    ones where no ramp was detected and so no tent was divided out. Deliberate:
    GC bias belongs to the sequencing chemistry, so one curve should describe the
    whole run rather than a different correction landing on each reference
    depending on whether its own OTR fit happened to clear a significance gate.

    Returns ({genome_id: df}, fit2).
    """
    ids = list(per_genome)
    if not ids:
        return dict(per_genome), None
    pooled = pd.concat([per_genome[g] for g in ids], ignore_index=True)

    # Fitted on the OTR-corrected coverage -- the residual with the current tent
    # removed, which is what makes this a backfitting step rather than a second
    # guess at the same thing.
    scratch = pooled.copy()
    scratch["norm_raw_cov"] = pooled["otr_gc_corr_norm_cov"]
    fit2 = fit_gc_bias(scratch)
    scratch = apply_gc_correction(scratch, fit2)

    g1 = pooled["gc_corr_fact"].to_numpy(dtype=float)
    g2 = scratch["gc_corr_fact"].to_numpy(dtype=float)
    pooled["gc_corr_fact_pass1"] = g1
    pooled["gc_corr_fact_pass2"] = g2
    pooled["gc_corr_fact"] = g1 * g2

    # G = g1 * g2, so log G = log g1 + log g2 and the RELATIVE variances add.
    # The two are fitted on different data (raw vs OTR-corrected, and the second
    # on CN=1 windows only), so treating them as independent is the right
    # first-order statement.
    if "gc_corr_tau" in pooled and "gc_corr_tau" in scratch:
        t1 = pooled["gc_corr_tau"].to_numpy(dtype=float)
        t2 = scratch["gc_corr_tau"].to_numpy(dtype=float)
        pooled["gc_corr_tau_pass1"] = t1
        pooled["gc_corr_tau_pass2"] = t2
        pooled["gc_corr_tau"] = np.sqrt(t1 ** 2 + t2 ** 2)

    # raw / G, freezing deletions at exactly 0 the way apply_gc_correction does,
    # so a real deletion is not divided back up toward 1.
    raw = pooled["norm_raw_cov"].to_numpy(dtype=float)
    total = np.where(np.isfinite(g1 * g2) & (g1 * g2 > 0), g1 * g2, 1.0)
    corrected = np.zeros_like(raw)
    if "is_deletion" in pooled.columns:
        live = ~pooled["is_deletion"].to_numpy(dtype=bool)
    else:
        live = np.ones(len(pooled), dtype=bool)
    corrected[live] = raw[live] / total[live]
    pooled["gc_corr_norm_cov"] = corrected

    # The residual this pass's own fit produced, raw/(g1*t*g2). Kept because it is
    # what the GC-pass-2 row of the correction-stages figure draws as its "after":
    # that row shows the fit's own before/after, which is not the same series as
    # the composed raw/G the next stage consumes.
    pooled["gc2_resid_cov"] = scratch["gc_corr_norm_cov"].to_numpy(dtype=float)

    out = dict(per_genome)
    for genome_id in ids:
        sub = pooled[pooled["genome_id"] == genome_id].copy()
        sub.reset_index(drop=True, inplace=True)
        out[genome_id] = sub
    return out, fit2


def plot_gc_passes(per_genome, output):
    """The two GC passes and their product, as one figure.

    g1 is the pooled fit on raw coverage; g2 the pooled fit on OTR-corrected
    coverage; G = g1 * g2 is the total GC correction actually applied. Drawn
    together because the interesting fact is that they OPPOSE each other at the
    extremes -- on ltee_ara_p5_75k_exp g1 climbs to 1.159 at high GC while g2
    falls to 0.926, so G reaches only 1.073. The second pass is mostly removing
    correction the first pass over-applied to the replication ramp.

    Pooled across sequences, so one file per run, like gc_cor_plots(). Makes its
    own directory as that function does. Returns the path.
    """
    pooled = pd.concat(per_genome.values(), ignore_index=True)
    label = "_and_".join(sorted(str(g) for g in per_genome))
    samplename = sample_prefix(output) + label
    savedir = os.path.join(output, "GC_bias")
    os.makedirs(savedir, exist_ok=True)

    gc = pooled["gc_percent"].to_numpy(dtype=float)
    keep = np.ones(len(pooled), dtype=bool)
    for col in ("is_deletion", "is_redundant"):
        if col in pooled:
            keep &= ~pooled[col].to_numpy(dtype=bool)
    if not keep.any():
        keep = np.ones(len(pooled), dtype=bool)

    order = np.argsort(gc[keep])
    x = gc[keep][order]
    g1 = pooled["gc_corr_fact_pass1"].to_numpy(dtype=float)[keep][order]
    g2 = pooled["gc_corr_fact_pass2"].to_numpy(dtype=float)[keep][order]
    tot = pooled["gc_corr_fact"].to_numpy(dtype=float)[keep][order]
    g1, g2, tot = (c / np.median(c) for c in (g1, g2, tot))

    lo, hi = np.percentile(x, [1, 99])
    m = (x >= lo) & (x <= hi)

    def span(c):
        return 100.0 * float(c[m].max() - c[m].min())

    plt.figure(figsize=(10, 8))
    plt.axhline(1.0, color="0.7", lw=0.8)
    plt.plot(x, g1, color="tab:blue", lw=1.8,
             label=f"g1: fitted on raw coverage (span {span(g1):.1f}%)")
    plt.plot(x, g2, color="tab:orange", lw=1.8,
             label=f"g2: fitted after OTR (span {span(g2):.1f}%)")
    plt.plot(x, tot, color="black", lw=2.4,
             label=f"G = g1 x g2, total applied (span {span(tot):.1f}%)")
    plt.axvspan(x.min(), lo, color="0.9", zorder=0)
    plt.axvspan(hi, x.max(), color="0.9", zorder=0)
    plt.xlabel("GC fraction of the window's fragment-sized neighbourhood")
    plt.ylabel("correction divisor, normalised to its median")
    plt.title(f"{samplename}_GC bias: both passes and their product\n"
              f"grey bands are outside the 1st-99th GC percentile",
              fontsize=10)
    plt.legend(loc="best", fontsize=9)

    path = os.path.join(savedir, "%s_GC_passes.pdf" % samplename.replace(" ", "_"))
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    return path


def apply_gc_correction(df, gc_fit, deletion_col="is_deletion"):
    """
    Apply stage of GC-bias correction: interpolate the fitted LOWESS curve
    at EVERY window's GC% (both fit-eligible and censored windows) and
    divide raw normalized coverage by it.

    `relative_copy_number` is this sequence's copies relative to the longest one
    in the run (see relative_copy_numbers()); it is written straight into the
    results JSON. The default of 1.0 is the honest answer for a caller holding a
    single frame -- it is its own longest sequence.

    Only windows flagged `deletion_col` (genuine near-zero/outlier
    coverage) are frozen at zero. Redundant-coverage windows (censored
    from fitting but NOT flagged as deletions) still get a real corrected
    value here -- masked for FITTING but not zeroed in the CORRECTED
    output, avoiding spurious deletion calls over repeats.

    Returns
    -------
    df : DataFrame with gc_corr_norm_cov and gc_corr_fact columns added.
    """
    df = df.copy()
    cov = df["norm_raw_cov"].to_numpy(dtype=float)
    gc = df["gc_percent"].to_numpy(dtype=float)
    is_deletion = df[deletion_col].to_numpy(dtype=bool)

    gc_out = np.interp(gc, gc_fit["gc_sorted"], gc_fit["fit_sorted"])
    gc_out = np.clip(gc_out, gc_fit["floor"], None)

    gc_corr = np.zeros_like(cov)
    valid = ~is_deletion
    gc_corr[valid] = cov[valid] / gc_out[valid]

    df["gc_corr_norm_cov"] = gc_corr
    df["gc_corr_fact"] = gc_out
    # How well the divisor itself is determined at this window's GC. run_HMM
    # turns it into emission variance; absent means "treat the curve as exact",
    # which is what every caller predating this did.
    if gc_fit.get("gc_tau") is not None and gc_fit.get("gc_grid") is not None:
        df["gc_corr_tau"] = np.interp(gc, gc_fit["gc_grid"], gc_fit["gc_tau"])
    return df


def _pooled_gc_stage(df_pooled):
    """mask -> fit -> apply, on one pooled frame. Extracted because the fragment
    scan can change `gc_percent`, and everything fitted against the old axis then
    has to be refitted against the new one."""
    df_pooled = mask_coverage_windows(df_pooled)
    return apply_gc_correction(df_pooled, fit_gc_bias(df_pooled))


def apply_frag_size(per_genome, gc_flags, win, frag):
    """Recompute gc_percent at a new fragment size and redo the pooled GC stage.

    The first pass has to happen before the scan can run -- it needs the tent and
    the copy-number calls to control its confounds -- so by the time a fragment
    size is chosen, `gc_corr_fact` was fitted against the OLD gc_percent. Both
    have to be rebuilt together: a curve fitted on one GC axis and applied on
    another is not a correction, it is a shuffle.

    `norm_raw_cov` is deliberately NOT recomputed. It is the pooled median
    normalisation, which has nothing to do with the fragment size.
    """
    out = {}
    for genome_id, df in per_genome.items():
        df = df.copy()
        flags = gc_flags.get(genome_id)
        if flags is not None and "win_st" in df and len(df):
            win_st0 = (df["win_st"].to_numpy(dtype=np.int64)
                       - int(df["win_st"].min()))
            df["gc_percent"] = gc_percent_for_frag(flags, win_st0, win, frag)
        out[genome_id] = df

    ids = list(out)
    pooled = _pooled_gc_stage(pd.concat([out[g] for g in ids], ignore_index=True))
    for genome_id in ids:
        sub_df = pooled[pooled["genome_id"] == genome_id].copy()
        sub_df.reset_index(drop=True, inplace=True)
        out[genome_id] = sub_df
    return out


def process_multi_genome(
    coverage_inputs,
    output_prefix,
    win=100,
    step=100,
    frag=400,
    collect_gc_flags=False,
):
    """
    Preprocess every coverage table, pool them for GC correction, plot the pooled bias,
    then return per-genome GC-corrected DataFrames keyed by sequence ID.

    `coverage_inputs` is an ordered {genome_id: path} mapping, as produced by
    resolve_coverage_inputs(). Resolution is the caller's job so that a bad set of inputs
    is rejected before any output directory is created.

    GC bias is fitted ONCE across every table -- chromosome, plasmids and contigs
    together -- because it is a property of the sequencing chemistry, not of any one
    reference. OTR correction and CN calling then happen per reference, back in
    get_CNV.main(). Note the shared global median this implies: the tables passed in one
    call should be the references of a single sample, not several samples.
    """

    preprocessed = {}
    gc_flags = {}
    for genome_id, path in coverage_inputs.items():
        df_raw = read_coverage_table(path)
        df_pre = preprocess(df_raw, win=win, step=step, frag=frag)
        df_pre["genome_id"] = genome_id
        preprocessed[genome_id] = df_pre
        # Per-base G-or-C flags, kept only when the caller intends to scan
        # fragment sizes -- 1 byte per base, so 4.6 MB on REL606, against the
        # 37 MB a prefix sum would cost.
        if collect_gc_flags:
            gc_flags[genome_id] = reference_gc_flags(df_raw)

    df_pooled = pd.concat(preprocessed.values(), ignore_index=True)

    global_median = df_pooled["read_count_cov"].median()
    # Every table in the run is empty or all-zero. Dividing by it would put NaN
    # (0/0) into norm_raw_cov and make every downstream mask nonsense; 1.0 leaves
    # the zeros as zeros, which is what they are. Gated on non-positive, so it
    # cannot touch a real median however many zero windows a genome carries.
    if not np.isfinite(global_median) or global_median <= 0:
        print("WARNING: no sequence in this run has any coverage; "
              "normalising against 1.0 and reporting no bias.")
        global_median = 1.0
    df_pooled["norm_raw_cov"] = df_pooled["read_count_cov"] / global_median

    df_pooled = _pooled_gc_stage(df_pooled)

    gc_cor_plots(df_pooled, output_prefix)

    per_genome_corrected = {}
    for genome_id in coverage_inputs:
        df_g = df_pooled[df_pooled["genome_id"] == genome_id].copy()
        df_g.reset_index(drop=True, inplace=True)
        per_genome_corrected[genome_id] = df_g

    if collect_gc_flags:
        return per_genome_corrected, gc_flags
    return per_genome_corrected


def plot_otr_corr(df, output, ori, ter):

    genome_id = str(df["genome_id"].iloc[0])
    samplename = sample_prefix(output) + genome_id
    saveplt = str(output+"/OTR_corr/")
  
    plt.figure(figsize=(10, 8))

    # Repeat windows are not drawn. On CWBI's chromosome they reach 18x the
    # single-copy level on zero unique coverage, which set the y-axis so the 1.17
    # ramp this figure exists to show was invisible.
    keep = plottable(df)
    drawn = df.loc[keep]
    n_hidden = int((~keep).sum())

    plt.scatter(drawn["win_st"], drawn["norm_raw_cov"], color="gray",
                label="Raw reads", s=8, alpha=0.2)
    plt.scatter(drawn["win_st"], drawn["gc_corr_norm_cov"], color="black",
                label="GC corrected", marker="*", s=15, alpha=0.5)
    plt.scatter(drawn["win_st"], drawn["otr_gc_corr_norm_cov"], color="orange",
                label="Ori/Ter bias corrected", s=20, alpha=0.85,
                marker=mplt.markers.MarkerStyle(marker="o", fillstyle="full"))
    # The fitted tent is a model evaluated everywhere, so it stays whole. The
    # median filter is a statistic OF the data, so it breaks where the data is
    # not drawn rather than bridging the gap with a value taken from repeats.
    plt.plot(df["win_st"], df["otr_gc_corr_fact"], color="black",
             label="OTR-bias-fit-line")
    med_fil = df["gc_cor_med_fil"].to_numpy(dtype=float).copy()
    med_fil[~keep] = np.nan
    plt.plot(df["win_st"], med_fil, color="blue", label="Med-fil")
    
    plt.axvline(x=ter, color='r', linestyle=':', label=f'Terminus: {ter}')
    plt.axvline(x=ori, color='r', linestyle=':', label=f'Origin: {ori}')
    plt.xlabel("Window (Genomic position)")
    plt.ylabel("Normalized read coverage")
    title = f'{samplename}_Ori/Ter bias correction'
    if n_hidden:
        title += f"\n{n_hidden} repeat window(s) not shown"
    plt.title(title)
    plt.legend(loc = 'upper right')

    plt_full_path = os.path.join(saveplt,'%s_OTR_corr.pdf' % samplename.replace(' ', '_'))
    plt.savefig(plt_full_path, format = 'pdf', bbox_inches = 'tight')
    
    df.reset_index(drop = True)
    
    plt.close()


def otr_predict(x, x_ori, x_ter, y_ori, y_ter, genome_len):
    """
    Predict coverage at position(s) `x` from a circular two-segment
    piecewise-linear ("tent") model anchored at (x_ori, y_ori) and
    (x_ter, y_ter).

    Walking forward (increasing position, wrapping at genome_len) from
    x_ori to x_ter traces one straight line; walking forward from x_ter
    back to x_ori traces the other. Together the two segments form a
    single closed shape around the whole circular genome -- this is the
    model fit as ONE unit across positions 1..genome_len, not two
    independently-fit pieces.

    All distances are computed modulo genome_len, so the model is
    agnostic to where the genome's coordinate origin (position 0/1)
    happens to fall relative to the biological origin/terminus. If x_ori
    or x_ter sits at (or near) the coordinate boundary, this same
    function produces a V-shaped or inverted-V-shaped profile when
    plotted on a linear axis -- that is just a different viewing angle on
    the same circular tent, not a different case for this function.

    Parameters
    ----------
    x : array-like
        Position(s) at which to predict coverage.
    x_ori, x_ter : float
        Fitted origin/terminus positions. Need not be integers or already
        wrapped into [0, genome_len) -- the modulo arithmetic here handles
        any real value, though callers using the RESULT as an array index
        (e.g. to report a window number) must wrap it themselves first
        (see otr_fit()).
    y_ori, y_ter : float
        Fitted coverage values at the origin/terminus.
    genome_len : float
        Genome length (number of windows) for the circular wraparound.

    Returns
    -------
    np.ndarray of predicted coverage values, one per element of x.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    L_A = (x_ter - x_ori) % genome_len       # arc length walking ori -> ter
    L_B = genome_len - L_A                    # arc length walking ter -> ori
    d = (x - x_ori) % genome_len               # distance from ori, walking forward
    on_a = d <= L_A

    y_pred = np.empty_like(x)

    t_a = np.divide(d, L_A, out=np.zeros_like(d), where=L_A > 0)
    y_pred[on_a] = y_ori + t_a[on_a] * (y_ter - y_ori)

    d_b = d - L_A
    t_b = np.divide(d_b, L_B, out=np.zeros_like(d_b), where=L_B > 0)
    y_pred[~on_a] = y_ter + t_b[~on_a] * (y_ori - y_ter)

    return y_pred


def _otr_design_matrix(x, x_ori, x_ter, genome_len):
    """
    Coefficient matrix M such that otr_predict(x, ...) == M @ [y_ori, y_ter]
    for FIXED (x_ori, x_ter).

    otr_predict() is affine in (y_ori, y_ter) once the breakpoints are
    fixed -- this is what lets the optimizer search over just the 2
    breakpoint POSITIONS instead of jointly optimizing all 4 parameters:
    for any candidate (x_ori, x_ter), the best-fitting (y_ori, y_ter) is
    solved exactly via ordinary least squares (see
    _otr_concentrated_rss()) rather than guessed or optimized alongside
    the positions.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    L_A = (x_ter - x_ori) % genome_len
    L_B = genome_len - L_A
    d = (x - x_ori) % genome_len
    on_a = d <= L_A

    M = np.zeros((len(x), 2))
    t_a = np.divide(d, L_A, out=np.zeros_like(d), where=L_A > 0)
    M[on_a, 0] = 1 - t_a[on_a]
    M[on_a, 1] = t_a[on_a]

    d_b = d - L_A
    t_b = np.divide(d_b, L_B, out=np.zeros_like(d_b), where=L_B > 0)
    M[~on_a, 0] = t_b[~on_a]
    M[~on_a, 1] = 1 - t_b[~on_a]
    return M


def _otr_concentrated_rss(breakpoints, x, y, genome_len):
    """
    The error function minimized to fit the whole-genome OTR model.

    Given candidate breakpoints (x_ori, x_ter), solves for the OPTIMAL
    (y_ori, y_ter) in closed form via ordinary least squares (using
    _otr_design_matrix()'s affine structure) and returns the resulting
    residual sum of squares between otr_predict(x; params) and the actual
    observed y across the WHOLE genome (all windows in x/y at once, not
    per-arc). This RSS is what scipy.optimize.minimize searches over,
    reduced to just the 2 breakpoint-position parameters -- (y_ori,
    y_ter) never need to be searched directly.

    Returns
    -------
    (rss, y_ori, y_ter) : the fit quality and the optimal anchor values
    for these breakpoints.
    """
    x_ori, x_ter = breakpoints
    if abs((x_ter - x_ori) % genome_len) < 1 or abs((x_ori - x_ter) % genome_len) < 1:
        return np.inf, np.nan, np.nan
    M = _otr_design_matrix(x, x_ori, x_ter, genome_len)
    beta, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
    y_pred = M @ beta
    rss = float(np.sum((y - y_pred) ** 2))
    return rss, beta[0], beta[1]


# ---------------------------------------------------------------------------
# Deciding whether the OTR tent is REAL, and whose breakpoints to believe.
#
# Everything below rests on one identity. _otr_design_matrix()'s rows sum to
# exactly 1 -- on arc A they are (1 - t_a, t_a), on arc B (t_b, 1 - t_b) -- so
# for fixed breakpoints the tent's column span equals span{1, u}, where
# u = M[:, 1] is the "phase" running 0 at the origin, 1 at the terminus, and
# back to 0. Fitting the anchor values is therefore ordinary least squares on a
# single regressor, and
#
#     1 - _otr_concentrated_rss(bp, x, y, L)[0] / SST  ==  r-squared(u, y)
#
# which is a dot product rather than a least-squares solve. That collapses "one
# lstsq per candidate breakpoint pair" into a single matrix multiply over the
# whole grid AND the whole surrogate batch at once, and -- the part that
# matters for correctness rather than speed -- it is the SAME objective the
# Nelder-Mead search in otr_fit() minimises, not an approximation of it.
#
# Do not "simplify" _otr_grid_scores() back into a loop over minimize(): the
# bootstraps below evaluate the grid on every one of 1000 surrogates, which is
# affordable only in this closed form.
# ---------------------------------------------------------------------------

#: Circular block bootstrap size. Deliberately the same 1000 as
#: DEFAULT_SKEW_SURROGATES (defined further down, with the GC-skew gate) so every
#: p-value CNery reports floors at the same 1/(B+1) and means the same thing.
#: Measured cost at this size is ~0.12 s per sequence for the detection gate and
#: ~0.2 s for the likelihood-ratio test, against ~4.4-6.8 s for preprocess().
DEFAULT_OTR_SURROGATES = 1000

#: A tent is only corrected for if it beats this. Same level as the GC-skew
#: gate: both are DETECTION gates, deciding whether to apply a correction at all.
OTR_MAX_P = 0.01

#: Origin and terminus must be roughly antipodal, consistent with bidirectional
#: replication. predict_ori_ter_from_skew() defaults to this same band, so the
#: two estimates are gated on one decision rather than two copies of it.
#:
#: These bound a SIGNED separation (x_ter - x_ori) mod L, which is how both
#: _otr_phase_grid() and otr_fit()'s refinement use them, and there the two ends
#: are distinct: s and L-s give the same unordered breakpoint pair. As a bound
#: on CIRCULAR distance -- min(s, L-s), which cannot exceed L/2 -- only the 0.35
#: end can ever bind, so do not reintroduce a `<= OTR_SEP_HI * genome_len` test
#: on that quantity and expect it to do anything.
OTR_SEP_LO, OTR_SEP_HI = 0.35, 0.65

#: The coverage series is decimated to at most this many uniform circular cells
#: before scoring, which caps the bootstrap cost independently of genome size
#: and of -w/-s. A 46,298-window frame and a 9,258-window one both land here.
OTR_SCORE_CELLS = 4000

#: Breakpoint search grid. Measured on ltee_ara_p5_75k_exp (m = 3086), the best
#: r-squared is 0.46391 at 32, 64 and 128 origins and 0.46392 at 256 -- the
#: surface is saturated well below this, and 64 keeps a 2x margin. The
#: likelihood-ratio verdicts are likewise unchanged from a 4x finer grid to a
#: 4x coarser one, which is what licenses a grid this cheap.
OTR_GRID_ORIGINS = 64
OTR_GRID_SEPARATIONS = 13

#: Block bootstrap shape. Coverage autocorrelation is far longer than GC skew's,
#: and unlike the skew gate the block length here governs VALIDITY, not just
#: power. Measured false-positive rate at a nominal 1% on AR(1) nulls: at
#: block ~= 5*tau it is 0.00, at block ~= 2.6*tau it is 0.03-0.08, and at
#: block ~= tau it is 0.10-0.33. Hence the floor at 5*tau as well as the
#: target block count.
OTR_TARGET_BLOCKS = 20
OTR_BLOCK_AUTOCORR_MULTIPLE = 5
OTR_MIN_BLOCKS = 8
OTR_MIN_BLOCK = 10

#: Below this many cells there is nothing to resample; the bootstrap declines
#: (p = 1.0, 0 surrogates) rather than inventing a null, as _skew_bootstrap_p()
#: does under 8 windows.
OTR_MIN_CELLS = 32

#: Coverage evidence required at the GC-skew breakpoints before its coordinates
#: may be used. None means "none": the skew's own bootstrap is the gate, and the
#: coverage only has to not contradict the orientation.
#:
#: The alternative is one constant wide and worth seeing. At 0.01 the skew arm
#: fires only on ltee_ara_p5_75k_exp, and there only to refine coordinates that
#: were already being corrected -- ZERO sequences are rescued from "not
#: detected". The fixed-breakpoint p-values across the eight authentic sequences
#: are 0.001, 0.001, 0.213, 0.243, 0.328, 0.343, 0.615 and 0.841, a clean gap,
#: so no threshold between 0.01 and 0.2 behaves any differently.
#:
#: The cost of None is explicit: on ltee_ara_m3_38k and adp1_mgd06_lb the ramp
#: that gets applied (ratio 1.069 and 1.067) is indistinguishable from noise by
#: the coverage's own evidence. It is applied because the reference sequence
#: says an origin is there and the coverage does not contradict it. What bounds
#: the damage is that the skew supplies only WHERE -- the amplitude is still
#: solved from the coverage, so a flat sample yields a flat tent. Measured over
#: 300 synthetic flat series: injected ratio median 0.999, 95th percentile
#: 1.051, maximum 1.089.
OTR_SKEW_MAX_P = None

#: Significance level for the coverage-vs-skew likelihood-ratio test. NOT the
#: 0.01 the two detection gates share, and deliberately so: those decide whether
#: to correct at all, while this one only chooses which of two already-credible
#: ramps supplies the breakpoints.
#:
#: Measured on a synthetic tent with AR(1) residuals (tau = 20): the rejection
#: rate under a true null is 7% at 0.05 and 0% at 0.01, and power at a
#: 5%-of-genome breakpoint displacement is 70% against 33%. At 0.01 the test is
#: conservative in the direction that favours the fallback, which is not the
#: safe direction here.
#:
#: On the eight authentic sequences every verdict is identical for any alpha in
#: 0.01..0.10, so this number currently decides nothing.
OTR_LR_ALPHA = 0.05

#: Surrogates for the residual-structure diagnostic. Deliberately NOT
#: DEFAULT_OTR_SURROGATES: that number sizes a p-value whose floor of 1/(B+1) is
#: load-bearing, while this one sizes a z-score whose only requirement is that
#: its Monte-Carlo error sit below the published precision. Measured on
#: ltee_ara_p1_50k_shift over 10 seeds, the score's standard deviation is 0.24 at
#: B=100, 0.08 at B=400 and 0.04 at B=1000 -- 400 is already finer than the two
#: decimals reported, and costs ~25 ms against the ~0.4 s the OTR bootstraps
#: already spend.
OTR_STRUCTURE_SURROGATES = 400

def _otr_decimate(y, keep, cells=OTR_SCORE_CELLS):
    """Coverage on a uniform circular grid of cells, with a WEIGHT per cell.

    Returns (values, weights). Each cell takes the mean of the unmasked windows
    falling in it, and its weight is HOW MANY there were. Everything downstream
    scores with those weights, which does two things:

      - A cell with no unmasked window at all contributes nothing, instead of
        being filled by interpolation from its neighbours. Interpolating was
        fabricating data that supports whatever trend the neighbours imply, and
        it was not a rare corner: on CWBI's plasmid_1, 121 of 232 cells had no
        unmasked window, so 52% of the scored series was invented and the
        statistic read r-squared 0.175 against 0.085 on the real windows.
      - A cell holding three windows now counts three times one holding one.
        Equal-weighting cells silently reweighted the genome wherever censoring
        was uneven, and it made the decimated statistic disagree with the
        full-resolution objective otr_fit() actually minimises, which weights
        per window.

    The cell count is additionally capped at the number of unmasked windows:
    asking for more cells than there are observations only manufactures empty
    ones.

    Deliberately NOT `y[keep]` compacted into a contiguous circle. That would
    shorten a 20 kb deletion to zero width and bend the tent through the gap;
    decimating on position keeps every arc length proportional to genomic
    coordinate. It also makes the bootstrap cost independent of genome size and
    of the window/step settings, which is what keeps the gate affordable.
    """
    y = np.asarray(y, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    n = y.size
    good = keep & np.isfinite(y)
    m = int(min(int(cells), n, max(int(good.sum()), 1)))
    if m < 1 or not good.any():
        return np.zeros(max(m, 0), dtype=float), np.zeros(max(m, 0), dtype=float)

    cell = (np.arange(n) * m) // n
    total = np.bincount(cell[good], weights=y[good], minlength=m).astype(float)
    weights = np.bincount(cell[good], minlength=m).astype(float)

    values = np.empty(m, dtype=float)
    filled = weights > 0
    values[filled] = total[filled] / weights[filled]
    # Empty cells are given the weighted mean purely so the series has no holes
    # for the block bootstrap to move around; their weight is 0, so they never
    # enter any r-squared, and a mean value landing on a weighted position under
    # resampling is the least informative value it could carry.
    values[~filled] = values[filled].mean() if filled.any() else 0.0
    return values, weights


def _otr_phase(m, x_ori, x_ter):
    """The tent's single regressor: 0 at the origin, 1 at the terminus, back to 0.

    Column 1 of _otr_design_matrix() evaluated on the decimated grid. Taken from
    that function rather than re-derived, so the geometry has exactly one
    definition and the modulo/wraparound handling cannot drift.
    """
    return _otr_design_matrix(np.arange(m, dtype=float), float(x_ori), float(x_ter), float(m))[:, 1]


def _otr_normalize_phases(rows, weights):
    """Weighted-centre and weighted-unit-normalise each phase row.

    With `u` centred and scaled so that sum(w * u^2) == 1, the weighted squared
    correlation against any series is just (sum(w * u * y_c))^2 / sum(w * y_c^2)
    -- still one matmul per surrogate batch, no per-candidate division and no
    lstsq, and now with cells weighted by how many real windows they hold.
    """
    P = np.atleast_2d(np.asarray(rows, dtype=float))
    w = np.asarray(weights, dtype=float)
    sw = w.sum()
    if sw <= 0:
        return np.zeros_like(P)
    P = P - (P * w).sum(axis=1, keepdims=True) / sw
    scale = np.sqrt((P * P * w).sum(axis=1, keepdims=True))
    scale[scale == 0] = 1.0
    return P / scale


def _otr_phase_grid(m, weights, sep_lo=OTR_SEP_LO, sep_hi=OTR_SEP_HI,
                    n_origins=OTR_GRID_ORIGINS, n_seps=OTR_GRID_SEPARATIONS):
    """(R, m) normalised phase rows over the band-restricted breakpoint grid.

    Restricting the grid to the same 35-65% separation band the gate uses is not
    an optimisation: the observed statistic is only ever acted on if it falls in
    the band, so the null must be the maximum over the band too, or observed and
    surrogate are drawn from different spaces. It turns OTR_SEP_LO/HI from an
    arbitrary threshold into the definition of the search space.

    Returns (phases, breakpoints) with breakpoints[i] == (x_ori, x_ter).
    """
    origins = np.linspace(0.0, float(m), int(n_origins), endpoint=False)
    seps = np.linspace(float(sep_lo), float(sep_hi), int(n_seps))
    rows, bps = [], []
    for xo in origins:
        for frac in seps:
            xt = (xo + frac * m) % m
            rows.append(_otr_phase(m, xo, xt))
            bps.append((float(xo), float(xt)))
    return _otr_normalize_phases(rows, weights), bps


def _otr_grid_scores(Y, phases, weights):
    """Best weighted tent r-squared for each column of `Y`, over `phases`.

    `Y` is (m,) or (m, B); the return is (B,). This one function scores the
    OBSERVED series and every surrogate -- the observed call is simply B == 1 --
    so "observed and null were scored the same way" is a property of there being
    one code path, not a convention someone has to maintain.

    `weights` are the per-cell window counts from _otr_decimate(), and they stay
    FIXED at their lattice positions while the bootstrap resamples values. That
    is the coherent pairing: where the repeats and deletions sit is a property
    of the reference, and the null being simulated is "the same censoring
    geometry, with the genome-scale trend destroyed" -- not "the censoring
    happened somewhere else".
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    w = np.asarray(weights, dtype=float)
    sw = w.sum()
    if sw <= 0:
        return np.zeros(Y.shape[1], dtype=float)

    Yc = Y - (w[:, None] * Y).sum(axis=0, keepdims=True) / sw
    sst = w @ (Yc * Yc)
    # Rows are already weight-normalised, so this is the weighted covariance and
    # the denominator needs no per-row term.
    C = (np.asarray(phases, dtype=float) * w) @ Yc
    best = np.max(C * C, axis=0)
    out = np.zeros_like(sst)
    nz = sst > 0
    out[nz] = best[nz] / sst[nz]
    return out


def _otr_autocorr_length(residual, max_lag=400):
    """First lag at which the autocorrelation drops below 1/e, at least 1.

    MUST be given the residual of the best-fitting tent, never the raw series.
    The ramp is itself long-range structure, so measuring on raw coverage lets
    the alternative hypothesis inflate the estimate -- on ltee_ara_p5_75k_exp
    tau reads 297 cells raw against 15 on the residual, which forces the block
    count down to 8 and pushes the p-value from 0.001 to 0.034. That is the test
    destroying its own detection.
    """
    r = np.asarray(residual, dtype=float)
    r = r - r.mean()
    n = r.size
    denom = float(r @ r)
    if n < 4 or denom <= 0:
        return 1
    thresh = 1.0 / np.e
    for lag in range(1, int(min(max_lag, n // 2)) + 1):
        if float(r[lag:] @ r[:-lag]) / denom < thresh:
            return lag
    return int(min(max_lag, max(1, n // 2)))


def _otr_cusum_range(residual, weights):
    """Kuiper's V for a weighted residual: how far its running sum wanders.

    A fit that is merely NOISY scatters its residuals about zero and its running
    sum stays near zero. A fit that is systematically WRONG over a stretch of
    genome accumulates same-sign residuals, and the running sum walks away and
    comes back. This measures that walk, scaled so it is comparable across
    sequences.

    Four choices here are load-bearing, and each has a plausible-looking
    alternative that is wrong:

    - The RANGE (max - min), not max|C|. The genome is circular, so max|C|
      depends on where the reference's coordinate 1 happens to fall and a
      circularly permuted copy of the same genome would score differently. The
      range is rotation-invariant -- this is Kuiper's V rather than
      Kolmogorov-Smirnov, and it is the same concern that makes preprocess()
      subtract the mean skew before cumulating for `cum_gc_skew`.

    - The sum accumulates w*r, not r. Under the weighted least-squares model
      Var(r_i) = s^2 / w_i, so Var(sum of w_i r_i) = s^2 * sum(w_i): the partial
      sum is Brownian in WEIGHT time, not in cell index. Dividing by sqrt(m)
      instead of sqrt(sum w) is wrong by sqrt(mean w) ~ 1.4 here, and drifts with
      how heavily the sequence is censored.

    - s^2 divides by (m - 2), the two degrees of freedom the tent's anchors
      remove. Using sum(w r^2)/sum(w) instead is wrong by mean w ~ 2.1, a factor
      1.46 in the returned statistic.

    - No re-centring. `w @ r` is already exactly 0, because
      _otr_normalize_phases() weight-centres the regressor, so this is a genuine
      bridge and Kuiper's asymptotics apply as-is. Subtracting the weighted mean
      "to be safe" would subtract zero and hide that invariant from the next
      reader.

    Returns (statistic, argmax_cell, argmin_cell). The two indices bracket the
    worst excursion and localise the misfit for free.
    """
    r = np.asarray(residual, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = r.size
    sw = w.sum()
    if m < 2 or sw <= 0:
        return 0.0, 0, 0

    s2 = float(w @ (r * r)) / max(m - 2, 1)
    if not np.isfinite(s2) or s2 <= 0:
        return 0.0, 0, 0

    c = np.cumsum(w * r)
    i_max, i_min = int(np.argmax(c)), int(np.argmin(c))
    stat = float((c[i_max] - c[i_min]) / np.sqrt(s2 * sw))
    return (stat if np.isfinite(stat) else 0.0), i_max, i_min


def _otr_residual_structure(residual, phases, weights, block,
                            n_surrogates=OTR_STRUCTURE_SURROGATES, seed=0):
    """How structured the residual is, as a z against a block-bootstrap null.

    REPORTED ONLY. Nothing gates on this, nothing selects a model with it.

    The raw statistic cannot be published on its own: coverage residuals are
    nowhere near white even under a perfect fit -- measured decorrelation lengths
    are 2 to 50 cells (2-45 kb) on GOOD fits -- so the raw number reads "high"
    everywhere and discriminates nothing. It also grows as sqrt(m), from 4.21 to
    9.84 as OTR_SCORE_CELLS goes 1000 to 8000 on the same sequence. The z is flat
    to +/-0.06 across that same range, which is what makes it reportable.

    The null is the same circular block bootstrap the detection gate uses, and
    for the same reason: block resampling PRESERVES short-range correlation while
    destroying long-range, so the surrogate distribution is exactly the intrinsic
    floor for this sequence. That is also why no simpler normalisation works.
    Measured alternatives, on the one sequence where a tent is known to be
    misfit (ltee_ara_p1_50k_shift, whose GC-skew terminus is 10.6% of the genome
    from the coverage fit's, leaving 1.4 Mb of same-sign residual):

      - raw statistic:      adp1_mgd06_lb, a merely WEAK fit, outscores it
      - lag-1 / Durbin-Watson, tau, Ljung-Q, runs: all rank the weak fit above
        the biased one, and bootstrap-normalising them is *identically*
        uninformative (z ~ 0.6) because block resampling preserves exactly the
        short-range correlation they measure -- tau is the FLOOR, not the signal
      - variance ratio V(b): reads z = 5-6 on the CWBI chromosome, r^2 = 0.005
      - Bartlett long-run variance at fixed bandwidth: reads 2.3-2.8 on the two
        plasmids, and cannot separate a 2x amplification over 15% of the genome
        from a 10% terminus displacement

    `phases` decides whether the null re-selects a tent, exactly as in
    _otr_bootstrap_p(): (R, m) for the coverage arm, whose breakpoints were
    chosen by looking at this coverage, so every surrogate must be allowed to
    choose its own; (1, m) for the GC-skew arm, whose breakpoints came from
    `ref_base` and involved no selection. Nothing below branches on R. Holding
    the phase fixed for the coverage arm would understate a data-chosen tent by
    0.85 z.

    Known limit, measured: a real copy-number event is partly absorbed rather
    than flagged, because it inflates tau, which lengthens the block, which
    raises the null with it. A 2x amplification over 5% of the genome reads 0.72
    against a 5% breakpoint displacement's 1.55; but a 2x amplification over 15%
    reaches 1.80, comparable to a 10% displacement. Large duplications are
    bounded here, not invisible.

    Returns (z, percentile), or (None, None) when there is too little to resample.
    """
    r = np.asarray(residual, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = r.size
    if m < OTR_MIN_CELLS or w.sum() <= 0:
        return None, None

    observed, _, _ = _otr_cusum_range(r, w)
    if observed <= 0:
        return None, None

    P = np.atleast_2d(np.asarray(phases, dtype=float))
    Pw = P * w
    sw = w.sum()
    rng = np.random.default_rng(seed)

    stats = np.empty(n_surrogates, dtype=float)
    done = 0
    chunk = max(1, min(250, n_surrogates))
    while done < n_surrogates:
        size = min(chunk, n_surrogates - done)
        idx = _otr_block_indices(rng, m, block, size)
        R = r[idx].T                                    # (m, size)
        R = R - (w[:, None] * R).sum(axis=0, keepdims=True) / sw
        # Let each surrogate fit -- and shed -- its own best tent, so the null
        # carries the same selection advantage the observed residual already had.
        C = Pw @ R                                      # (R, size)
        best = np.argmax(C * C, axis=0)
        R = R - P[best].T * C[best, np.arange(size)]
        for j in range(size):
            stats[done + j] = _otr_cusum_range(R[:, j], w)[0]
        done += size

    sd = float(stats.std())
    z = (observed - float(stats.mean())) / sd if sd > 0 else 0.0
    pct = float(np.count_nonzero(stats >= observed) + 1) / (n_surrogates + 1)
    return (float(z) if np.isfinite(z) else None), pct


def _otr_block_length(m, tau, block=None):
    """Block length in cells: enough blocks to shuffle, long enough to stay valid.

    Two constraints, and unlike the GC-skew gate the second one is about
    VALIDITY. Blocks shorter than a few multiples of the autocorrelation length
    leak the long-range structure into the surrogates and the test stops holding
    its nominal size -- measured at 10-33% false positives when block ~= tau
    against 0% at 5*tau. So the length is the LARGER of "aim for
    OTR_TARGET_BLOCKS blocks" and "at least 5 tau", then bounded so at least
    OTR_MIN_BLOCKS blocks survive.
    """
    if block is None:
        block = max(m // OTR_TARGET_BLOCKS, OTR_BLOCK_AUTOCORR_MULTIPLE * int(tau))
    upper = max(1, m // OTR_MIN_BLOCKS)
    return int(np.clip(int(block), min(OTR_MIN_BLOCK, upper), upper))


def _otr_block_indices(rng, m, block, size):
    """(size, m) circular block-bootstrap index matrix."""
    n_blocks = int(np.ceil(m / block))
    offsets = np.arange(block)
    starts = rng.integers(0, m, size=(size, n_blocks))
    idx = starts[:, :, None] + offsets[None, None, :]
    return idx.reshape(size, n_blocks * block)[:, :m] % m


def _otr_bootstrap_p(series, phases, weights, observed, block,
                     n_surrogates=DEFAULT_OTR_SURROGATES, seed=0):
    """Circular block bootstrap p-value for a tent r-squared. Covers BOTH arms.

    The null is "a sequence with this much local coverage autocorrelation but no
    genome-scale origin-to-terminus ramp": resampling whole blocks preserves the
    short-range structure, reshuffling their order destroys the long-range one.
    The RAW series is resampled, not residuals -- there is no fitted signal to
    hold fixed under a no-ramp null, and residual resampling would bake the
    alternative into it. (The likelihood-ratio test in _otr_lr_bootstrap_p()
    resamples residuals precisely because ITS null is a fitted model. The two
    disagree on purpose.)

    `phases` is (R, m) and its shape is the whole difference between the arms:

      R > 1  -- the free coverage fit. Its breakpoints were chosen by looking at
                the coverage, so the null must re-run that selection on every
                surrogate, exactly as _skew_score() re-runs the extrema search.
      R == 1 -- the GC-skew fit. Its breakpoints came from `ref_base`, which the
                coverage resampling never touches, so no selection has occurred
                and max-over-one-row IS the fixed statistic. Taking a maximum
                here would be a conservative error, not a safe default. This is
                why the skew arm is the more powerful of the two: on CWBI's
                chromosome it scores a SMALLER statistic (0.0152 against 0.0255)
                and returns a SMALLER p-value (0.213 against 0.278).

    Nothing in the body branches on R.

    Returns (p, surrogates_used). p is (#{surrogate >= observed} + 1) / (B + 1),
    so it is floored at 1/(B+1) and is an upper bound, not a measurement.
    """
    series = np.asarray(series, dtype=float)
    m = series.size
    if m < OTR_MIN_CELLS:
        return 1.0, 0

    rng = np.random.default_rng(seed)
    at_least = 0
    done = 0
    chunk = max(1, min(250, n_surrogates))
    while done < n_surrogates:
        size = min(chunk, n_surrogates - done)
        idx = _otr_block_indices(rng, m, block, size)
        scores = _otr_grid_scores(series[idx].T, phases, weights)
        at_least += int(np.count_nonzero(scores >= observed))
        done += size

    return (at_least + 1) / (n_surrogates + 1), int(n_surrogates)


def _otr_tent_fit(series, phase_row, weights):
    """(fitted, residual, r_squared) for one normalised phase row.

    With a weight-centred, weight-unit-norm regressor the weighted least-squares
    fit is just the projection, so this needs no lstsq and stays consistent with
    _otr_grid_scores() by construction. Zero-weight cells get a fitted value like
    any other -- the tent is defined everywhere -- but contribute to neither the
    coefficient nor the r-squared.
    """
    y = np.asarray(series, dtype=float)
    u = np.asarray(phase_row, dtype=float)
    w = np.asarray(weights, dtype=float)
    sw = w.sum()
    if sw <= 0:
        return y.copy(), np.zeros_like(y), 0.0
    ybar = float(w @ y) / sw
    yc = y - ybar
    c = float(w @ (u * yc))
    fitted = ybar + c * u
    sst = float(w @ (yc * yc))
    return fitted, y - fitted, (c * c / sst if sst > 0 else 0.0)


def _otr_lr_statistic(series, skew_phase, phases, weights):
    """Lambda = m * ln(RSS_skew / RSS_free), plus the two r-squareds.

    The two OTR models are NESTED. Both fit the anchor VALUES by ordinary least
    squares -- that is what _otr_design_matrix()'s affine structure buys. The
    GC-skew model additionally FIXES the breakpoint POSITIONS; the free model
    fits them too. The difference is exactly 2 parameters, which is what makes a
    likelihood ratio the right instrument here and "r-squared must beat the
    other by 10%" the wrong one.

    RSS_free is the minimum over the same band-restricted grid the detection
    statistic maximises over -- NOT a fresh Nelder-Mead fit. Comparing an
    unrestricted optimum against a band-restricted null is apples to oranges:
    fed Nelder-Mead's degenerate 78-window-separation spike on adp1_mgd06_lb,
    the same test returns p = 0.001 instead of 0.181 and hands a no-gradient
    sequence to the coverage fit.

    Lambda is clamped at 0. Nesting guarantees RSS_free <= RSS_skew
    mathematically, but grid discretisation does not, because the skew's
    breakpoints are snapped to the nearest grid cell. Measured, this never binds
    on real data and binds on 59/1000 surrogates of one sequence whose skew
    separation falls outside the band -- there the models genuinely are not
    nested over the search set, and without the clamp Lambda goes negative and
    biases the verdict toward the free fit.
    """
    m = np.asarray(series, dtype=float).size
    r2_skew = float(_otr_grid_scores(series, np.atleast_2d(skew_phase), weights)[0])
    r2_free = float(_otr_grid_scores(series, phases, weights)[0])
    r2_free = max(r2_free, r2_skew)
    rss_skew, rss_free = 1.0 - r2_skew, 1.0 - r2_free
    if rss_free <= 0 or rss_skew <= 0:
        return 0.0, r2_skew, r2_free

    # Snap a negligible difference to exactly zero. When the free grid's optimum
    # IS the skew's row -- the two agree -- floating-point dust can leave Lambda
    # at ~1e-13 instead of 0, and that is not harmless: a large share of
    # surrogates score exactly 0 too, so `surrogate >= observed` flips from true
    # to false for all of them and the p-value collapses from 1.0 to ~0.4. An
    # r-squared difference below this is not evidence about anything.
    if r2_free - r2_skew <= 1e-12:
        return 0.0, r2_skew, r2_free
    return max(0.0, m * float(np.log(rss_skew / rss_free))), r2_skew, r2_free


def _otr_lr_bootstrap_p(series, skew_phase, phases, weights, observed, block,
                        n_surrogates=DEFAULT_OTR_SURROGATES, seed=0):
    """Bootstrap p-value for Lambda under the null "the GC-skew tent is the truth".

    Fit the tent at the skew's breakpoints, hold that fit FIXED, circular-block
    resample its RESIDUALS, add them back, and recompute Lambda on each
    surrogate. A bootstrap for a composite null must simulate FROM the null
    model, and here the null is a fully specified one -- unlike the detection
    gate's, whose null has no fitted signal to hold fixed. Resampling the raw
    series here would answer "is there a ramp", which is not what is being
    arbitrated.

    The resampling is by BLOCKS, and that is the entire reason this function
    exists rather than a chi-square(2) lookup. Measured on all eight authentic
    sequences: with IID residuals the null of Lambda IS chi-square(2) to within
    Monte-Carlo error (median 1.2-1.9 against 1.39, 99th percentile 8.3-9.3
    against 9.21), and it stays chi-square(2) even with the separation band
    removed entirely -- neighbouring tents are near-collinear, so the maximum
    over breakpoints costs essentially nothing despite the breakpoints being
    change-points. What breaks chi-square(2) is that the residuals remain
    spatially autocorrelated, which inflates Lambda's 99th percentile by
    2.8x-128x (24x-51x on the E. coli chromosomes).

    That is not academic. chi-square(2) reports p < 0.001 for every one of the
    eight sequences, including ltee_ara_p5_75k_exp, where the two tents agree to
    1.8% of the genome and the GC skew lands on REL606's oriC -- the bootstrap
    p-value there is 0.82.

    Returns (p, surrogates_used).
    """
    series = np.asarray(series, dtype=float)
    m = series.size
    if m < OTR_MIN_CELLS:
        return 1.0, 0

    fitted, residual, _ = _otr_tent_fit(series, skew_phase, weights)
    if not np.isfinite(residual).all() or float(residual @ residual) <= 0:
        return 1.0, 0

    rng = np.random.default_rng(seed)
    at_least = 0
    done = 0
    chunk = max(1, min(250, n_surrogates))
    while done < n_surrogates:
        size = min(chunk, n_surrogates - done)
        idx = _otr_block_indices(rng, m, block, size)
        surro = fitted[:, None] + residual[idx].T
        r2s = _otr_grid_scores(surro, np.atleast_2d(skew_phase), weights)
        r2f = np.maximum(_otr_grid_scores(surro, phases, weights), r2s)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = m * np.log((1.0 - r2s) / (1.0 - r2f))
        lam = np.where(np.isfinite(lam), np.maximum(lam, 0.0), 0.0)
        at_least += int(np.count_nonzero(lam >= observed))
        done += size

    return (at_least + 1) / (n_surrogates + 1), int(n_surrogates)


#Fit the coverage based on the presence and the degree of origin and terminus biased read counts observed
#
# Fits the ENTIRE circular coverage profile (positions 1..genome_len) as
# ONE unit: a two-segment piecewise-linear ("tent") model connecting an
# origin anchor (x_ori, y_ori) and a terminus anchor (x_ter, y_ter) --
# see otr_predict() -- rather than picking a peak/trough by argmax/argmin
# and then fitting each of the resulting two arcs independently.
#
# For any FIXED pair of breakpoints (x_ori, x_ter), the predicted
# coverage is affine in (y_ori, y_ter) (_otr_design_matrix()), so the
# best-fitting anchor VALUES are solved exactly via ordinary least
# squares; the only real search is over the 2 breakpoint POSITIONS
# (_otr_concentrated_rss() is the error function minimized for that
# search). This is a well-posed 2-parameter problem, unlike the
# scipy.optimize.minimize call this design replaces, which packed only 2
# real degrees of freedom (a line's slope and intercept) into 4 free
# parameters -- an infinite flat valley with no unique answer -- and
# whose "solution" a later algebraic identity collapsed into fixed
# constants regardless of what the optimizer returned.
#
# Windows flagged as deletions or redundant coverage (via
# mask_coverage_windows()) are excluded from BOTH the multi-start seed
# search and the least-squares fit itself, so a repeat's inflated total
# coverage or a real deletion's near-zero coverage cannot distort the
# fitted breakpoints or anchor values.
#
# The search is a band-constrained multi-start: the exhaustive grid's argmax
# plus n_seeds origins spread around the circle, each refined by Nelder-Mead
# over (x_ori, separation), lowest RSS kept. Multi-start is for genuine
# MULTIMODALITY -- CWBI's plasmid_1 has 74 distinct local minima over 111
# usable windows -- and not, as this comment used to claim, for the kinks
# where a window switches arcs. Those are real (2*n_fit lines in the parameter
# plane, one per window per axis) but measured at ~0.03% of the local slope,
# and with xatol=0.5 the simplex terminates inside a single smooth cell
# without ever resolving one. Each fit is fast, so the multi-start is cheap.
#
# Because every distance here is computed modulo genome_len, the fit is
# agnostic to where the genome's coordinate origin (position 0/1) happens
# to fall relative to the biological origin/terminus -- a V-shaped or
# inverted-V-shaped profile (origin or terminus sitting at the coordinate
# boundary, common when an assembly is oriented to start at oriC) is not
# a special case, just a different linear "cut" of the same circular
# model. Verified against synthetic data with: origin/terminus in both
# genome-coordinate orderings, both coordinate-boundary orientations
# (V-shape and inverted-V-shape), an injected repeat that previously
# hijacked the origin search, a deletion that previously hijacked the
# terminus search, and a deletion-adjacent "shoulder" window -- all
# recovered within 1-2 window-widths of the true simulated positions, and
# noticeably more accurately than the previous per-arc approach on
# realistically noisy Poisson-sampled coverage.
def _otr_detail(s_free=None, p_free=None, s_skew=None, p_skew=None,
                lam=None, p_lr=None, surrogates=0, skew_result=None,
                source="not corrected", structure=None, decorr_bp=None):
    """The evidence behind the OTR verdict, ready for the results JSON.

    Written whatever the verdict, so a REJECTED fit stays diagnosable from the
    file -- and so does an accepted one whose coverage evidence is thin, which
    under OTR_SKEW_MAX_P = None is a real case rather than a hypothetical. Every
    value is rounded here and passed through _json_safe() on the way out, so a
    missing statistic emits `null` and never a bare NaN.

    These are additions only, and "Origin window"/"Terminus window" stay
    non-null, because breseq parses this file with nlohmann and adding keys is
    the only change that is safe there. The one exception already taken is
    "Origin-to-Terminus/Bias Ratio", spelled correctly as of this release: the
    key was "Termius" for breseq's benefit, and fixing the spelling knowingly
    breaks that reader until it is updated to match.
    """
    def _r(v, nd):
        return None if v is None or not np.isfinite(v) else round(float(v), nd)

    confident = None
    if skew_result:
        confident = bool(skew_result.get("Prediction confident"))

    return {
        "Coverage fit r-squared": _r(s_free, 4),
        "Coverage fit p-value": _r(p_free, 5),
        "GC skew fit r-squared": _r(s_skew, 4),
        "GC skew fit p-value": _r(p_skew, 5),
        "GC skew prediction confident": confident,
        "Coverage vs skew likelihood ratio": _r(lam, 2),
        "Coverage vs skew likelihood-ratio p-value": _r(p_lr, 5),
        "Bootstrap surrogates": int(surrogates),
        "Breakpoint source": source,
        # Reported only -- nothing gates on either. The score says whether the
        # applied tent is the wrong SHAPE (systematically off over a stretch)
        # rather than merely a weak one; the decorrelation length is the floor it
        # is normalised against, published so the score is auditable from the
        # file alone. adp1_mgd06_lb's 45 kb against ltee_ara_p5_75k_exp's 20 kb is
        # why adp1's much larger raw excursion is not the more alarming of the
        # two. Roughly: < 1 unstructured, 1-2 mild, > 2 structured (about 6% of
        # the time under the null), > 3 strongly structured.
        "Residual structure score": _r(structure, 2),
        "Residual decorrelation length (bp)": (
            None if decorr_bp is None else int(decorr_bp)
        ),
    }


def _otr_skew_candidate(skew_result, df, series, phases, weights, m):
    """The GC-skew coordinates as an OTR candidate, or None.

    Four ways to get None, and they say different things:

      - there is no prediction, or the skew's own bootstrap called it low
        confidence;
      - the prediction was made against a different window count, so its indices
        do not address this frame (see below);
      - OTR_SKEW_MAX_P is set and the coverage does not support a tent here;
      - the coverage says the skew's ORIGIN is the low point.

    That last one is a REJECTION, not a relabelling. otr_fit() swaps its own
    breakpoints when they come out inverted, which is legitimate there because
    they are unlabelled and the antipodal seeds are blind to which is which.
    Here the labels ARE the imported prior: swapping them keeps the coordinates
    and discards the only thing the GC skew contributed. Measured, 1 of the 8
    authentic sequences contradicts (ltee_ara_m3_32k_2rg, ratio 0.907) and its
    skew tent explains r-squared 0.007 -- so what is rejected there is the sign
    of noise. On synthetic flat coverage the sign is a coin flip (49.3% over 300
    trials), which makes this a 50% filter on no-signal sequences rather than
    evidence about them.

    Returns a dict with the phase row, breakpoints, anchor values and r-squared.
    """
    if not skew_result or not skew_result.get("Prediction confident"):
        return None

    # The skew's indices are POSITIONAL (np.argmin over cum_gc_skew) while
    # otr_fit works in df.index. process_multi_genome() resets the index, so the
    # two agree today -- but the skew arm makes a silently wrong answer possible
    # where before there was only a reported one, so check rather than trust.
    # Declining is right: --bias otr on an oddly shaped frame still gets its own
    # free fit.
    if int(skew_result.get("Windows", -1)) != len(df):
        print("OTR: GC-skew prediction is for a different window count; ignoring it")
        return None

    n = len(df)
    o_win = int(skew_result["Origin window index"]) % n
    t_win = int(skew_result["Terminus window index"]) % n
    o_cell = (o_win * m) / n
    t_cell = (t_win * m) / n

    phase = _otr_normalize_phases(_otr_phase(m, o_cell, t_cell), weights)
    fitted, residual, r2 = _otr_tent_fit(series, phase[0], weights)

    # Anchor values at the skew's breakpoints, on the real windows rather than
    # the decimated ones -- these are what get reported and applied.
    x = df.index.to_numpy().astype(float)
    y = df["gc_corr_norm_cov"].to_numpy(dtype=float)
    keep = np.ones(n, dtype=bool)
    for column in ("is_deletion", "is_redundant"):
        if column in df.columns:
            keep &= ~df[column].to_numpy(dtype=bool)
    if keep.sum() < 4:
        keep = np.ones(n, dtype=bool)
    _, y_ori, y_ter = _otr_concentrated_rss(
        (float(o_win), float(t_win)), x[keep], y[keep], float(n)
    )

    if not (np.isfinite(y_ori) and np.isfinite(y_ter)) or y_ter <= 0 or y_ori < y_ter:
        return None

    return {
        "phase": phase, "residual": residual, "r2": r2,
        "o_win": o_win, "t_win": t_win, "y_ori": float(y_ori), "y_ter": float(y_ter),
    }


def otr_fit(df, n_seeds=8, skew_result=None, max_p=OTR_MAX_P,
            skew_max_p=OTR_SKEW_MAX_P, lr_alpha=OTR_LR_ALPHA,
            n_surrogates=DEFAULT_OTR_SURROGATES, block=None, seed=0):

    x = df.index.to_numpy().astype(float)
    y = df["gc_corr_norm_cov"].to_numpy(dtype=float)
    y_med_fil = df["gc_cor_med_fil"].to_numpy(dtype=float)
    n = len(x)
    genome_len = float(n)

    # Windows to exclude from both the seed search and the least-squares
    # fit: genuine deletions and redundant/repeat-coverage windows, if
    # mask_coverage_windows() has already flagged them, plus anything the
    # previous pass's HMM did not call CN = 1 (`is_cn_variant`, absent on the
    # first pass). All three follow the same column-presence convention, and
    # this function does not care which pass it is in -- the caller decides by
    # what it puts on the frame.
    exclude = np.zeros(n, dtype=bool)
    for column in ("is_deletion", "is_redundant", "is_cn_variant"):
        if column in df.columns:
            exclude |= df[column].to_numpy(dtype=bool)

    # Dilated by one window on each side, circularly, for the seed search
    # only: a window merely adjacent to a deletion/repeat boundary can
    # still drag a naive argmax/argmin off the true peak/trough even
    # though it doesn't cross the flagging threshold itself.
    exclude_for_search = exclude | np.roll(exclude, 1) | np.roll(exclude, -1)

    _yv = y[np.isfinite(y) & (y > 0)]
    _yref = np.median(_yv) if _yv.size else 1.0
    otr_floor = 0.1 * _yref if _yref > 0 else 1e-6

    if exclude_for_search.all():
        y_for_max = y_med_fil
        y_for_min = y_med_fil
    else:
        y_for_max = np.where(exclude_for_search, -np.inf, y_med_fil)
        y_for_min = np.where(exclude_for_search, np.inf, y_med_fil)
    o_idx_seed = int(np.nanargmax(y_for_max))
    t_idx_seed = int(np.nanargmin(y_for_min))

    # Fit using only unmasked windows -- deletions/repeats never inform
    # the breakpoint search or the anchor-value least-squares solve.
    fit_mask = ~exclude
    x_fit = x[fit_mask]
    y_fit_data = y[fit_mask]

    if fit_mask.sum() < 4:
        # Not enough clean data to fit anything meaningful.
        y_flat = np.repeat(np.mean(y), n)
        print("OTR bias not detected (insufficient clean windows)")
        return y, y_flat, o_idx_seed, t_idx_seed, False, _otr_detail()

    # ---- The statistic's exhaustive grid, computed FIRST ------------------
    #
    # It both supplies the observed statistic for the bootstrap and seeds the
    # refinement below, which is what keeps the tested tent and the applied
    # tent the same object. Getting this order wrong is what made them two
    # different searches over one objective.
    series, cell_w = _otr_decimate(y, fit_mask)
    m = series.size
    grid_ok = m >= OTR_MIN_CELLS
    if grid_ok:
        phases, grid_bps = _otr_phase_grid(m, cell_w)
        proj = (phases * cell_w) @ (series - float(cell_w @ series) / cell_w.sum())
        best_row = int(np.argmax(proj * proj))
        s_free = float(_otr_grid_scores(series, phases, cell_w)[0])
    else:
        phases, grid_bps, best_row = np.zeros((1, max(m, 1))), [(0.0, 0.0)], 0
        cell_w = np.ones(max(m, 1), dtype=float)
        s_free = 0.0

    # ---- Refine, CONSTRAINED to the same band the grid searched ----------
    #
    # Parametrised as (x_ori, separation) rather than (x_ori, x_ter) so the
    # band is a box constraint scipy can hold, instead of a filter applied to
    # the answer afterwards. That matters more than it looks: unconstrained,
    # this objective's global optimum is usually NOT a tent at all. As the
    # separation goes to zero the short arc vanishes and the regressor tends to
    # 1 - ((x - x_ori) mod L)/L -- a circular SAWTOOTH, a straight line across
    # the whole genome with one free discontinuity. Same two parameters, but a
    # strictly larger shape class: it fits any monotone drift plus one step,
    # so RSS is monotone non-increasing as the separation shrinks unless the
    # coverage really is V-shaped. Measured over 40 flat AR(1) nulls the free
    # optimum landed below 5% separation 20 times and inside the band 6; on
    # adp1_mgd06_lb it parks on a real 2.7x amplification edge and scores
    # r-squared 0.114 against 0.049 for the best genuine tent, and on
    # ltee_ara_m3_38k it comes to rest on the 1-window guard in
    # _otr_concentrated_rss, which is the only thing stopping the descent.
    #
    # The old code let the search find that and then rejected it by separation
    # afterwards -- so a degenerate optimum VETOED whatever real tent was
    # there, rather than deferring to it. A 6% coverage step is enough to
    # trigger it while the grid still reports p = 0.002.
    lo_sep, hi_sep = OTR_SEP_LO * genome_len, OTR_SEP_HI * genome_len

    def _rss_at(p):
        return _otr_concentrated_rss((p[0], p[0] + p[1]), x_fit, y_fit_data, genome_len)[0]

    # Seeds are (x_ori, separation). The grid argmax comes first -- it is a
    # global search of the band, so the refinement starts from the right basin
    # rather than hoping one of the spread seeds lands in it. The spread seeds
    # are offset by half a step so no coordinate is ever exactly 0: scipy
    # builds its initial simplex with a RELATIVE 5% perturbation except on an
    # exactly-zero coordinate, which gets an absolute 0.00025 -- measured, that
    # froze x_ori to a total excursion of 0.002 windows on two of the eight
    # authentic sequences, a one-dimensional search in disguise.
    #
    # They are also no longer paired with antipodes. Separation is now its own
    # coordinate, so seed k and seed k+4 used to be the same UNORDERED pair and
    # returned bit-identical results on every sequence -- four of the nine
    # starts bought nothing. The masked argmax/argmin seed is gone too: it
    # converged to a strictly worse minimum on five of eight sequences and
    # never uniquely won.
    seeds = []
    if grid_ok:
        go, gt = grid_bps[best_row]
        scale = genome_len / m
        seeds.append((go * scale, ((gt - go) % m) * scale))
    for k in range(n_seeds):
        seeds.append((((k + 0.5) / n_seeds) * genome_len, 0.5 * genome_len))

    best = None
    for x0 in seeds:
        x0 = (float(x0[0]), float(np.clip(x0[1], lo_sep, hi_sep)))
        res = minimize(
            _rss_at,
            x0=list(x0),
            method="Nelder-Mead",
            bounds=[(-genome_len, 2.0 * genome_len), (lo_sep, hi_sep)],
            options={"xatol": 0.5, "fatol": 1e-8, "maxiter": 2000},
        )
        cand = (res.x[0], res.x[0] + res.x[1])
        rss, y_ori_cand, y_ter_cand = _otr_concentrated_rss(
            cand, x_fit, y_fit_data, genome_len
        )
        if not np.isfinite(rss):
            continue
        if best is None or rss < best[0]:
            best = (rss, cand[0], cand[1], y_ori_cand, y_ter_cand)

    if best is None:
        y_flat = np.repeat(np.mean(y), n)
        print("OTR bias not detected (no seed converged)")
        return y, y_flat, o_idx_seed, t_idx_seed, False, _otr_detail()

    _, x_ori_opt, x_ter_opt, y_ori_opt, y_ter_opt = best

    # Orient the labels by the fitted anchor values: the origin is whichever
    # anchor came out higher. _otr_concentrated_rss() is symmetric under
    # swapping the two breakpoints -- the same tent, the same residuals, the
    # same RSS -- so which one comes back as x_ori is arbitrary, and everything
    # downstream reads y_ori as the PEAK. Note _otr_skew_candidate() does NOT do
    # this: its
    # labels come from the reference sequence, so an inversion there is evidence
    # against the prediction rather than a bookkeeping detail.
    if y_ori_opt < y_ter_opt:
        x_ori_opt, x_ter_opt = x_ter_opt, x_ori_opt
        y_ori_opt, y_ter_opt = y_ter_opt, y_ori_opt

    # Wrap fitted positions back into [0, genome_len). The optimizer can
    # legitimately return an out-of-range value (e.g. -1.6, mathematically
    # equivalent to genome_len - 1.6 under the circular model) that
    # otr_predict()/_otr_design_matrix() already handle via their own
    # internal modulo, but which MUST be wrapped before use as an array
    # index below (and by callers, e.g. apply_otr_correction()).
    x_ori_opt = x_ori_opt % genome_len
    x_ter_opt = x_ter_opt % genome_len
    o_idx = int(round(x_ori_opt)) % n
    t_idx = int(round(x_ter_opt)) % n

    # ---- Is this tent real, and whose breakpoints do we believe? -----------
    #
    # There used to be a `magnitude_ok` here comparing the anchor ratio against
    # a `bias_threshold` of 1.0. It was vacuous: the label swap above guarantees
    # y_ori >= y_ter, so it reduced to y_ter > 0, and nothing anywhere tested the
    # tent against a null. A flat, pure-noise genome still produces a best-fit
    # tent, and if its breakpoints happened to land 35-65% apart the coverage was
    # divided by noise. Both statistics below are computed unconditionally, even
    # when the answer is no, so a rejected fit stays diagnosable from the JSON --
    # the same choice predict_ori_ter_from_skew() makes.
    #
    # There is also no `separation_ok` any more, because there is nothing left
    # for it to catch: the refinement above is box-constrained to the band, so
    # every candidate it can return is in it by construction. It used to be a
    # sanity filter on the optimiser's output masquerading as a property of the
    # hypothesis -- and it was ANDed with a p-value computed from a DIFFERENT
    # tent (the grid's), so the thing being gated and the thing being tested
    # were not the same object. They are now.
    if grid_ok:
        _, resid_free, _ = _otr_tent_fit(series, phases[best_row], cell_w)
        block_free = _otr_block_length(m, _otr_autocorr_length(resid_free), block)
        p_free, surrogates = _otr_bootstrap_p(
            series, phases, cell_w, s_free, block_free, n_surrogates, seed
        )
    else:
        p_free, surrogates = 1.0, 0

    free_live = bool(p_free <= max_p)

    skew = _otr_skew_candidate(skew_result, df, series, phases, cell_w, m)
    p_skew, s_skew = None, None
    if skew is not None:
        s_skew = skew["r2"]
        block_skew = _otr_block_length(m, _otr_autocorr_length(skew["residual"]), block)
        p_skew, _ = _otr_bootstrap_p(
            series, skew["phase"], cell_w, s_skew, block_skew, n_surrogates, seed
        )
        if skew_max_p is not None and p_skew > skew_max_p:
            skew = None

    # The likelihood ratio only arbitrates between two candidates that have each
    # already earned their place -- it is not a substitute for the gates above.
    lam, p_lr = None, None
    if free_live and skew is not None:
        lam, s_skew, s_free_cmp = _otr_lr_statistic(series, skew["phase"], phases, cell_w)
        block_lr = _otr_block_length(m, _otr_autocorr_length(skew["residual"]), block)
        p_lr, _ = _otr_lr_bootstrap_p(
            series, skew["phase"][0], phases, cell_w, lam, block_lr, n_surrogates, seed
        )
        use_free = p_lr <= lr_alpha
    elif free_live:
        use_free = True
    elif skew is not None:
        use_free = False
    else:
        y_fit = np.repeat(np.mean(y), n)
        print(f"OTR bias not detected (coverage fit p={p_free:.4f}; "
              "no usable GC-skew prediction)")
        return y, y_fit, o_idx, t_idx, False, _otr_detail(
            s_free=s_free, p_free=p_free, s_skew=s_skew, p_skew=p_skew,
            surrogates=surrogates, skew_result=skew_result, source="not corrected",
        )

    if use_free:
        source = "coverage fit"
    else:
        source = "GC skew"
        x_ori_opt, x_ter_opt = float(skew["o_win"]), float(skew["t_win"])
        y_ori_opt, y_ter_opt = skew["y_ori"], skew["y_ter"]
        o_idx, t_idx = skew["o_win"], skew["t_win"]
        # No separation re-check: predict_ori_ter_from_skew() gates on the same
        # OTR_SEP_LO..OTR_SEP_HI band, so this candidate is in-band by construction.

    print(f"OTR bias detected, breakpoints from the {source}"
          + (f" (likelihood ratio p={p_lr:.4f})" if p_lr is not None else ""))

    # How structured is what the APPLIED tent failed to explain? Computed on the
    # winning candidate only -- a tent that was never applied has no residual
    # worth publishing -- and never on the not-detected paths, where the bare
    # _otr_detail() call already emits nulls. Reported, never gating.
    #
    # Evaluated over the FULL series, censored windows included: the evaluation
    # set must never shrink, or censoring would flatter the very score it is
    # judged by. Same reason the gate is frozen.
    structure, decorr_bp = None, None
    if grid_ok:
        struct_phase = _otr_normalize_phases(
            _otr_phase(m, x_ori_opt * m / genome_len, x_ter_opt * m / genome_len),
            cell_w)
        struct_rows = phases if use_free else struct_phase
        _, struct_resid, _ = _otr_tent_fit(series, struct_phase[0], cell_w)
        struct_tau = _otr_autocorr_length(struct_resid)
        structure, _ = _otr_residual_structure(
            struct_resid, struct_rows, cell_w,
            _otr_block_length(m, struct_tau, block), seed=seed,
        )
        # tau is in CELLS. One cell spans n/m windows, and one window advances
        # `step` bp, so the reported length is comparable across sequences and
        # across -w/-s -- which the cell count itself is not, since
        # OTR_SCORE_CELLS caps it.
        win_st = df["win_st"].to_numpy()
        step_bp = float(win_st[1] - win_st[0]) if n > 1 else float(
            df["win_end"].iloc[0] - win_st[0])
        decorr_bp = int(round(struct_tau * (genome_len / m) * step_bp))

    y_fit = otr_predict(x, x_ori_opt, x_ter_opt, y_ori_opt, y_ter_opt, genome_len)
    y_fit = np.clip(y_fit, otr_floor, None)
    y_corr = y / y_fit

    return y_corr, y_fit, o_idx, t_idx, True, _otr_detail(
        s_free=s_free, p_free=p_free, s_skew=s_skew, p_skew=p_skew,
        lam=lam, p_lr=p_lr, surrogates=surrogates,
        skew_result=skew_result, source=source,
        structure=structure, decorr_bp=decorr_bp,
    )


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# ---------------------------------------------------------------------------
# mask -> fit -> apply split for origin-to-terminus (OTR) bias. otr_fit()
# (above) does its own masking internally (via the is_deletion/is_redundant
# columns), so this only needs to ensure those columns exist before calling
# it, then split the former monolithic otr_correction() into a fit stage
# and an apply stage.
# ---------------------------------------------------------------------------
def fit_otr_bias(df, output, skew_result=None, **otr_kwargs):
    """
    Fit stage: runs the median-filter smoothing (if not already present),
    ensures deletion/redundant masking columns exist (running
    mask_coverage_windows() if they don't -- e.g. when OTR correction is
    run without a prior GC-correction pass), and runs otr_fit().

    `skew_result` is predict_ori_ter_from_skew()'s dict, when the caller has
    one. It is optional and defaults to None, which reproduces the old
    coverage-only behaviour exactly -- that is what keeps every existing call
    site working.

    Returns
    -------
    dict with keys "y_corr", "y_fit", "o_idx", "t_idx", "bias", "detail",
    "df_with_medfil". `y_corr` is otr_fit()'s own corrected-coverage array
    (== unchanged input when bias is not detected, == y/y_fit when it is)
    and must be used as-is by apply_otr_correction() rather than
    recomputed from y_fit, since y_fit is a flat line at mean(y) (not 1.0)
    in the no-bias case.
    """
    df = df.copy()

    if "is_deletion" not in df.columns or "is_redundant" not in df.columns:
        df = mask_coverage_windows(df)

    if "gc_cor_med_fil" not in df.columns:
        n = len(df)
        win = max(3, int(n / 50))
        win = min(win, n)
        if win % 2 == 0:
            win -= 1
        if win < 1:
            df["gc_cor_med_fil"] = df["gc_corr_norm_cov"].copy()
        else:
            df["gc_cor_med_fil"] = ndimage.median_filter(
                df["gc_corr_norm_cov"], size=win, mode="reflect"
            )

    y_corr, y_fit, o_idx, t_idx, bias, detail = otr_fit(
        df, skew_result=skew_result, **otr_kwargs
    )

    return {
        "y_corr": np.asarray(y_corr, dtype=float),
        "y_fit": np.asarray(y_fit, dtype=float),
        "o_idx": o_idx,
        "t_idx": t_idx,
        "bias": bias,
        "detail": detail,
        "df_with_medfil": df,
    }


def censored_median_coverage(df):
    """Median GC-corrected coverage over windows that are neither deletions nor repeats.

    GC-corrected rather than raw so a plasmid whose base composition differs from
    the chromosome does not have that read as copy number. NOT otr-corrected:
    OTR fires on some sequences and not others, so post-OTR values are not
    comparable within one sample.

    The mask is built from `is_deletion` / `is_redundant` rather than from
    `exclude_from_fit`, because fit_otr_bias() only guarantees the first two are
    present. On CWBI's plasmid_1 the censoring moves the estimate from 2.824 to
    2.946 -- 121 of its 232 windows carry redundant coverage.
    """
    values = df["gc_corr_norm_cov"].to_numpy(dtype=float)

    keep = np.ones(len(df), dtype=bool)
    for column in ("is_deletion", "is_redundant"):
        if column in df.columns:
            keep &= ~df[column].to_numpy(dtype=bool)
    if not keep.any():
        keep = np.ones(len(df), dtype=bool)

    values = values[keep & np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def relative_copy_numbers(per_genome):
    """{genome_id: copies relative to the LONGEST sequence}, which reads exactly 1.0.

    process_multi_genome() normalises every sequence against one pooled median, so
    a multi-copy plasmid arrives at a multiple of the chromosome. run_HMM then
    refits the single-copy level from whichever sequence it is handed, so that
    multiple is otherwise computed and thrown away -- CWBI's plasmids sit at 2.95x
    and 1.90x the chromosome and are both called copy number 1.

    Deliberately non-integral: 2.95 is a measurement, and rounding it to 3 would
    discard the precision that makes it worth reporting.

    Sequences are ranked by `win_end.max()`. The frame does not carry the true
    sequence length and preprocess() drops the trailing partial window, so that is
    3,354,501 against a real 3,354,690 -- only the ordering matters.
    """
    if not per_genome:
        return {}

    medians = {gid: censored_median_coverage(df) for gid, df in per_genome.items()}

    # Rank only sequences that HAVE windows. A zero-window frame gives
    # win_end.max() == NaN, and max() with a NaN key silently returns the first
    # key it was given rather than raising -- so a single empty table sorting
    # first would anchor the whole run on a NaN median and write
    # "Relative copy number": null into every OTHER sequence's JSON, including
    # perfectly healthy ones.
    ranked = [gid for gid in per_genome
              if len(per_genome[gid]) and np.isfinite(per_genome[gid]["win_end"].max())]
    if not ranked:
        return {gid: float("nan") for gid in per_genome}

    longest = max(ranked, key=lambda gid: float(per_genome[gid]["win_end"].max()))
    anchor = medians[longest]

    if not np.isfinite(anchor) or anchor <= 0:
        return {gid: float("nan") for gid in per_genome}
    return {gid: medians[gid] / anchor for gid in per_genome}


def write_otr_results(results, output, genome_id):
    """Write one sequence's OTR record to OTR_corr/<sample><seq_id>_otr_results.json.

    THE ONE FILE BRESEQ READS. It parses this with nlohmann/json, which is strict
    RFC JSON with no allow_nan, so a single bare NaN anywhere makes the whole file
    unparseable: breseq falls into `catch (...)`, warns, and reports no ori-ter
    bias at all. _json_safe() maps non-finite values to null on the way out, and
    allow_nan=False makes a value it missed fail HERE, loudly, in the suite --
    rather than silently costing breseq the file in the field. That is a
    deliberate new failure mode; a run that would have written unparseable JSON
    was already failing, just quietly and somewhere else.

    Sole writer of this file, so that every path producing one -- a fitted tent,
    a rejected fit, a bias mode that never fits, and a sequence with no coverage
    at all -- produces the same shape.
    """
    savedir = os.path.join(str(output), "OTR_corr")
    os.makedirs(savedir, exist_ok=True)

    samplename = sample_prefix(output) + str(genome_id)
    path = os.path.join(savedir, f"{samplename}_otr_results.json")
    with open(path, "w") as fh:
        json.dump({k: _json_safe(v) for k, v in results.items()}, fh,
                  indent=4, allow_nan=False)
    return path


def declined_otr_results(df, correction_type, reason=None,
                         relative_copy_number=float("nan")):
    """The OTR record for a sequence no tent was fitted to.

    Covers the cases apply_otr_correction() never sees: --bias gc and --bias
    none, which return before any OTR stage runs, and a sequence with no usable
    coverage, whose frame cannot be fitted at all. Without this those paths wrote
    NO json, and a missing file costs breseq exactly what an unparseable one does.

    The shape matches apply_otr_correction()'s rejected-fit branch, including the
    two constraints that reader imposes: "Origin-to-Terminus/Bias Ratio" is the
    string "Not detected", and "Origin window"/"Terminus window" are real ints
    and never null -- breseq does not type-check them. On a frame with no windows
    there is no coordinate to report and both are 0.
    """
    if len(df):
        xori = int(df["win_st"].iloc[0])
        xter = int(df["win_end"].iloc[len(df) - 1])
    else:
        xori = xter = 0

    results = {
        "Origin window": xori,
        "Origin coverage (normalized)": np.nan,
        "Terminus window": xter,
        "Terminus coverage (normalized)": np.nan,
        "Origin-to-Terminus/Bias Ratio": "Not detected",
        "Relative copy number": relative_copy_number,
        "Correction type": correction_type,
    }
    results.update(_otr_detail())
    # Added, never substituted: CLAUDE.md records that adding keys is the only
    # change that is safe for breseq's reader.
    results["No usable coverage reason"] = reason
    return results


def _json_safe(value):
    """NaN/inf -> None, so json.dump emits `null` rather than bare `NaN`.

    breseq reads this file with nlohmann/json, which is strict JSON and has no
    allow_nan: a single bare NaN makes the WHOLE file unparseable, so it falls
    into `catch (...)`, warns, and reports no ori-ter bias. That has been true of
    every "Not detected" file CNery has written, since yori/yter are NaN there.

    np.floating rather than float: np.float64 subclasses float and would be
    caught anyway, but np.float32 does not, and neither does np.float16. The
    values written here are float64 today; the point is that this is the guard
    standing between breseq and an unparseable file, so it should not depend on
    which numpy width a future statistic happens to come back as.
    """
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def apply_otr_correction(otr_fit_result, output, deletion_col="is_deletion",
                         relative_copy_number=1.0, extra_results=None):
    """
    Apply stage: evaluate the fitted OTR curve at every window, write
    plots/results JSON, and return (df, ori_win, ter_win) -- SAME
    signature as the original otr_correction().

    Only windows flagged `deletion_col` (genuine near-zero/outlier
    coverage) are left un-scaled at their GC-corrected value. Redundant-
    coverage windows (flagged via mask_coverage_windows(), but NOT flagged
    as deletions) DO receive the OTR scaling factor here, so they are not
    zeroed or frozen and won't be miscalled as deletions by the HMM.

    Uses otr_fit_result["y_corr"] (otr_fit()'s own corrected-coverage
    output) directly, instead of recomputing gc_corr_norm_cov / y_fit.
    When no bias is detected, otr_fit()'s y_fit is a flat line at mean(y)
    -- not 1.0 -- so recomputing the division would silently rescale
    every window by 1/mean(y) even when "no correction" is the intended
    behavior.
    """
    df = otr_fit_result["df_with_medfil"].copy()
    genome_id = str(df["genome_id"].iloc[0])
    # write_otr_results() makes this itself; kept here because plot_otr_corr()
    # writes into the same directory and does not.
    os.makedirs(str(output + "/OTR_corr/"), exist_ok=True)

    y_corr = otr_fit_result["y_corr"]
    f1 = otr_fit_result["y_fit"]
    o_idx, t_idx, bias = (
        otr_fit_result["o_idx"], otr_fit_result["t_idx"], otr_fit_result["bias"]
    )

    if bias:
        ori_idx, ter_idx = o_idx, t_idx
        xori = df["win_st"].iloc[ori_idx]
        xter = df["win_st"].iloc[ter_idx]
        yori = f1[ori_idx]
        yter = f1[ter_idx]
        # Plain yori/yter, so the file's own three numbers agree. It used to be
        # yori / (yter + 0.001), a divide-by-zero guard that put the reported
        # ratio ~0.1% below what the two coverage values printed beside it give
        # (1.06627 against 1.06733 on adp1) -- a reader checking the arithmetic
        # found it wrong. The guard is also unnecessary here: yter is an
        # anchor value from the least-squares fit and y_fit is clipped at
        # otr_floor = 0.1 * median coverage in otr_fit(), so it cannot be zero.
        # The explicit test is kept because "cannot" is doing work in that
        # sentence, and a null is honest where a fabricated ratio is not.
        OTR = float(yori / yter) if yter > 0 else None
    else:
        xori = df["win_st"].iloc[0]
        xter = df["win_end"].iloc[len(df) - 1]
        yori = np.nan
        yter = np.nan
        OTR = "Not detected"

    detail = otr_fit_result.get("detail") or {}

    # "Correction type" was already the field naming where the coordinates came
    # from, and GC_SKEW_METHOD is already the matching string -- so the arm that
    # won is reported by giving this key its other legal value, not by adding a
    # parallel one. It stays on the coverage string when nothing fired, since
    # nothing was used and changing it would move goldens for no information.
    correction_type = "Ori-ter coordinates fit by coverage"
    if bias and detail.get("Breakpoint source") == "GC skew":
        correction_type = GC_SKEW_METHOD
    # An all-zero sequence declines every fit and reaches here on the ordinary
    # path, where the default string would claim coordinates were fitted by
    # coverage there was none of. Say what actually happened.
    no_coverage = no_usable_coverage_reason(df)
    if no_coverage is not None:
        correction_type = "No usable coverage"

    results = {
        "Origin window": int(xori),
        "Origin coverage (normalized)": yori,
        "Terminus window": int(xter),
        "Terminus coverage (normalized)": yter,
        "Origin-to-Terminus/Bias Ratio": OTR,
        "Relative copy number": relative_copy_number,
        "Correction type": correction_type,
    }
    results.update(detail)

    # Whatever the caller wants recorded beside the evidence -- in practice the
    # first pass's verdict, so a verdict that CHANGED under the CN censor is
    # visible from the file alone. Added last so it cannot be shadowed, and safe
    # for breseq's nlohmann reader, which tolerates unknown keys.
    # Present on every OTR record, so a reader never has to tell "no coverage"
    # from "this CNery predates the key". Measured rather than passed in: an
    # all-zero sequence reaches this function on the ordinary path -- otr_fit()
    # declines it, run_HMM() calls it CN 0 -- so nothing upstream would otherwise
    # have said so in the file.
    results["No usable coverage reason"] = no_coverage

    if extra_results:
        results.update(extra_results)

    df["otr_gc_corr_norm_cov"] = df["gc_corr_norm_cov"].copy()

    if deletion_col in df.columns:
        low = df[deletion_col].to_numpy(dtype=bool)
    else:
        low = (df["read_count_cov"] <= df["read_count_cov"].median() * 0.1).to_numpy()

    # scale everything that's not a genuine deletion (redundant windows included),
    # using otr_fit()'s own y_corr rather than a fresh division by f1.
    df.loc[~low, "otr_gc_corr_norm_cov"] = y_corr[~low]
    df["otr_gc_corr_fact"] = f1

    write_otr_results(results, output, genome_id)

    return df, xori, xter


# ---------------------------------------------------------------------------
# Origin/terminus from cumulative GC skew.
#
# An independent estimate of the same two coordinates otr_fit() reads off the
# coverage profile, derived from the reference sequence instead of from read
# depth. Grigoriev 1998 (NAR 26:2286): the cumulative sum of (G-C)/(G+C)
# "reaches its global maximum at the E.coli terminus, while the minimum resides
# over the replication origin".
#
# otr_fit() CONSUMES these, as its second breakpoint candidate: see
# _otr_skew_candidate() and the arbitration at the end of otr_fit(). They are
# still reported in their own JSON and marked on their own plot, and under
# --bias gc/none nothing reads them.
#
# The two methods disagree about which sequences even have a usable origin, and
# that disagreement is the whole reason this is worth wiring in rather than a
# problem with it. otr_fit needs an active replication gradient in the COVERAGE,
# so its own fit clears significance on exponential-phase samples and not on
# stationary-phase ones; the skew reads the sequence and returns the same answer
# either way. Where the coverage has nothing, the skew can still say where to
# look -- and because the ramp's AMPLITUDE is then solved by least squares
# against the observed coverage, a sample with no gradient gets a near-flat tent
# rather than an imported one. Measured over 300 synthetic flat series, the
# largest ratio that produced was 1.089.
# ---------------------------------------------------------------------------
GC_SKEW_METHOD = "Ori-ter coordinates from cumulative GC skew (Grigoriev 1998)"

#: Circular block bootstrap defaults. 1000 surrogates costs ~0.14 s on a 4.6 Mb
#: genome against ~4.4 s for preprocess(), and resolves p down to 1/1001.
DEFAULT_SKEW_SURROGATES = 1000
#: Blocks the resample aims for. What actually governs the p-value is the NUMBER
#: of blocks, not their length: with fewer than ~20, reshuffling them frequently
#: reassembles a two-arm pattern by chance and the test loses all power. Measured
#: on a synthetic switch -- at 24 blocks p sits at its floor, at 12 it is 0.004,
#: at 6 it is 0.035, and at 3 it is 0.041 whatever the block length.
SKEW_TARGET_BLOCKS = 20
#: Bounds on block length in WINDOWS. At least 10 so a block still carries the
#: local compositional autocorrelation (which decays by lag ~10) rather than
#: shuffling it away; at most 200 because nothing is gained past that and long
#: blocks only cost blocks.
SKEW_MIN_BLOCK, SKEW_MAX_BLOCK = 10, 200


def _skew_block_length(n, block=None):
    """Block length for the circular bootstrap, in windows.

    Adaptive by default: aim for SKEW_TARGET_BLOCKS blocks, bounded so a block
    is never shorter than the local autocorrelation nor pointlessly long. Short
    sequences cannot satisfy both constraints at once -- that is a real
    statement about how little evidence they carry, not a defect to tune away.
    """
    if block is None:
        block = n // SKEW_TARGET_BLOCKS
        block = int(np.clip(block, SKEW_MIN_BLOCK, SKEW_MAX_BLOCK))
    return int(max(1, min(block, n // 2)))


def _replichore_t(skew, o_idx, t_idx, overlap):
    """Pooled two-sample t between the two arcs ori->ter and ter->ori.

    Overlapping windows are not independent observations, so the effective
    sample size is deflated by the win/step overlap factor -- otherwise t grows
    as sqrt(win/step) from nothing but a smaller --step-size, and the same
    genome scores differently at different resolutions.

    Returns (t, mean_a, mean_b). Note this is an effect SIZE, not a test
    statistic with a usable null: skew is spatially autocorrelated, so t's
    magnitude is inflated by an unknown factor and must not be read as a
    p-value. That is what _skew_bootstrap_p() is for.
    """
    n = skew.size
    idx = np.arange(n)
    arm_a = ((idx - o_idx) % n) <= ((t_idx - o_idx) % n)
    a, b = skew[arm_a], skew[~arm_a]
    if a.size < 2 or b.size < 2:
        return 0.0, 0.0, 0.0

    pooled_var = (
        (a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)
    ) / (n - 2)
    eff_a, eff_b = a.size / overlap, b.size / overlap
    denom = np.sqrt(pooled_var * (1 / eff_a + 1 / eff_b))
    t = 0.0 if denom <= 0 else float((a.mean() - b.mean()) / denom)
    return t, float(a.mean()), float(b.mean())


def _skew_score(skew, overlap):
    """|t| for the best-fitting ori/ter on this skew series.

    The WHOLE procedure -- locate the extrema, then score the arms -- because
    the breakpoints are chosen by looking at the data. A null that held them
    fixed would ignore that selection and be far too easy to beat.
    """
    cum = np.cumsum(skew - skew.mean())
    o_idx = int(np.argmin(cum))
    t_idx = int(np.argmax(cum))
    return abs(_replichore_t(skew, o_idx, t_idx, overlap)[0])


def _skew_bootstrap_p(skew, observed_t, overlap, n_surrogates, block, seed):
    """Circular block bootstrap p-value for the replichore split.

    The null is "a sequence with this much LOCAL skew autocorrelation but no
    single origin/terminus". Resampling whole blocks preserves the short-range
    structure; reshuffling their order destroys the long-range two-arm pattern.
    Circular because the genome is, so there are no edge effects to correct.

    Returns (p, surrogates_used). p is (#{surrogate >= observed} + 1) / (B + 1),
    so it is FLOORED at 1/(B+1) and never zero -- a real chromosome exhausts
    every surrogate and reads back exactly that floor. Report it as an upper
    bound, not as a measurement.

    `seed` is fixed by default so a given input always gives the same p and the
    golden files stay stable.
    """
    n = skew.size
    if n < 8:
        return 1.0, 0
    block = _skew_block_length(n, block)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)

    at_least = 0
    # Chunked so the index matrix stays small regardless of n_surrogates.
    chunk = max(1, min(250, n_surrogates))
    done = 0
    while done < n_surrogates:
        size = min(chunk, n_surrogates - done)
        starts = rng.integers(0, n, size=(size, n_blocks))
        idx = (starts[:, :, None] + offsets[None, None, :])
        idx = idx.reshape(size, n_blocks * block)[:, :n] % n
        for row in idx:
            if _skew_score(skew[row], overlap) >= observed_t:
                at_least += 1
        done += size

    return (at_least + 1) / (n_surrogates + 1), n_surrogates


def predict_ori_ter_from_skew(
    df,
    win=None,
    step=None,
    sep_lo=OTR_SEP_LO,
    sep_hi=OTR_SEP_HI,
    max_p=0.01,
    n_surrogates=DEFAULT_SKEW_SURROGATES,
    block=None,
    seed=0,
):
    """
    Locate the replication origin and terminus on the cumulative GC skew curve:
    origin = its global minimum, terminus = its global maximum.

    Deliberately does NOT censor is_deletion / is_redundant windows, unlike
    fit_gc_bias(), fit_otr_bias() and run_HMM(). GC skew is a property of the
    REFERENCE SEQUENCE -- a deletion in the sample does not change the
    reference's base composition -- and because the curve is a running sum,
    dropping windows would not merely omit them, it would displace every point
    after them. Please don't "fix" this.

    Confidence is two conditions: the extrema must be roughly antipodal
    (`sep_lo`..`sep_hi`, the same band otr_fit uses) and the replichore split
    must survive a circular block bootstrap at `max_p`. There is deliberately
    NO minimum window count -- the bootstrap subsumes it, since a short
    sequence cannot reach significance on its own, and one arbitrary constant
    is better than two.

    The returned values are always the measured ones. The gate sets
    `confident`; it never suppresses a coordinate, so a rejected prediction
    stays diagnosable.
    """
    n = len(df)
    if n == 0:
        raise ValueError(
            "cannot predict origin/terminus from an empty window frame."
        )

    skew = df["gc_skew"].to_numpy(dtype=float)
    cum = df["cum_gc_skew"].to_numpy(dtype=float)
    win_st = df["win_st"].to_numpy()

    o_idx = int(np.argmin(cum))
    t_idx = int(np.argmax(cum))

    circular_dist = min(abs(o_idx - t_idx), n - abs(o_idx - t_idx))
    separation = circular_dist / n if n else 0.0
    amplitude = float(cum.max() - cum.min())

    # Split the circle into the two replichores at the origin and terminus and
    # ask whether their mean skew really differs in sign.
    overlap = (win / step) if (win and step) else 1.0
    t_stat, mean_a, mean_b = _replichore_t(skew, o_idx, t_idx, overlap)
    opposite_signs = bool(mean_a * mean_b < 0)

    p_value, surrogates = _skew_bootstrap_p(
        skew, abs(t_stat), overlap, n_surrogates, block, seed
    )

    confident = bool(
        sep_lo <= separation <= sep_hi
        and opposite_signs
        and p_value <= max_p
    )

    return {
        "Origin (bp)": int(win_st[o_idx]),
        "Terminus (bp)": int(win_st[t_idx]),
        "Origin window index": o_idx,
        "Terminus window index": t_idx,
        "Windows": n,
        "Separation (fraction of genome)": round(separation, 4),
        "Cumulative skew amplitude": round(amplitude, 4),
        "Replichore skew t-statistic": round(t_stat, 2),
        "Replichore skew p-value": round(p_value, 5),
        # Reported so the p-value's floor of 1/(B+1) is legible from the file
        # alone: a chromosome exhausts every surrogate and reads back exactly
        # that floor, which is an upper bound rather than a measurement.
        "Bootstrap surrogates": surrogates,
        "Prediction confident": confident,
        "Prediction method": GC_SKEW_METHOD,
    }


def plot_no_data(output, subdir, filename, title, message):
    """A one-page PDF saying there was nothing to draw.

    Emitted for a sequence with no windows, in place of each figure the ordinary
    path would have produced. Every plotting function here starts by indexing row
    0 of the frame, so none of them survives an empty one -- and a MISSING plot
    is the worst of the three outcomes: it is indistinguishable from a run that
    died partway, which is exactly the confusion this whole code path exists to
    remove. A page that says "no usable coverage" is a result.
    """
    savedir = os.path.join(str(output), subdir)
    os.makedirs(savedir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, wrap=True)

    path = os.path.join(savedir, filename)
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    return path


def no_skew_prediction():
    """The GC-skew record for a reference with no windows to measure skew over.

    predict_ori_ter_from_skew() raises on an empty frame, deliberately -- a
    caller reaching it with no windows has a bug. A coverage table with no
    position rows is not that case: there is genuinely nothing to predict from,
    and the file still has to exist, because a reader that finds one sequence's
    GC_skew JSON missing cannot tell "no data" from "the run died here".

    Every value is finite: write_gc_skew_results() dumps with allow_nan=False,
    and plot_gc_skew() formats the separation as a percentage and the p-value
    with :.3g, neither of which tolerates None.
    """
    return {
        "Origin (bp)": 0,
        "Terminus (bp)": 0,
        "Origin window index": 0,
        "Terminus window index": 0,
        "Windows": 0,
        "Separation (fraction of genome)": 0.0,
        "Cumulative skew amplitude": 0.0,
        "Replichore skew t-statistic": 0.0,
        "Replichore skew p-value": 1.0,
        "Bootstrap surrogates": 0,
        "Prediction confident": False,
        "Prediction method": GC_SKEW_METHOD,
    }


def write_gc_skew_results(result, output, genome_id):
    """Write one reference's skew prediction to GC_skew/<name>_gc_skew_results.json.

    Makes its own directory, as apply_otr_correction() does, so callers and
    tests need not pre-create it.
    """
    samplename = sample_prefix(output) + str(genome_id)
    savedir = os.path.join(output, "GC_skew")
    os.makedirs(savedir, exist_ok=True)

    path = os.path.join(savedir, f"{samplename}_gc_skew_results.json")
    with open(path, "w") as fh:
        # allow_nan=False: the OTR writer emits bare NaN, which json.load
        # tolerates but is not valid RFC JSON. Every value here is finite, so
        # this stays strict -- and fails loudly if that ever stops being true.
        json.dump(result, fh, indent=4, allow_nan=False)
    return path


# ---------------------------------------------------------------------------
# Per-sequence correction diagnostic: a before/after for every fitting step, and
# what each of those fits was allowed to see.
#
# The existing plot_otr_corr() overlays all three coverage stages on one axis,
# which buries the "before": raw is drawn first and the GC and OTR clouds cover
# it. One row per CHANGE, with its own censoring strip, keeps each comparison
# visible and makes the censoring legible where it actually applies.
# ---------------------------------------------------------------------------

#: Rows drawn per --bias mode, as (before, after, label, factor). The `otr` mode
#: is why this is a table rather than "whatever columns are present":
#: get_CNV.main() aliases gc_corr_norm_cov = norm_raw_cov there, so a GC row
#: would plot a bit-identical copy of the raw series under a label that is a lie.
# (before column, after column, label, factor column, pass number).
#
# A table rather than "whatever columns are present" because --bias otr aliases
# gc_corr_norm_cov = norm_raw_cov (get_CNV.py), so a GC row there would plot a
# bit-identical copy of the raw series under a label that is false.
#
# The pass number picks the censoring strip: pass 2's fits additionally exclude
# every window the pass-1 HMM did not call CN = 1.
#
# Rows 1-3 chain: each "after" column IS the next row's "before" column. Row 4
# does not, and that is the one thing about this table worth staring at. Pass 2's
# GC row draws its own before/after -- raw/(g1*t) -> raw/(g1*t*g2), the residual
# that fit saw and flattened. Pass 2's OTR row starts from raw/G instead, with the
# pass-1 tent divided back OUT, because a tent must be fitted to a series that
# still contains the ramp; fitting the residual would return a flat tent. So the
# break is real, it is between the two pass-2 rows rather than at the pass
# boundary, and the figure marks it by comparing column names -- never by
# comparing pass numbers, which get this wrong.
OTR_STAGE_ROWS = {
    "all": (("norm_raw_cov", "gc_corr_norm_cov_pass1",
             "GC correction (pass 1)", "gc_corr_fact_pass1", 1),
            ("gc_corr_norm_cov_pass1", "otr_gc_corr_norm_cov_pass1",
             "OTR correction (pass 1)", "otr_gc_corr_fact_pass1", 1),
            ("otr_gc_corr_norm_cov_pass1", "gc2_resid_cov",
             "GC correction (pass 2)", "gc_corr_fact_pass2", 2),
            ("gc_corr_norm_cov", "otr_gc_corr_norm_cov",
             "OTR correction (pass 2)", "otr_gc_corr_fact", 2)),
    "gc": (("norm_raw_cov", "gc_corr_norm_cov_pass1",
            "GC correction (pass 1)", "gc_corr_fact_pass1", 1),
           ("otr_gc_corr_norm_cov_pass1", "gc2_resid_cov",
            "GC correction (pass 2)", "gc_corr_fact_pass2", 2)),
    "otr": (("norm_raw_cov", "otr_gc_corr_norm_cov_pass1",
             "OTR correction (pass 1)", "otr_gc_corr_fact_pass1", 1),
            ("norm_raw_cov", "otr_gc_corr_norm_cov",
             "OTR correction (pass 2)", "otr_gc_corr_fact", 2)),
    "none": (),
}

#: Bins for the repeat-density lane. Repeats come in hundreds of 2-4 window
#: fragments (measured: 79-332 runs per sequence, median 2-4 windows), so drawing
#: them as spans on a 9,258-window genome gives ~0.001 inch each -- invisible
#: stippling that reads as "nothing censored". Deletions are the opposite, a
#: handful of 12-55 window blocks, and get true spans on their own lane: folding
#: them into this density would put a 12-window deletion in a 23-window bin at
#: 0.5 and understate it.
OTR_CENSOR_BINS = 400


def _stage_rows(bias):
    """The before/after rows to draw for this --bias mode."""
    return OTR_STAGE_ROWS.get(bias, OTR_STAGE_ROWS["all"])


def _in_band_fractions(values, scale, censored):
    """(all windows, uncensored only) fraction within 20% of single copy.

    Scaled by the sequence's own censored median first. Without that the measure
    reads 0.000 for every stage on a multi-copy replicon -- norm_raw_cov is
    normalised against the POOLED median, so CWBI's plasmids sit at 2.95x and
    1.90x and never enter the band at all, which would draw as "the pipeline did
    nothing" on the one sequence where it plainly did something. The scaling is a
    no-op where it should be: measured 0.981-1.001 on all five chromosomes.

    Both denominators are reported because the LEVEL differs even where the
    change does not. On CWBI's plasmid_1 the uncensored figure reads a flat 1.000
    against an honest 0.819, since 52% of its windows are repeats sitting at
    0.6-0.75 -- quoting only the uncensored number would claim perfection on a
    sequence half of which is off-band.
    """
    v = np.asarray(values, dtype=float)
    if scale and np.isfinite(scale) and scale > 0:
        v = v / scale
    in_band = (v > 0.8) & (v < 1.2)
    keep = ~np.asarray(censored, dtype=bool)
    return (float(in_band.mean()),
            float(in_band[keep].mean()) if keep.any() else float("nan"))


def _apply_tent(df, y_fit, base):
    """Coverage after dividing by `y_fit`, using apply_otr_correction()'s rule.

    Deletion windows keep their uncorrected value -- apply_otr_correction()
    leaves them alone so a real deletion is not divided back up toward 1 -- so
    reproducing that here is what makes an intermediate stage comparable to the
    column the pipeline actually wrote.
    """
    out = np.asarray(base, dtype=float).copy()
    keep = ~df["is_deletion"].to_numpy(dtype=bool) if "is_deletion" in df else np.ones(out.size, bool)
    y = np.asarray(y_fit, dtype=float)
    out[keep] = np.asarray(base, dtype=float)[keep] / y[keep]
    return out


def _correction_chain(df, otr_result, bias):
    """The corrections in the order they actually happen.

    Returns [(label, before, after, pass_no, chains_from_previous)]. The last
    element is computed from the COLUMN NAMES -- this row's "before" being the
    previous row's "after" -- so a row that does not continue the chain is marked
    as such wherever it happens to fall. See OTR_STAGE_ROWS for why one row does
    not.
    """
    steps, prev_after = [], None
    for before_col, after_col, label, _factor, pass_no in _stage_rows(bias):
        if before_col not in df or after_col not in df:
            continue
        steps.append([label, df[before_col].to_numpy(dtype=float),
                      df[after_col].to_numpy(dtype=float), pass_no,
                      prev_after is None or before_col == prev_after])
        prev_after = after_col
    return steps


def _censor_bins(mask, n_bins=OTR_CENSOR_BINS):
    """Fraction of windows censored per positional bin.

    Degenerates to one bin per window on a short sequence, so the plasmids get
    exact spans with no special case.
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    bins = int(min(n_bins, n))
    edges = (np.arange(n) * bins) // n
    total = np.bincount(edges, minlength=bins).astype(float)
    hit = np.bincount(edges, weights=mask.astype(float), minlength=bins)
    total[total == 0] = 1.0
    return np.arange(bins) * (n / bins), hit / total


def _spans(mask):
    """Contiguous True runs as (start, stop) index pairs."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.view(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def plot_correction_stages(df, output, otr_result=None, bias="all"):
    """Before/after for each fitting step, and what each fit was allowed to see.

    Makes its own directory, as plot_gc_skew() does, and returns the path.

    `otr_result` is fit_otr_bias()'s dict, carrying the OTR verdict and its
    evidence. None under --bias gc/none, where no OTR stage runs at all.

    `bias` is required rather than inferred: see OTR_STAGE_ROWS.
    """
    genome_id = str(df["genome_id"].iloc[0])
    samplename = sample_prefix(output) + genome_id
    savedir = os.path.join(output, "corr_plots")
    os.makedirs(savedir, exist_ok=True)

    n = len(df)
    mb = df["win_st"].to_numpy(dtype=float) / 1e6
    scale = censored_median_coverage(df)
    deletion = df["is_deletion"].to_numpy(dtype=bool) if "is_deletion" in df else np.zeros(n, bool)
    redundant = df["is_redundant"].to_numpy(dtype=bool) if "is_redundant" in df else np.zeros(n, bool)
    cn_variant = (df["is_cn_variant"].to_numpy(dtype=bool)
                  if "is_cn_variant" in df else np.zeros(n, bool))
    # What each pass's fits could see. Pass 2 additionally excludes every window
    # the pass-1 HMM did not call CN = 1 -- the whole reason the second pass
    # exists, so the strips are where it has to be legible.
    censor_by_pass = {1: deletion | redundant, 2: deletion | redundant | cn_variant}
    base_censor = censor_by_pass[1]

    detail = (otr_result or {}).get("detail") or {}

    steps = _correction_chain(df, otr_result, bias)
    n_rows = len(steps)

    # Origin and terminus of the applied fit, in Mb, for the rows that have one.
    ori_mb = ter_mb = None
    if otr_result is not None and otr_result.get("bias"):
        o_i, t_i = int(otr_result["o_idx"]) % n, int(otr_result["t_idx"]) % n
        ori_mb, ter_mb = float(mb[o_i]), float(mb[t_i])

    # Each correction row gets a tall coverage panel and a thin censoring strip
    # directly beneath it, so "what changed" and "what this fit could see" share
    # an x-axis. The strips genuinely differ between the passes, which is the
    # point: pass 1 has only near-zero depth and repeat overlap to go on, and
    # pass 2 adds everything the pass-1 HMM called CN != 1.
    heights = []
    for _ in range(max(n_rows, 1)):
        heights += [3.0, 0.55]
    fig = plt.figure(figsize=(11, 1.4 + 2.6 * max(n_rows, 1)))
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.16)

    def censor_strip(ax, pass_no):
        """The reasons this fit could not use a window, drawn where they apply.

        Three different things share this axis, so it carries a colour key: two
        of them are bare axvspans with nothing else to name them, and without it
        a reader has no way to tell a deletion from a copy-number event.
        """
        censored = censor_by_pass[pass_no]
        x, dens = _censor_bins(redundant)
        ax.bar(x * (mb[-1] - mb[0]) / max(n - 1, 1) + mb[0],
               dens, width=(mb[-1] - mb[0]) / max(len(dens), 1),
               color="tab:purple", alpha=0.75, linewidth=0)

        # CN != 1 stretches are drawn first and widest: they are the new
        # information on this figure, and an amplification is a handful of long
        # blocks, so spans read correctly where the repeat density lane would not.
        # `& ~deletion` so a called deletion is not drawn twice -- it is already
        # CN 0, and red is the more specific statement.
        cn_only = cn_variant & ~deletion
        drew_cn = False
        if pass_no == 2:
            for a, b in _spans(cn_only):
                ax.axvspan(mb[a], mb[min(b, n - 1)], color="tab:green", alpha=0.55,
                           linewidth=0)
                drew_cn = True
        drew_del = False
        for a, b in _spans(deletion):
            ax.axvspan(mb[a], mb[min(b, n - 1)], color="tab:red", alpha=0.85,
                       linewidth=0)
            drew_del = True

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_ylabel("censored", fontsize=7)
        ax.tick_params(labelsize=7)

        # Only reasons actually DRAWN are listed. A sequence whose CN censor was
        # declined (CN_CENSOR_MIN_KEEP -- CWBI's plasmid_1, 52% repeats) saw
        # exactly what pass 1 saw, and naming "CN != 1" there would describe a
        # restriction the reader can see is not present.
        handles = []
        if drew_del:
            handles.append(mpatches.Patch(
                color="tab:red", alpha=0.85,
                label=f"deletion ({int(deletion.sum())} win)"))
        handles.append(mpatches.Patch(
            color="tab:purple", alpha=0.75,
            label=f"repeat, fraction per bin ({100 * redundant.mean():.1f}%)"))
        if drew_cn:
            handles.append(mpatches.Patch(
                color="tab:green", alpha=0.55,
                label=f"CN != 1 ({int(cn_only.sum())} win)"))
        ax.legend(handles=handles, loc="upper right", fontsize=6,
                  ncol=len(handles), framealpha=0.85, handlelength=1.2,
                  handleheight=0.7, borderpad=0.25, columnspacing=0.9)
        ax.text(0.005, 0.90,
                f"censored from this fit: {100 * censored.mean():.1f}% of windows",
                transform=ax.transAxes, fontsize=7, va="top", color="0.25")

    def banner(ax, text):
        """Row label inside the axes -- set_title() collides with the strip above."""
        ax.text(0.006, 0.965, text, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left", zorder=5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="0.8", lw=0.5))

    def band(ax, series):
        """Grey 0.8-1.2 band, and a y-range driven by the data.

        The floor stays at 0 because apply_gc_correction() freezes deletion
        windows there and that is real information, but the ceiling comes from a
        high percentile so a single amplification does not leave most of the
        panel empty. Clipped points are counted rather than silently dropped.
        """
        ax.axhspan(0.8, 1.2, color="0.85", alpha=0.6, linewidth=0, zorder=0)
        finite = np.concatenate([s[np.isfinite(s)] for s in series]) if series else np.array([1.0])
        top = max(1.6, float(np.percentile(finite, 99.0)) * 1.10) if finite.size else 1.6
        clipped = int((finite > top).sum())
        ax.set_ylim(0, top)
        if clipped:
            # Below the banner, which spans the top-left and can run long.
            ax.text(0.994, 0.86, f"{clipped} window(s) above {top:.1f}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="right",
                    color="0.35")

    axes = []
    for i in range(max(n_rows, 1)):
        ax = fig.add_subplot(gs[2 * i, 0], sharex=axes[0] if axes else None)
        axc = fig.add_subplot(gs[2 * i + 1, 0], sharex=ax)
        axes.append(ax)

        if i < len(steps):
            label, before_v, after_v, pass_no, chains = steps[i]
            row_censor = censor_by_pass[pass_no]
            b = before_v / (scale or 1.0)
            a = after_v / (scale or 1.0)
            band(ax, [b[~row_censor], a[~row_censor]])
            # REPEAT WINDOWS ARE NOT DRAWN AT ALL. Their depth measures how many
            # reference copies collapsed onto the locus rather than anything
            # about this sample -- CWBI's chromosome reaches 18x on exactly zero
            # unique coverage -- and no stage of the pipeline ever reads them.
            # The repeat-density lane in the strip below is where they are
            # accounted for, and it is the only place they belong.
            #
            # Deletions and CN-variant windows ARE drawn, faintly. They are
            # censored from this fit too, but they are real measurements of real
            # events, so hiding them would hide the amplifications the second
            # pass exists to exclude. Faint keeps them clearly outside the
            # before/after comparison.
            #
            # "after" is drawn before "before" so "before" lands on top: where a
            # step changed nothing the row reads grey, which is the honest
            # picture; where it changed something the two clouds separate and
            # both are visible whatever the order.
            fit_seen = ~row_censor
            shown_censored = (~fit_seen) & (~redundant)
            if shown_censored.any():
                ax.scatter(mb[shown_censored], a[shown_censored], s=2.5, alpha=0.18,
                           linewidths=0, rasterized=True, color="tab:orange",
                           label="deletion / CN != 1 (not used in this fit)")
            ax.scatter(mb[fit_seen], a[fit_seen], s=2.5, alpha=0.30, linewidths=0,
                       rasterized=True, color="tab:blue", label="after")
            ax.scatter(mb[fit_seen], b[fit_seen], s=2.5, alpha=0.30, linewidths=0,
                       rasterized=True, color="0.45", label="before")
            b_all, b_un = _in_band_fractions(before_v, scale, row_censor)
            a_all, a_un = _in_band_fractions(after_v, scale, row_censor)
            banner(ax, f"{i + 1}. {label}   within 20% of single copy: "
                       f"{b_all:.1%} \u2192 {a_all:.1%} all windows   |   "
                       f"{b_un:.1%} \u2192 {a_un:.1%} uncensored")
            # This row does not continue the one above: the pass-1 tent has been
            # divided back out, because a tent has to be fitted to a series that
            # still contains the ramp. Said on the figure rather than left for the
            # reader to notice the discrepancy and distrust the whole thing.
            if i and not chains:
                ax.text(0.5, 1.015, "\u2500\u2500 pass-1 tent divided back out; "
                                    "refitted on CN=1 windows \u2500\u2500",
                        transform=ax.transAxes, fontsize=7.5, ha="center",
                        va="bottom", color="0.35")
            censor_strip(axc, pass_no=pass_no)

        else:
            # --bias none: nothing was corrected, so there is no before/after.
            # The censoring is still worth drawing -- it is what any later run
            # WOULD exclude -- so the strip below is populated as usual.
            shown = ~redundant
            raw = df["norm_raw_cov"].to_numpy(dtype=float) / (scale or 1.0)
            band(ax, [raw[shown]])
            ax.scatter(mb[shown], raw[shown], s=2.5, alpha=0.30, linewidths=0,
                       rasterized=True, color="0.55", label="raw coverage")
            r_all, r_un = _in_band_fractions(df["norm_raw_cov"].to_numpy(dtype=float),
                                             scale, base_censor)
            banner(ax, f"no correction applied   within 20% of single copy: "
                       f"{r_all:.1%} all windows   |   {r_un:.1%} uncensored")
            censor_strip(axc, pass_no=1)

        ax.set_ylabel("cov / median", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9, markerscale=3.5)
        plt.setp(ax.get_xticklabels(), visible=False)
        if i < max(n_rows, 1) - 1:
            plt.setp(axc.get_xticklabels(), visible=False)

    fig.axes[-1].set_xlabel("genomic position (Mb)", fontsize=9)
    src = detail.get("Breakpoint source", "not corrected")
    fig.suptitle(
        f"{samplename}  --  {n:,} windows; "
        f"{100 * base_censor.mean():.1f}% censored before fitting "
        f"({int(redundant.sum()):,} repeat window(s) not drawn); "
        f"--bias {bias}; OTR {src}",
        fontsize=10)

    path = os.path.join(savedir, "%s_correction_stages.pdf" % samplename.replace(" ", "_"))
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gc_skew(df, output, result):
    """Cumulative GC skew across the reference, with the predicted ori/ter marked."""
    genome_id = str(df["genome_id"].iloc[0])
    samplename = sample_prefix(output) + genome_id
    savedir = os.path.join(output, "GC_skew")
    os.makedirs(savedir, exist_ok=True)

    ori = result["Origin (bp)"]
    ter = result["Terminus (bp)"]
    confident = result["Prediction confident"]

    plt.figure(figsize=(10, 8))
    plt.plot(df["win_st"], df["cum_gc_skew"], color="purple",
             label="Cumulative GC skew")
    plt.axhline(0, color="gray", linewidth=0.8)

    # A rejected prediction is far easier to diagnose as a picture than as a
    # boolean, so the extrema are always drawn -- the title carries the verdict.
    plt.axvline(x=ori, color="r", linestyle=":", label=f"Origin: {ori}")
    plt.axvline(x=ter, color="b", linestyle=":", label=f"Terminus: {ter}")
    plt.scatter(
        [ori, ter],
        [df["cum_gc_skew"].min(), df["cum_gc_skew"].max()],
        color=["r", "b"], s=60, zorder=3,
    )

    verdict = "confident" if confident else "LOW CONFIDENCE"

    # p is floored at 1/(B+1), so a chromosome that beat every surrogate is shown
    # as "<floor" rather than as an exact figure it cannot support.
    surrogates = result.get("Bootstrap surrogates") or 0
    p_value = result["Replichore skew p-value"]
    if surrogates and p_value <= (1 / (surrogates + 1)) * 1.01:
        # Shown as 1/B rather than the exact 1/(B+1) floor: still a true bound,
        # and it reads as "p<0.001" instead of "p<0.000999".
        p_text = f"p<{1 / surrogates:.3g}"
    else:
        p_text = f"p={p_value:.3g}"

    plt.xlabel("Window (Genomic position)")
    plt.ylabel("Cumulative GC skew  (running sum of (G-C)/(G+C))")
    plt.title(
        f"{samplename}_Cumulative GC skew\n"
        f"separation {result['Separation (fraction of genome)']:.1%} of genome, "
        f"t={result['Replichore skew t-statistic']}, {p_text} -- {verdict}"
    )
    plt.legend(loc="upper right")

    plt_full_path = os.path.join(
        savedir, "%s_GC_skew.pdf" % samplename.replace(" ", "_")
    )
    plt.savefig(plt_full_path, format="pdf", bbox_inches="tight")
    plt.close()
    return plt_full_path


def _cnv_axis_ticks(candidate_cn_ticks, lo1, hi1, lo2, hi2):
    """(copy-number ticks, read-count ticks) at the SAME heights.

    The two axes of the CNV plot are one scale in different units, so they have
    to be labelled at the same physical positions -- otherwise a reader taking a
    coverage off the right axis for a state marked on the left gets a number the
    plot does not mean.
    """
    scale = ((hi2 - lo2) / (hi1 - lo1)) if hi1 > lo1 else 1.0
    cn = [float(t) for t in candidate_cn_ticks if lo1 <= t <= hi1]
    if len(cn) < 2:
        cn = [float(lo1), float(hi1)]
    return cn, [lo2 + (t - lo1) * scale for t in cn]


def _cnv_axis_limits(df_cnv, drawn, delta):
    """Limits for plot_copy's twinned axes: ((CN lo, hi), (read count lo, hi)).

    ONE SCALE, TWO LABELS. The read-count axis is the copy-number axis multiplied
    by the median read depth, because that is exactly how otr_gc_corr_rdcnt_cov
    was formed (run_HMM: corrected coverage * that median). Deriving the second
    from the first is what makes a copy number of k land on the coverage a copy
    number of k actually produces.

    The two used to be set INDEPENDENTLY -- one from corrected coverage, the
    other from raw counts, each with its own padding -- so nothing tied them
    together and any alignment was accidental. Measured, it was wrong everywhere:
    the CN-1 line was drawn 56% too high on ltee_ara_m3_32k_2rg and 92% too high
    on cwbi_ssym_ht04's chromosome, and the coverage those calls describe read as
    CN 0.23-0.46 instead of 1.00. The line did not pass through the data it was
    labelling.

    `scale` comes from EVERY window rather than the drawn ones, because that is
    the median run_HMM used when it built otr_gc_corr_rdcnt_cov.
    """
    scale = float(df_cnv["read_count_cov"].median())
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    shown = np.concatenate([
        drawn["read_count_cov"].to_numpy(dtype=float),
        drawn["otr_gc_corr_rdcnt_cov"].to_numpy(dtype=float),
    ]) if len(drawn) else np.array([0.0])
    shown = shown[np.isfinite(shown)]
    if not shown.size:
        shown = np.array([0.0])

    # FLOORED AT ZERO, both of them. A read count cannot be negative and neither
    # can a copy number, so padding the bottom the way the top is padded put a
    # "-85 reads" tick on the axis and invited the reader to believe the scale
    # meant something there. Windows at zero coverage -- real deletions, and the
    # CN-0 calls that describe them -- sit on the bottom edge, which is where
    # zero belongs.
    lo2 = 0.0
    hi2 = float(shown.max()) + delta
    return (lo2, hi2 / scale), (lo2, hi2)


def plot_copy(df_cnv, pltstart, pltend, output):
    
    genome_id = str(df_cnv["genome_id"].iloc[0])
    samplename = sample_prefix(output) + genome_id
    # samplename = sample.strip().split('.')[0]
    saveplt = str(output+"/CNV_plt/")
    
    win_st = df_cnv["win_st"]
    win_end = df_cnv["win_end"]

    # Check if the region of the genome to plot is defined:

    if pltstart == 0 and pltend == 0:
        df_plt = df_cnv
    elif pltstart == 0 and pltend > 0:
        endidx = find_nearest(win_end, pltend)
        df_plt = df_cnv.iloc[:endidx]
    elif pltend == 0 and pltstart > 0:
        stidx = find_nearest(win_st, pltstart)
        df_plt = df_cnv.iloc[stidx:]
    else:
        stidx =find_nearest(win_st,pltstart)
        endidx = find_nearest(win_end, pltend)
        df_plt = df_cnv.iloc[stidx:endidx]

    # find_nearest() clamps rather than failing, so a region lying wholly outside this
    # sequence collapses both ends onto the same window and slices to nothing. That
    # used to surface as "cannot convert float NaN to integer" from the median below.
    if df_plt.empty:
        print(
            f"WARNING: the requested region ({pltstart}-{pltend}) does not overlap "
            f"{genome_id}, which spans {int(win_st.min())}-{int(win_end.max())}. "
            "Plotting the whole sequence instead."
        )
        df_plt = df_cnv

    # ONE figure, at the same 10x8 the other four plotters use. There used to be a
    # bare plt.figure(figsize=(10, 8)) here as well, which did two things wrong:
    # the subplots() call below created a SECOND figure that savefig and close
    # both acted on, so the first leaked -- one empty figure per call, which is
    # what raised matplotlib's "More than 20 figures have been opened" on any run
    # touching enough sequences -- and every CNV plot came out at matplotlib's
    # default 6.4x4.8 rather than the 10x8 the code appeared to ask for.
    fig, ax1 = plt.subplots(figsize=(10, 8))

    ax2 = ax1.twinx()
    ax1.patch.set_visible(False)

    ax1.set_zorder(2)  # Higher than ax2
    ax2.set_zorder(1)  # Lower than ax1

    # Repeat windows are not drawn -- not their coverage, and not their copy
    # number. Their depth measures how many reference copies collapsed onto the
    # locus, not how many the sample carries: CWBI's chromosome shows 18x at
    # 2.13 Mb on exactly zero unique coverage. And the call there is INHERITED
    # from the surrounding segment rather than voted for, since run_HMM drops
    # these windows from its observation sequence, so a red tick would claim a
    # measurement that was never made. The gaps this leaves in the copy-number
    # track are the honest picture: that is where CNery has no evidence.
    keep = plottable(df_plt)
    drawn = df_plt.loc[keep]
    n_hidden = int((~keep).sum())
    if drawn.empty:
        drawn, n_hidden = df_plt, 0

    ax2.scatter(drawn["win_st"], drawn["read_count_cov"], color="gray",
                label="Raw reads", s=10, alpha=0.2)
    ax2.scatter(drawn["win_st"], drawn["otr_gc_corr_rdcnt_cov"], color="orange",
                label="Corrected reads", s=5, alpha=0.5,
                marker=mplt.markers.MarkerStyle(marker="o", fillstyle="none"))
    # COPY-NUMBER CALLS AS SEGMENTS, AT THEIR TRUE EXTENT.
    #
    # They used to be one scatter marker per window ("_", s=30), whose width is
    # fixed in POINTS rather than in data units. Measured at this figure size,
    # 5.5 pt is 44 kb of genome: a 1-window CN-0 call was drawn 440x too wide, a
    # 7-window CN-12 block 63x, and a 26-window deletion 17x. The marks bled
    # across neighbouring features, which is what made deletion and
    # amplification calls appear to overlap regions they do not cover.
    #
    # Drawn from df_plt rather than `drawn` because a copy-number call is a
    # property of the SEGMENT: it genuinely spans the repeat windows this figure
    # does not plot coverage for, and breaking the line there would suggest the
    # segment stops where only the evidence does.
    #
    # It is also 29 lines instead of 46,298 markers.
    cn_all = df_plt["prob_copy_number"].to_numpy()
    win_a = df_plt["win_st"].to_numpy()
    win_b = df_plt["win_end"].to_numpy()
    edges = np.r_[0, np.flatnonzero(np.diff(cn_all) != 0) + 1, len(cn_all)]
    for a, b in zip(edges[:-1], edges[1:]):
        ax1.hlines(cn_all[a], win_a[a], win_b[b - 1], color="red", linewidth=1.6,
                   zorder=6)
    ax1.plot([], [], color="red", linewidth=1.6, label="Predicted Copy Number")

    delta = int(drawn["read_count_cov"].median() * 0.5)
    
    (lo1, hi1), (lo2, hi2) = _cnv_axis_limits(df_cnv, drawn, delta)
    ax1.set_ylim(lo1, hi1)
    ax2.set_ylim(lo2, hi2)

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # THE RIGHT AXIS IS LABELLED AT THE LEFT AXIS'S TICKS, times the median depth.
    #
    # It used to carry a LinearLocator with the same NUMBER of ticks as the left
    # axis but no relation to their POSITIONS, so the two rulers disagreed: a
    # copy number of 2 is 339 reads on ltee_ara_m3_32k_2rg, while the right
    # axis's second tick read 381. Tying the SCALES is not enough on its own --
    # anyone reading a coverage off the right axis for a state marked on the left
    # still got a number the plot does not mean.
    #
    # And the minor locator was MultipleLocator(1) -- one minor tick per READ, so
    # roughly 3,000 of them across this range. That is what drew the right spine
    # as a solid black bar on every CNV plot, and what raised matplotlib's
    # "Locator attempting to generate 5800 ticks ... exceeds MAXTICKS" on every
    # run.
    ticks_cn, ticks_reads = _cnv_axis_ticks(ax1.get_yticks(), lo1, hi1, lo2, hi2)
    ax1.yaxis.set_major_locator(ticker.FixedLocator(ticks_cn))
    ax2.yaxis.set_major_locator(ticker.FixedLocator(ticks_reads))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax2.yaxis.set_minor_locator(ticker.NullLocator())

    
    ax1.set_xlabel("Window (Genomic position)")
    ax1.yaxis.label.set_color('red')
    ax2.set_ylabel("Read Counts (/)")
    ax1.set_ylabel("Copy Number (#)")
    
    plt.title(f'{samplename}_Copy Number Prediction')
    
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()

    # Combine handles and labels
    handles = handles_ax1 + handles_ax2
    labels = labels_ax1 + labels_ax2

    ax1.legend(handles, labels, loc='best')
    
    plt_full_path = os.path.join(saveplt,'%s_copy_numbers.pdf' % samplename)
    plt.savefig(plt_full_path, format = 'pdf', bbox_inches = 'tight')
    
    plt.close()    

#Probability calculations for the Emission and Transition matrices
def solve_pr(mean, variance):
    # A negative binomial needs variance > mean; at or below it there is no
    # finite `size`, and the arithmetic below divides by zero. Callers are
    # expected to have applied the var = mean * (1 + 1e-3) guard, but that guard
    # is written `if mean > 0`, so an all-zero frame arrives here as (0.0, 0.0)
    # and used to raise ZeroDivisionError from inside run_HMM -- taking down a
    # whole multi-sequence run because one plasmid got no reads. Decline instead;
    # run_HMM short-circuits such a frame before it gets here, and this is the
    # backstop for any other caller.
    if not (variance > mean):
        return 0.0, float("inf")
    r = (mean * mean)/(variance - mean)
    p = 1 - (mean/variance)
    return p, r

def calculate_logprob(p, r, obs):
    # log probabilities under a negative binomial (Poisson family), to account for
    # a wide dispersion of coverage data points given noisy data and high copy
    # number possibilities (amplifications). gammaln allows the calculation
    # without computational over-flow.
    return (gammaln(r + obs) - gammaln(obs + 1) - gammaln(r)
            + obs * np.log(p) + r * np.log(1 - p))


def calculate_prob(p, r, obs):
    return np.exp(calculate_logprob(p, r, obs))


def _nb_logpmf_mu(counts, mu, size):
    """NB log pmf in the (mu, size) parameterisation breseq and R use."""
    mu = np.asarray(mu, dtype=float)
    return calculate_logprob(mu / (size + mu), size, counts)


def _censor_bounds(values, lo_mult=0.5, hi_mult=1.5):
    """breseq's censoring window: fixed multiples either side of the mode.

    Ports `fit_censored_negative_binomial` in breseq's coverage_distribution.cpp:
    the mode is the peak of a 5-point centred moving average of the histogram,
    searched upward from `max(mean / 4, 1)` so a deletion spike near zero cannot
    win it, and the window is `[floor(lo_mult * mode), ceil(hi_mult * mode)]`.

    Low side removes deletions; high side removes amplifications and any repeat
    pile-up that survived window censoring. Single pass, no re-fit loop.

    Returns (lo, hi), or (None, None) if the histogram is degenerate.
    """
    v = np.asarray(values, dtype=int)
    v = v[v >= 1]                      # as in breseq, the histogram has no zero bin
    if v.size == 0:
        return None, None

    n_bins = int(v.max())
    hist = np.bincount(v, minlength=n_bins + 1).astype(float)
    total = hist[1:].sum()
    if total <= 0:
        return None, None
    mean = float((np.arange(n_bins + 1) * hist).sum() / total)

    if n_bins >= 5:
        smoothed = np.convolve(hist, np.full(5, 0.2), mode="same")
        valid = np.zeros(n_bins + 1, dtype=bool)
        valid[3:n_bins - 1] = True     # undefined within 2 bins of either end
    else:
        smoothed = hist
        valid = np.ones(n_bins + 1, dtype=bool)
        valid[0] = False

    valid[:max(int(mean / 4.0), 1)] = False
    if not valid.any():
        return None, None
    mode = int(np.argmax(np.where(valid, smoothed, -np.inf)))

    lo = max(int(np.floor(lo_mult * mode)), 1)
    hi = min(int(np.ceil(hi_mult * mode)), n_bins)
    if lo >= hi:
        return None, None
    return lo, hi


def fit_censored_negative_binomial(counts, offsets=None, min_windows=30,
                                   lo_mult=0.5, hi_mult=1.5, n_offset_bins=64):
    """(mu, size) of the SINGLE-COPY count distribution, or None if degenerate.

    `counts` are raw window depths and `offsets` the multiplicative GC/OTR bias
    factors, so E[count_i] = mu * offsets[i] at copy number 1. The bias belongs
    here, in the mean, rather than being divided out of the data: dividing a
    count by f scales its variance by 1/f^2, which a single global variance
    cannot represent, and rounding after dividing inflates the quantisation
    error of exactly the low-coverage windows that can least afford it.

    Censoring is breseq's (see _censor_bounds) and is what makes this measure
    WITHIN-state dispersion. np.var over every window measures the spread of a
    mixture -- deletions at 0x, amplifications at 2-3x -- so it reports BETWEEN
    -state variance and over-disperses every state at once. On REL606 at
    -w 100 -s 100 that is var/mean 6.0 where the within-state value is 2.6, and
    it costs ~13 nats per window of CN2-vs-CN1 evidence at 2.5x coverage: more
    than the entire budget of a two-window event.

    Unlike breseq, the objective is the truncated likelihood rather than least
    squares on a renormalised histogram -- breseq's choice is a concession to
    its thousands of coarse-grained per-base bins. The fitted `size` is
    therefore not numerically comparable to breseq's `nbinom_size_parameter`,
    which is also fitted to unique-only PER-BASE counts rather than per-window
    medians.
    """
    counts = np.asarray(counts, dtype=float)
    offsets = (np.ones_like(counts) if offsets is None
               else np.asarray(offsets, dtype=float))

    usable = (np.isfinite(counts) & np.isfinite(offsets)
              & (offsets > 0) & (counts >= 0))
    counts, offsets = counts[usable], offsets[usable]
    if counts.size < min_windows:
        return None

    # Offsets only matter up to a constant -- it trades off exactly against mu --
    # so normalise, which makes mu the single-copy depth at a typical window and
    # makes the fit independent of how apply_otr_correction scales its no-bias
    # fallback (a constant equal to the mean, not 1.0).
    offsets = offsets / np.median(offsets)

    ratio = np.rint(counts / offsets).astype(int)
    lo, hi = _censor_bounds(ratio, lo_mult, hi_mult)
    if lo is None:
        return None

    keep = (ratio >= lo) & (ratio <= hi)
    if keep.sum() < min_windows:
        return None

    kept_counts = np.rint(counts[keep]).astype(int)
    kept_offsets = offsets[keep]
    kept_ratio = ratio[keep]

    # A negative binomial cannot represent under-dispersed data: the MLE would
    # drive size off to infinity against the optimiser bound. Synthetic frames
    # with Gaussian noise land here, and so does a perfectly flat one. Fall back
    # to moments plus the historical `var = mean * (1 + 1e-3)` guard -- but
    # taken over the CENSORED subset, so an amplification cannot inflate the
    # dispersion that is supposed to detect it.
    kept_mean = float(kept_ratio.mean())
    kept_var = float(kept_ratio.var())
    if kept_var <= kept_mean:
        if kept_mean <= 0:
            return None
        guarded_var = kept_mean * (1.0 + 1e-3)
        return kept_mean, kept_mean * kept_mean / (guarded_var - kept_mean)

    # Collapse to (offset bin, count) cells so each objective evaluation costs
    # thousands of gammaln calls rather than tens of thousands -- the same
    # histogram trick breseq uses, extended to the offset dimension.
    edges = np.linspace(kept_offsets.min(), kept_offsets.max(), n_offset_bins + 1)
    obin = np.clip(np.digitize(kept_offsets, edges[1:-1]), 0, n_offset_bins - 1)
    bin_offset = np.array([
        kept_offsets[obin == b].mean() if np.any(obin == b) else np.nan
        for b in range(n_offset_bins)
    ])
    live = np.isfinite(bin_offset)
    bin_offset = bin_offset[live]
    obin = np.searchsorted(np.flatnonzero(live), obin)

    cell_key = obin.astype(np.int64) * (kept_counts.max() + 1) + kept_counts
    uniq_key, cell_weight = np.unique(cell_key, return_counts=True)
    cell_obin, cell_count = np.divmod(uniq_key, kept_counts.max() + 1)
    cell_offset = bin_offset[cell_obin]
    cell_weight = cell_weight.astype(float)

    bin_weight = np.bincount(cell_obin, weights=cell_weight,
                             minlength=bin_offset.size)

    def neg_log_likelihood(params):
        mu, size = np.exp(params)
        if not (np.isfinite(mu) and np.isfinite(size)) or mu <= 0 or size <= 0:
            return 1e12
        ll = float((cell_weight * _nb_logpmf_mu(cell_count, mu * cell_offset, size)).sum())
        # Conditioning on the censoring window: the bounds are on the ratio, so
        # per window they are [lo * offset, hi * offset].
        bin_mu = mu * bin_offset
        prob = size / (size + bin_mu)
        mass = (nbinom.cdf(np.floor(hi * bin_offset), size, prob)
                - nbinom.cdf(np.ceil(lo * bin_offset) - 1, size, prob))
        if not np.all(mass > 0) or not np.isfinite(ll):
            return 1e12
        return -(ll - float((bin_weight * np.log(mass)).sum()))

    mu0 = kept_mean
    size0 = mu0 * mu0 / max(kept_var - mu0, 1e-6)

    best, best_score = None, np.inf
    for mu_try in (mu0, 0.5 * (lo + hi), float(hi), float(lo)):
        for size_try in (size0, 1e3, 1e1, 1e-1):
            if mu_try <= 0 or size_try <= 0:
                continue
            result = minimize(neg_log_likelihood, np.log([mu_try, size_try]),
                              method="Nelder-Mead",
                              options=dict(xatol=1e-6, fatol=1e-6, maxiter=2000))
            if not result.success and not np.isfinite(result.fun):
                continue
            if result.fun < best_score:
                best_score, best = result.fun, np.exp(result.x)

    if best is None or best_score >= 1e12:
        return None

    mu, size = float(best[0]), float(best[1])
    if not (np.isfinite(mu) and np.isfinite(size)) or mu <= 0 or size <= 0:
        return None

    # breseq rejects a fit whose mass mostly falls outside the fitting window.
    prob = size / (size + mu)
    included = float(nbinom.cdf(hi, size, prob) - nbinom.cdf(lo - 1, size, prob))
    if included < 0.01:
        return None

    return mu, size


def robust_state_count(counts, offsets, mu, min_states=5, max_states=100, support=3):
    """How many copy-number states the model needs, ignoring lone spikes.

    Taking `int(max(coverage))` lets a SINGLE outlier window set the state space
    for the whole genome. That is not just wasted work: with a flat off-diagonal
    the cost of every state change carries -log(n_states), so one window at 40x
    would make calling a duplication ~1.3 nats dearer everywhere.

    A state that no segment ever occupies costs sensitivity and buys nothing --
    a one-window excursion cannot pay for its own entry and exit regardless --
    so the ceiling comes from a `support`-window rolling median. A real
    high-copy segment spans several windows and survives it; a spike does not.
    """
    counts = np.asarray(counts, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if mu <= 0 or counts.size == 0:
        return int(min_states)

    ratio = np.nan_to_num(counts / (mu * np.where(offsets > 0, offsets, 1.0)))
    if ratio.size >= support:
        ratio = ndimage.median_filter(ratio, size=support, mode="nearest")

    needed = int(np.ceil(np.nan_to_num(ratio.max())))
    return int(min(max(needed, int(min_states)), int(max_states)))


def log_emission_with_offsets(counts, offsets, mu, size, n_states,
                              deletion_coverage_fraction, offset_tau=None):
    """(n_obs, n_states + 1) log emission matrix with per-window bias offsets.

    State index == copy number. Row k is NegBinom(mu = k * mu * offset_i,
    size = k * size); scaling `size` with the state alongside the mean keeps
    variance proportional to copy number. Row 0 is a geometric of mean
    `deletion_coverage_fraction * mu * offset_i`, so it is the same statement
    with k = deletion_coverage_fraction rather than a special case -- every row
    now reads the fitted baseline and the window's bias offset.

    That matters because the zero state used to be `geom.pmf(count + 1,
    1 - error_rate)`, a geometric of mean `error_rate / (1 - error_rate)` =
    0.176 counts ABSOLUTE, with nothing tying it to the sample's coverage. As a
    fraction of baseline that is 0.28% at 60x and 0.018% at 1000x, so the
    largest residual coverage it would still call CN0 drifted from 19% to 4%:
    the same deletion was called on a shallow run and missed on a deep one.

    A per-window matrix replaces the old (state, count) lookup table because
    the offsets vary per window, which no shared table can express. It also
    removes the table's `absmax` ceiling and the read-count clipping that fed it.
    """
    counts = np.asarray(counts, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    out = np.empty((counts.size, n_states + 1), dtype=float)

    # The floor is load-bearing, not defensive: on an all-zero frame the
    # censored fit declines and run_HMM falls back to moments, where mu is 0.
    zero_mean = np.maximum(float(deletion_coverage_fraction) * mu * offsets, 1e-9)
    out[:, 0] = counts * np.log(zero_mean / (1.0 + zero_mean)) - np.log1p(zero_mean)
    # `offset_tau` is the relative sd of the bias offset. Treating the offset as
    # exact understates the variance, and understates it MORE at high copy number:
    # writing o = o_hat * (1 + eps) with Var(eps) = tau^2, the law of total
    # variance gives
    #
    #     Var(y | k) = m + m^2 * (1/(k*size) + tau^2)      m = k * mu * o_hat
    #
    # so the uncertainty adds in the reciprocal-size scale and the extra term
    # m^2 * tau^2 grows as k^2 -- negligible at CN 1, largest exactly where the
    # correction is multiplied up. The k^2 scaling falls out; it is not imposed.
    #
    # Note the size must be formed PER STATE: k*size/(1 + k*size*tau^2), not
    # k*(size/(1 + size*tau^2)), or the k^2 property is lost.
    tau2 = None
    if offset_tau is not None:
        tau2 = np.asarray(offset_tau, dtype=float) ** 2
        if not np.any(tau2 > 0):
            tau2 = None

    for state in range(1, n_states + 1):
        state_size = state * size
        if tau2 is not None:
            state_size = state_size / (1.0 + state_size * tau2)
        out[:, state] = _nb_logpmf_mu(counts, state * mu * offsets, state_size)

    out[~np.isfinite(out)] = -np.inf
    return out

#Emission Matrix
def setup_emission_matrix(n_states, mean, variance, absmax,
                          deletion_coverage_fraction):
    emission = np.full((n_states, absmax + 1), np.nan)
    
    for state in range(n_states):
        pr = solve_pr(mean * (state + 1), variance * (state + 1))
        p, r = pr[0], pr[1]
        
        for obs in range(absmax + 1):
            emission[state, obs] = calculate_prob(p, r, obs)
    
    # Row 0 is a geometric whose mean is a FRACTION of the single-copy level,
    # matching log_emission_with_offsets(). An absolute mean here would make the
    # threshold for "deleted" depend on how deeply the sample was sequenced.
    obs_range = np.arange(absmax + 1)
    zero_mean = max(float(deletion_coverage_fraction) * mean, 1e-9)
    zero_row = geom.pmf(obs_range + 1, 1.0 / (1.0 + zero_mean))
    emission = np.vstack((zero_row, emission))
    # np.savetxt("emission.csv", emission, delimiter=",")  
    return emission

#: Prior probability per BASE that copy number changes. 1e-6 is one boundary per
#: ~1 Mb, against the ~25-40 segments these genomes actually carry.
DEFAULT_CHANGE_RATE = 1e-6

#: Coverage a deleted region still shows, as a fraction of the single-copy
#: level. Measured on REL606's called deletions: mean 2.2% of baseline, 90th
#: percentile 3.2% -- mismapping and repeat spill. It sets the mean of the
#: HMM's copy-number-0 emission, which is a fraction rather than an absolute
#: count precisely so that the threshold does not move with sequencing depth.
DEFAULT_DELETION_COVERAGE_FRACTION = 0.02


def window_geometry(df):
    """(step, window) in bases, recovered from the frame.

    run_HMM is not given -w/-s, but preprocess() writes win_st/win_end/win_len in
    every --bias branch and every test fixture, so the geometry is recoverable
    without changing any signature.
    """
    starts = np.asarray(df["win_st"], dtype=float)
    step = float(np.median(np.diff(starts))) if starts.size > 1 else 0.0

    if "win_len" in df.columns:
        window = float(np.median(np.asarray(df["win_len"], dtype=float)))
    else:
        window = float(np.median(np.asarray(df["win_end"], dtype=float) - starts))

    if not (step > 0):
        step = window if window > 0 else 1.0
    if not (window > 0):
        window = step
    return step, window


def remain_prob_for_step(change_rate, step):
    """P(copy number does not change across `step` bases).

    Boundaries are modelled as a Poisson process of rate `change_rate` per base,
    so P(remain) = exp(-rate * step). The prior then describes the GENOME rather
    than the tiling: `changeprob` was a flat per-window probability, which means
    the implied per-base rate was changeprob/step and re-tiling the same genome
    silently restated the biology. That is what made short amplifications
    uncallable at one geometry and callable at another.
    """
    remain = float(np.exp(-float(change_rate) * float(step)))
    return float(np.clip(remain, 1e-12, 1.0 - 1e-12))


#Transition Matrix setup
def setup_transition_matrix(n_states, remain_prob):
    #include zero state:
    n_states += 1
    
    change_prob = 1 - remain_prob
    per_state_prob = change_prob / (n_states - 1)
    
    transition = np.full((n_states, n_states), per_state_prob)
    
    for i in range(n_states):
        transition[i, i] = remain_prob
    # np.savetxt("transition.csv", transition, delimiter=",") 
    return transition

def _log_emission_lookup(obs, emission_matrix):
    """Select log emission probabilities for `obs` from a (state, count) table.

    Returns an (n_obs, n_states) array -- the orientation _viterbi_forward()
    wants -- with exact zeros mapped to -inf rather than a warning.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        logemi = np.log(np.asarray(emission_matrix, dtype=float))
    logemi[np.isnan(logemi)] = -np.inf
    return logemi[:, np.asarray(obs, dtype=int)].T


def _viterbi_forward(log_emission_obs, log_transition, log_start):
    """Forward pass of Viterbi, keeping backpointers.

    `log_transition` is indexed [from, to] -- the orientation the recursion
    actually needs. The old code indexed it [to, from] and got away with it
    only because setup_transition_matrix() returns a symmetric matrix.

    Returns (logv, ptr) where ptr[i, l] is the state at i-1 on the best path
    that ends in state `l` at window i. ptr[0] is unused.
    """
    log_emission_obs = np.asarray(log_emission_obs, dtype=float)
    n_obs, n_states = log_emission_obs.shape

    logv = np.full((n_obs, n_states), -np.inf)
    ptr = np.zeros((n_obs, n_states), dtype=np.int32)

    logv[0] = log_start + log_emission_obs[0]

    state_idx = np.arange(n_states)
    for i in range(1, n_obs):
        # cand[k, l] = score of the best path reaching k at i-1, then k -> l
        cand = logv[i - 1][:, None] + log_transition
        ptr[i] = np.argmax(cand, axis=0)
        logv[i] = cand[ptr[i], state_idx] + log_emission_obs[i]

    return logv, ptr


def _backtrace(logv, ptr):
    """Recover the single most probable state path from the forward pass.

    This is the step the old code was missing: it took np.argmax(logv, axis=1)
    per window, which is a per-window max over paths *ending* there and is not
    a path at all. Inside an elevated run the high-state column climbs relative
    to CN1 window by window and only overtakes at the last one, which is why an
    amplification could come out labelled `1,1,3`.
    """
    n_obs = logv.shape[0]
    path = np.empty(n_obs, dtype=int)
    path[-1] = int(np.argmax(logv[-1]))
    for i in range(n_obs - 1, 0, -1):
        path[i - 1] = ptr[i, path[i]]
    return path


def viterbi_path(log_emission_obs, log_transition, log_start):
    """Most probable state path. See _viterbi_forward() for the conventions."""
    logv, ptr = _viterbi_forward(log_emission_obs, log_transition, log_start)
    return _backtrace(logv, ptr)


def _default_log_start(log_transition):
    """Start distribution: the reference is entered from copy number 1.

    Window 0 therefore contributes its own emission and may differ from CN1 at
    the cost of exactly one transition, instead of being pinned to CN1 by the
    old `logv[0, 1] = log(1e-100)`. That matters for a circularly permuted
    reference, which can begin inside a deletion.
    """
    return log_transition[1, :].copy()


#Make Viterbi Matrtix
def make_viterbi_mat(obs, transition_matrix, emission_matrix):
    """Forward Viterbi scores only, kept for callers that want the matrix.

    Segment calling goes through HMM_copy_number(), which needs the
    backpointers this discards.
    """
    log_transition = np.log(transition_matrix)
    logv, _ = _viterbi_forward(
        _log_emission_lookup(obs, emission_matrix),
        log_transition,
        _default_log_start(log_transition),
    )
    return logv


def _segments_from_path(path, win_st, win_end, chr_length):
    """Collapse a per-window state path into Startpos/Endpos/State rows."""
    def _at(seq, i):
        return seq.iloc[i] if hasattr(seq, "iloc") else seq[i]

    rows = []
    start_pos = 0
    prev_state = path[0]

    # range(1, len(path)) -- the old loop ran to len(obs) - 1 and so could never
    # emit a state change at the final window.
    for i in range(1, len(path)):
        state = path[i]
        if state != prev_state:
            rows.append({
                "Startpos": start_pos,
                "Endpos": _at(win_end, i - 1),
                "State": prev_state,
            })
            start_pos = _at(win_st, i)
        prev_state = state

    rows.append({
        "Startpos": start_pos,
        "Endpos": chr_length,
        "State": prev_state,
    })

    return pd.DataFrame(rows, columns=["Startpos", "Endpos", "State"])


def HMM_copy_number(obs, transition_matrix, emission_matrix, win_st, win_end, chr_length,
                    *, log_emission_obs=None, emission_weight=1.0):
    """Segment the genome by Viterbi decoding.

    `log_emission_obs` lets a caller supply a per-window (n_obs, n_states) log
    emission matrix directly -- needed once the GC/OTR corrections enter as
    per-window offsets on the mean, which a shared (state, count) lookup table
    cannot express. When omitted, `emission_matrix` is used as that lookup.

    `emission_weight` tempers the likelihood by a constant factor, used to stop
    overlapping windows counting the same bases more than once.
    """
    if log_emission_obs is None:
        log_emission_obs = _log_emission_lookup(obs, emission_matrix)
    log_emission_obs = np.asarray(log_emission_obs, dtype=float) * float(emission_weight)

    log_transition = np.log(transition_matrix)
    path = viterbi_path(
        log_emission_obs, log_transition, _default_log_start(log_transition)
    )
    return _segments_from_path(path, win_st, win_end, chr_length)


#: which correction factors compose the emission offset, per --bias mode.
BIAS_OFFSET_COLUMNS = {
    "all": ("gc_corr_fact", "otr_gc_corr_fact"),
    "gc": ("gc_corr_fact",),
    "otr": ("otr_gc_corr_fact",),
    "none": (),
}


def offset_tau(df, bias="all"):
    """Relative uncertainty of the bias offset, per window.

    Only the GC arm contributes: `gc_corr_tau` is measured by resampling the
    LOWESS fit (see _gc_curve_tau). The OTR tent has its own uncertainty which is
    NOT modelled here -- it is a two-parameter fit whose error is structured
    along the genome rather than pointwise, so a per-window sd would misrepresent
    it. Returns zeros when the column is absent or the mode applies no GC
    correction, which reproduces the exact-offset behaviour.
    """
    zero = np.zeros(len(df), dtype=float)
    if "gc_corr_fact" not in BIAS_OFFSET_COLUMNS.get(bias, ()):
        return zero
    if "gc_corr_tau" not in df.columns:
        return zero
    tau = df["gc_corr_tau"].to_numpy(dtype=float)
    return np.where(np.isfinite(tau) & (tau > 0), tau, 0.0)


def bias_offsets(df, bias="all"):
    """Multiplicative GC/OTR bias factor per window, for the emission mean.

    Both factor columns are DIVISORS applied to normalized coverage
    (apply_gc_correction: `gc_corr_norm_cov = norm_raw_cov / gc_corr_fact`;
    apply_otr_correction: `otr_gc_corr_fact` is otr_fit's `y_fit`, and
    `y_corr = y / y_fit`). So the same numbers multiply the EXPECTED raw count,
    which is where they belong.

    `bias` has to be passed in rather than inferred: process_multi_genome runs
    GC correction unconditionally, so `gc_corr_fact` is on the frame even under
    --bias none, and only the caller knows which factors were meant to apply.
    """
    offsets = np.ones(len(df), dtype=float)
    for column in BIAS_OFFSET_COLUMNS.get(bias, BIAS_OFFSET_COLUMNS["all"]):
        if column in df.columns:
            factor = df[column].to_numpy(dtype=float)
            offsets *= np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)
    return offsets


def run_HMM(df, output, deletion_coverage_fraction=DEFAULT_DELETION_COVERAGE_FRACTION,
            n_states=5, changeprob=None,
            max_copy_number=100, min_called_windows=100, bias="all",
            change_rate=DEFAULT_CHANGE_RATE, overlap_weighting=True, write=True,
            genome_id=None):
    """
    Viterbi copy-number calling.

    `write=False` runs the identical numeric path but emits no files and prints
    nothing. That is what the FIRST of the two fitting passes uses: its calls
    exist only to build the CN censor for the second pass, and writing them
    would put provisional numbers in CNV_csv/ that the second pass then
    overwrites -- or worse, leaves behind if the run dies in between.

    `change_rate` is the prior probability PER BASE that copy number changes, so
    the per-window probability is 1 - exp(-change_rate * step) and re-tiling the
    same genome no longer restates the biology. `changeprob` is the escape
    hatch: pass a float to go back to a flat per-window probability, ignoring
    `change_rate`.

    `overlap_weighting` tempers the log emissions by step/window. At the default
    -w 200 -s 100 every base sits in two windows, so the likelihood would
    otherwise count each base's evidence twice while the transition prior counts
    it once.

    The observation is the RAW window depth (`read_count_cov`); the GC and OTR
    corrections enter as multiplicative offsets on the emission mean, so that
    E[count | CN = k] = k * mu * offset. Dividing them out of the data instead
    would scale each window's variance by 1/offset^2 while a single global
    variance was applied to all of them, and would round after dividing, which
    inflates the quantisation error of exactly the low-coverage windows that
    can least afford it.

    Windows flagged `is_redundant` by mask_coverage_windows() are censored
    from the observation sequence and from the emission-model fit:
    coverage over a repeat reflects how many copies collapsed onto that
    locus, not the sample's copy number there, so leaving them in both
    inflates the variance -- pushing `n_states` up until pile-ups get
    their own spurious high-CN segments -- and lets a single repeat
    window break a genuine deletion in two.

    They are not dropped from the frame. Each still carries its
    bias-corrected coverage and inherits `prob_copy_number` from the
    segment it falls in, and `is_redundant` (if present) survives into
    the CNV.csv alongside so an inherited call can be told from a real
    one.

    `min_called_windows` is a floor: if censoring would leave fewer
    windows than this, every window is used instead. That keeps small
    references and the synthetic test fixtures -- which may carry no
    `is_redundant` column at all -- behaving exactly as before. It governs
    the OBSERVATION SEQUENCE only: the emission fit excludes every window
    with any redundant coverage unconditionally, because a window that
    merely clips the edge of an IS element sits at 1.2-1.4x, inside the
    fit's censoring window, where it would inflate the dispersion precisely
    where sharpness matters most.
    """

    saveloc = os.path.join(output, "CNV_csv")
    # `genome_id` is normally read off the frame, but a sequence with no
    # coverage has no row to read it from, and its CSVs still have to be named.
    if genome_id is None:
        genome_id = str(df["genome_id"].iloc[0])
    genome_id = str(genome_id)
    samplename = sample_prefix(output) + genome_id

    new_exp = df.copy()

    new_exp.loc[:, "otr_gc_corr_norm_cov"] = np.nan_to_num(new_exp["otr_gc_corr_norm_cov"].to_numpy())

    med = new_exp["read_count_cov"].median()
    # No windows at all: the median is NaN and int(NaN) raises. Zero is the
    # honest baseline for a sequence with no coverage, and it keeps the
    # back-converted read-count column below well-defined.
    if not np.isfinite(med):
        med = 0.0

    rc_cap = int(max(1.0, max_copy_number) * med)

    # Back-converted read counts are computed for EVERY window -- the
    # column is part of the pipeline contract and feeds the diagnostic
    # plots.
    new_exp.loc[:, "otr_gc_corr_rdcnt_cov"] = (
        (new_exp["otr_gc_corr_norm_cov"] * med)
        .round()
        .astype(int)
        .clip(upper=rc_cap)
    )

    # Raw counts and their bias offsets, over every window.
    counts_all = np.rint(
        np.nan_to_num(new_exp["read_count_cov"].to_numpy(dtype=float))
    ).clip(min=0)
    offsets_all = bias_offsets(new_exp, bias=bias)

    # NOTHING TO CALL. Either the sequence has no windows, or every window is at
    # zero coverage. Both used to reach solve_pr(0.0, 0.0) and raise
    # ZeroDivisionError -- the moment fallback's guard is written `if mean > 0`,
    # so it steps over exactly the case that needs it. Every window is CN 0,
    # which is what "no reads mapped here" means, and the files are still
    # written: a caller that gets no CSV cannot tell this from a crash.
    #
    # The predicate is deliberately the DATA being empty, not `fit_result is
    # None` -- that also fires on healthy-but-small frames, where the moment
    # fallback below is a live and correct path.
    if len(new_exp) == 0 or counts_all.max(initial=0.0) <= 0:
        new_exp = new_exp.reset_index(drop=True)
        new_exp.loc[:, "prob_copy_number"] = np.zeros(len(new_exp), dtype=int)
        if write:
            empty_breaks = pd.DataFrame(
                {"Startpos": pd.Series(dtype=int),
                 "State": pd.Series(dtype=int),
                 "Segment_Size": pd.Series(dtype=int)}
            )
            # Three columns, header row, no data. breseq asserts the column
            # count and the assert is fatal, so an empty file is not an option.
            empty_breaks.to_csv(
                os.path.join(saveloc, f"{samplename}_break_pts.csv"), index=False
            )
            new_exp.to_csv(
                os.path.join(saveloc, f"{samplename}_CNV.csv"), index=False
            )
            print(f"{samplename}: no coverage to call; "
                  "copy number 0 across the sequence. .csv files saved.")
        return new_exp

    if "is_redundant" in new_exp.columns:
        not_redundant = ~new_exp["is_redundant"].to_numpy(dtype=bool)
    else:
        not_redundant = np.ones(len(new_exp), dtype=bool)

    # An offset is only meaningful up to a constant -- it trades off exactly
    # against mu -- so anchor it once, on the windows the fit will see, and use
    # that same scale for the emissions. Otherwise mu would be estimated on one
    # normalisation and applied on another.
    anchor = np.median(offsets_all[not_redundant])
    if np.isfinite(anchor) and anchor > 0:
        offsets_all = offsets_all / anchor

    # The fit sees only clean windows, always. `min_called_windows` softens the
    # OBSERVATION mask below, never this one.
    fit_result = fit_censored_negative_binomial(
        counts_all[not_redundant], offsets_all[not_redundant]
    )

    called = not_redundant.copy()
    if called.sum() < min_called_windows:
        called = np.ones(len(new_exp), dtype=bool)

    obs_exp = new_exp.loc[called]
    counts = counts_all[called]
    offsets = offsets_all[called]

    if fit_result is not None:
        mu, size = fit_result
    else:
        # Degenerate input -- a flat or under-dispersed frame, where a negative
        # binomial has no finite `size`. Fall back to the historical moment
        # estimate and its guard so such frames behave exactly as before.
        mean = float(np.mean(counts))
        var = float(np.var(counts))
        if mean > 0 and var <= mean:
            var = mean * (1.0 + 1e-3)
        p, size = solve_pr(mean, var)
        mu = mean

    n_states = robust_state_count(
        counts, offsets, mu, min_states=5, max_states=int(max_copy_number)
    )

    this_log_emission = log_emission_with_offsets(
        counts, offsets, mu=mu, size=size,
        n_states=n_states,
        deletion_coverage_fraction=deletion_coverage_fraction,
        offset_tau=offset_tau(new_exp, bias=bias)[called],
    )
    step_bp, window_bp = window_geometry(new_exp)

    if changeprob is not None:
        remain_prob = 1.0 - float(changeprob)
    else:
        remain_prob = remain_prob_for_step(change_rate, step_bp)

    this_transition = setup_transition_matrix(n_states, remain_prob=remain_prob)

    # Every observed transition is charged one step, whatever gap the censored
    # windows left. Pricing a wide repeat gap as a proportionally cheaper
    # crossing would make a censored repeat a cheap place to break a segment --
    # exactly what censoring them was meant to prevent.
    emission_weight = min(1.0, step_bp / window_bp) if overlap_weighting else 1.0

    # HMM_copy_number indexes win_st/win_end positionally, so the
    # censored subset can be passed straight through. chr_length stays
    # the genome end over ALL windows -- the last segment must reach it
    # even if the final window is censored.
    copy_numbers = HMM_copy_number(
        counts,
        this_transition,
        None,
        obs_exp["win_st"],
        obs_exp["win_end"],
        new_exp["win_end"].max(),
        log_emission_obs=this_log_emission,
        emission_weight=emission_weight,
    )

    if write:
        brk_full_path = os.path.join(saveloc, f"{samplename}_break_pts.csv")
        cn_brk = copy_numbers.loc[:, ["Startpos", "Endpos", "State"]].copy()
        cn_brk.loc[:, "Segment_Size"] = cn_brk["Endpos"] - cn_brk["Startpos"]
        cn_brk = cn_brk.drop(columns="Endpos")
        cn_brk.to_csv(brk_full_path, index=False)

    # Assign by window index rather than by appending to a flat list:
    # once censored windows are absent from the observation sequence a
    # segment can span windows that never voted for it, so the old
    # len(CN_HMM) == len(new_exp) invariant no longer holds.
    CN_HMM = pd.Series(np.nan, index=new_exp.index, dtype=float)

    for cnrow in copy_numbers.itertuples():
        in_segment = (
            (new_exp["win_st"] >= int(cnrow.Startpos))
            & (new_exp["win_st"] < int(cnrow.Endpos))
        )
        CN_HMM[in_segment] = int(cnrow.State)

    # Windows on a segment boundary can fall outside every half-open
    # interval; carry the neighbouring call across rather than leaving a
    # hole.
    new_exp.loc[:, "prob_copy_number"] = CN_HMM.ffill().bfill().astype(int)

    new_exp = new_exp.reset_index(drop=True)

    if write:
        csv_full_path = os.path.join(saveloc, f"{samplename}_CNV.csv")
        new_exp.to_csv(csv_full_path, index=False)
        print(f"{samplename}: Copy number prediction complete. .csv files saved.")

    return new_exp
