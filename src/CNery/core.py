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

    start_coord = int(df_b2c.index[0])
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

    return df_gc


def gc_cor_plots(df, output):
    genome_ids = sorted(df["genome_id"].unique())
    if len(genome_ids) > 1:
        label = "_and_".join(str(g) for g in genome_ids)
    else:
        label = str(genome_ids[0])
    samplename = f"{label}_GC_vs_NormRds"
    saveplt = str(output + "/GC_bias/")

    os.makedirs(saveplt, exist_ok=True)

    plt.figure(figsize=(10, 8))

    uniq = (
        df[['gc_percent', 'gc_corr_fact']]
        .drop_duplicates(subset='gc_percent')
        .sort_values('gc_percent')
    )
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


def fit_gc_bias(
    df,
    censor_col="exclude_from_fit",
    n_robust_iter=3,
    resid_mad=5.0,
    fit_floor_frac=0.05,
):
    """
    Fit stage of GC-bias correction: iterative robust LOWESS of
    norm_raw_cov vs gc_percent, using only windows where
    df[censor_col] is False (i.e. neither deletions nor redundant-coverage
    windows).

    Returns
    -------
    dict : {"gc_sorted", "fit_sorted", "floor"} -- pass to
    apply_gc_correction().
    """
    cov = df["norm_raw_cov"].to_numpy(dtype=float)
    gc = df["gc_percent"].to_numpy(dtype=float)
    censored = df[censor_col].to_numpy(dtype=bool)

    loess = sm.nonparametric.lowess
    fit_mask = (~censored) & np.isfinite(cov) & np.isfinite(gc)

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

    return {"gc_sorted": gc_sorted, "fit_sorted": fit_sorted, "floor": floor}


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
    return df


def process_multi_genome(
    coverage_inputs,
    output_prefix,
    win=100,
    step=100,
    frag=400,
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
    for genome_id, path in coverage_inputs.items():
        df_raw = read_coverage_table(path)
        df_pre = preprocess(df_raw, win=win, step=step, frag=frag)
        df_pre["genome_id"] = genome_id
        preprocessed[genome_id] = df_pre

    df_pooled = pd.concat(preprocessed.values(), ignore_index=True)

    global_median = df_pooled["read_count_cov"].median()
    df_pooled["norm_raw_cov"] = df_pooled["read_count_cov"] / global_median

    df_pooled = mask_coverage_windows(df_pooled)
    gc_fit = fit_gc_bias(df_pooled)
    df_pooled = apply_gc_correction(df_pooled, gc_fit)

    gc_cor_plots(df_pooled, output_prefix)

    per_genome_corrected = {}
    for genome_id in coverage_inputs:
        df_g = df_pooled[df_pooled["genome_id"] == genome_id].copy()
        df_g.reset_index(drop=True, inplace=True)
        per_genome_corrected[genome_id] = df_g

    return per_genome_corrected


def plot_otr_corr(df, output, ori, ter):

    genome_id = str(df["genome_id"][0])
    samplename = output.strip().split('/')[-1] + genome_id
    saveplt = str(output+"/OTR_corr/")
  
    plt.figure(figsize=(10, 8))
    plt.scatter(df["win_st"],df["norm_raw_cov"], color="gray", label="Raw reads",s=8, alpha = 0.2)
    plt.scatter(df["win_st"],df["gc_corr_norm_cov"], color="black", label="GC corrected", marker = '*', s=15, alpha = 0.5)
    plt.scatter(df["win_st"],df["otr_gc_corr_norm_cov"], color = 'orange', label="Ori/Ter bias corrected", s = 20, alpha = 0.85, 
                marker = mplt.markers.MarkerStyle(marker = 'o', fillstyle = 'full'))
    plt.plot(df["win_st"], df["otr_gc_corr_fact"], color = "black", label = "OTR-bias-fit-line")
    plt.plot(df["win_st"],df["gc_cor_med_fil"], color="blue", label="Med-fil")
    
    plt.axvline(x=ter, color='r', linestyle=':', label=f'Terminus: {ter}')
    plt.axvline(x=ori, color='r', linestyle=':', label=f'Origin: {ori}')
    plt.xlabel("Window (Genomic position)")
    plt.ylabel("Normalized read coverage")
    plt.title(f'{samplename}_Ori/Ter bias correction')
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
# The concentrated RSS surface has kinks wherever a window switches which
# arc it belongs to, so Nelder-Mead is run from several starting points
# spread evenly around the circle (each paired with its antipode, since
# bidirectional replication puts ori/ter roughly opposite each other),
# plus the masked argmax/argmin seed used previously, and the lowest-RSS
# result across all starts is kept. This guards against a single bad seed
# landing in a worse local optimum. Each fit is fast (well under a second
# even at ~10,000 windows with the default 9 seeds), so the multi-start
# adds negligible cost.
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
def otr_fit(df, bias_threshold=1.0, n_seeds=8):

    x = df.index.to_numpy().astype(float)
    y = df["gc_corr_norm_cov"].to_numpy(dtype=float)
    y_med_fil = df["gc_cor_med_fil"].to_numpy(dtype=float)
    n = len(x)
    genome_len = float(n)

    # Windows to exclude from both the seed search and the least-squares
    # fit: genuine deletions and redundant/repeat-coverage windows, if
    # mask_coverage_windows() has already flagged them.
    exclude = np.zeros(n, dtype=bool)
    if "is_deletion" in df.columns:
        exclude |= df["is_deletion"].to_numpy(dtype=bool)
    if "is_redundant" in df.columns:
        exclude |= df["is_redundant"].to_numpy(dtype=bool)

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

    print(f'o_idx_seed:{o_idx_seed} and t_idx_seed: {t_idx_seed}')

    # Fit using only unmasked windows -- deletions/repeats never inform
    # the breakpoint search or the anchor-value least-squares solve.
    fit_mask = ~exclude
    x_fit = x[fit_mask]
    y_fit_data = y[fit_mask]

    if fit_mask.sum() < 4:
        # Not enough clean data to fit anything meaningful.
        y_flat = np.repeat(np.mean(y), n)
        print("OTR bias not detected (insufficient clean windows)")
        return y, y_flat, o_idx_seed, t_idx_seed, False

    # Multi-start: the masked argmax/argmin seed, plus n_seeds evenly
    # spaced positions around the circle, each paired with its antipode.
    seeds = [(o_idx_seed, t_idx_seed)]
    for k in range(n_seeds):
        s = (k / n_seeds) * genome_len
        seeds.append((s, (s + genome_len / 2.0) % genome_len))

    best = None
    for x0 in seeds:
        res = minimize(
            lambda p: _otr_concentrated_rss(p, x_fit, y_fit_data, genome_len)[0],
            x0=list(x0),
            method="Nelder-Mead",
            options={"xatol": 0.5, "fatol": 1e-8, "maxiter": 2000},
        )
        rss, y_ori_cand, y_ter_cand = _otr_concentrated_rss(res.x, x_fit, y_fit_data, genome_len)
        if not np.isfinite(rss):
            continue
        if best is None or rss < best[0]:
            best = (rss, res.x[0], res.x[1], y_ori_cand, y_ter_cand)

    if best is None:
        y_flat = np.repeat(np.mean(y), n)
        print("OTR bias not detected (no seed converged)")
        return y, y_flat, o_idx_seed, t_idx_seed, False

    _, x_ori_opt, x_ter_opt, y_ori_opt, y_ter_opt = best

    # Orient the labels by the fitted anchor values: the origin is whichever
    # anchor came out higher. _otr_concentrated_rss() is symmetric under
    # swapping the two breakpoints -- the same tent, the same residuals, the
    # same RSS -- and the seeds below are blind antipodal pairs, so which one
    # is returned as x_ori is arbitrary. magnitude_ok further down reads y_ori
    # as the PEAK, so without this a perfectly good fit fails the gate on a
    # coin flip: on the p5_75k_exp dataset it found the right breakpoints
    # (1.56 Mb and 3.80 Mb, 48.5% apart) and then rejected them because the
    # ratio came out 0.49 instead of 2.05.
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

    print(f'fitted x_ori:{x_ori_opt:.2f} and x_ter: {x_ter_opt:.2f}')

    # Bias is only corrected if ori/ter are roughly opposite each other on
    # the circular genome (consistent with bidirectional replication) and
    # the fitted anchor-value ratio clears bias_threshold. bias_threshold
    # is left at 1.0 for now (any ori>ter passes); revisit if noise alone
    # is triggering false-positive corrections.
    circular_dist = min(abs(x_ori_opt - x_ter_opt), genome_len - abs(x_ori_opt - x_ter_opt))
    separation_ok = (0.35 * genome_len <= circular_dist <= 0.65 * genome_len)
    magnitude_ok = (y_ter_opt > 0) and ((y_ori_opt / y_ter_opt) > bias_threshold)

    if not (separation_ok and magnitude_ok):
        y_fit = np.repeat(np.mean(y), n)
        print("OTR bias not detected")
        return y, y_fit, o_idx, t_idx, False

    y_fit = otr_predict(x, x_ori_opt, x_ter_opt, y_ori_opt, y_ter_opt, genome_len)
    y_fit = np.clip(y_fit, otr_floor, None)
    y_corr = y / y_fit

    return y_corr, y_fit, o_idx, t_idx, True


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
def fit_otr_bias(df, output):
    """
    Fit stage: runs the median-filter smoothing (if not already present),
    ensures deletion/redundant masking columns exist (running
    mask_coverage_windows() if they don't -- e.g. when OTR correction is
    run without a prior GC-correction pass), and runs otr_fit().

    Returns
    -------
    dict with keys "y_corr", "y_fit", "o_idx", "t_idx", "bias",
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

    y_corr, y_fit, o_idx, t_idx, bias = otr_fit(df)

    return {
        "y_corr": np.asarray(y_corr, dtype=float),
        "y_fit": np.asarray(y_fit, dtype=float),
        "o_idx": o_idx,
        "t_idx": t_idx,
        "bias": bias,
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
    longest = max(per_genome, key=lambda gid: float(per_genome[gid]["win_end"].max()))
    anchor = medians[longest]

    if not np.isfinite(anchor) or anchor <= 0:
        return {gid: float("nan") for gid in per_genome}
    return {gid: medians[gid] / anchor for gid in per_genome}


def _json_safe(value):
    """NaN/inf -> None, so json.dump emits `null` rather than bare `NaN`.

    breseq reads this file with nlohmann/json, which is strict JSON and has no
    allow_nan: a single bare NaN makes the WHOLE file unparseable, so it falls
    into `catch (...)`, warns, and reports no ori-ter bias. That has been true of
    every "Not detected" file CNery has written, since yori/yter are NaN there.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def apply_otr_correction(otr_fit_result, output, deletion_col="is_deletion",
                         relative_copy_number=1.0):
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
    genome_id = str(df["genome_id"][0])
    samplename = output.strip().split("/")[-1] + genome_id
    saveplt = str(output + "/OTR_corr/")
    os.makedirs(saveplt, exist_ok=True)

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
        OTR = yori / (yter + 0.001)
    else:
        xori = df["win_st"].iloc[0]
        xter = df["win_end"].iloc[len(df) - 1]
        yori = np.nan
        yter = np.nan
        OTR = "Not detected"

    results = {
        "Origin window": int(xori),
        "Origin coverage (normalized)": yori,
        "Terminus window": int(xter),
        "Terminus coverage (normalized)": yter,
        "Origin-to-Termius/Bias Ratio": OTR,
        "Relative copy number": relative_copy_number,
        "Correction type": "Ori-ter coordinates fit by coverage",
    }

    df["otr_gc_corr_norm_cov"] = df["gc_corr_norm_cov"].copy()

    if deletion_col in df.columns:
        low = df[deletion_col].to_numpy(dtype=bool)
    else:
        low = (df["read_count_cov"] <= df["read_count_cov"].median() * 0.1).to_numpy()

    # scale everything that's not a genuine deletion (redundant windows included),
    # using otr_fit()'s own y_corr rather than a fresh division by f1.
    df.loc[~low, "otr_gc_corr_norm_cov"] = y_corr[~low]
    df["otr_gc_corr_fact"] = f1

    with open(saveplt + str(samplename) + '_otr_results.json', 'w') as f:
        json.dump({k: _json_safe(v) for k, v in results.items()}, f, indent=4)

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
# Nothing downstream consumes these values yet -- they are reported in their own
# JSON and marked on their own plot, and OTR correction and the HMM are
# unchanged. Worth knowing before wiring them in: the two methods disagree about
# which sequences even have a usable origin, and that is expected. otr_fit needs
# an active replication gradient in the COVERAGE, so it fires on exponential-phase
# samples and not on stationary-phase ones; the skew estimate reads the sequence
# and returns the same answer either way.
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
    sep_lo=0.35,
    sep_hi=0.65,
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


def write_gc_skew_results(result, output, genome_id):
    """Write one reference's skew prediction to GC_skew/<name>_gc_skew_results.json.

    Makes its own directory, as apply_otr_correction() does, so callers and
    tests need not pre-create it.
    """
    samplename = output.strip().split("/")[-1] + str(genome_id)
    savedir = os.path.join(output, "GC_skew")
    os.makedirs(savedir, exist_ok=True)

    path = os.path.join(savedir, f"{samplename}_gc_skew_results.json")
    with open(path, "w") as fh:
        # allow_nan=False: the OTR writer emits bare NaN, which json.load
        # tolerates but is not valid RFC JSON. Every value here is finite, so
        # this stays strict -- and fails loudly if that ever stops being true.
        json.dump(result, fh, indent=4, allow_nan=False)
    return path


def plot_gc_skew(df, output, result):
    """Cumulative GC skew across the reference, with the predicted ori/ter marked."""
    genome_id = str(df["genome_id"].iloc[0])
    samplename = output.strip().split("/")[-1] + genome_id
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


def plot_copy(df_cnv, pltstart, pltend, output):
    
    genome_id = str(df_cnv["genome_id"][0])
    samplename = output.strip().split('/')[-1] + genome_id
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

    plt.figure(figsize=(10, 8))

    fig, ax1 = plt.subplots()

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax2 = ax1.twinx()
    ax1.patch.set_visible(False)

    ax1.set_zorder(2)  # Higher than ax2
    ax2.set_zorder(1)  # Lower than ax1

    
    ax2.scatter(df_plt["win_st"],df_plt["read_count_cov"], color="gray", label="Raw reads",s=10, alpha = 0.2)
    ax2.scatter(df_plt["win_st"],df_plt["otr_gc_corr_rdcnt_cov"], color="orange", label="Corrected reads",s=5, alpha = 0.5,
                marker = mplt.markers.MarkerStyle(marker = 'o', fillstyle = 'none'))
    ax1.scatter(df_plt["win_st"],df_plt["prob_copy_number"], color="red", label="Predicted Copy Number", marker="_", s = 30)

    delta = int(df_plt['read_count_cov'].median()*0.5)
    
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    n_ticks = len(ax1.get_yticks())
    ax2.yaxis.set_major_locator(ticker.LinearLocator(n_ticks))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    
    ax2.set_ylim(int(df_plt['read_count_cov'].min() - delta), int(df_plt['read_count_cov'].max() + delta))
    ax1.set_ylim(int(df_plt['otr_gc_corr_norm_cov'].min() - 1), int(df_plt['otr_gc_corr_norm_cov'].max() + 1))

    
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
                              deletion_coverage_fraction):
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
    for state in range(1, n_states + 1):
        out[:, state] = _nb_logpmf_mu(counts, state * mu * offsets, state * size)

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
            change_rate=DEFAULT_CHANGE_RATE, overlap_weighting=True):
    """
    Viterbi copy-number calling.

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
    genome_id = str(df["genome_id"].iloc[0])
    samplename = output.rstrip("/").split("/")[-1] + genome_id

    new_exp = df.copy()

    new_exp.loc[:, "otr_gc_corr_norm_cov"] = np.nan_to_num(new_exp["otr_gc_corr_norm_cov"].to_numpy())

    med = new_exp["read_count_cov"].median()

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

    csv_full_path = os.path.join(saveloc, f"{samplename}_CNV.csv")

    new_exp = new_exp.reset_index(drop=True)
    new_exp.to_csv(csv_full_path, index=False)

    print(f"{samplename}: Copy number prediction complete. .csv files saved.")

    return new_exp
