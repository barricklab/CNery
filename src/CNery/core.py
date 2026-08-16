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
    through: supplying --file-ending REPLACES the default rather than adding to it.
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


def preprocess(df, win=100, step=100, frag=150):

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
    genome_cyc = list(
        islice(
            cycle(genome),
            int(genome_len * 0.75),
            genome_len + int(genome_len * 1.25)
        )
    )

    fragseq = []
    fragment = []
    winseq = []
    seq = []
    gcp_s = []
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

    # Every full-width window is kept, using TOTAL coverage
    # (unique + redundant) for its median -- so a repeat's real sequencing
    # depth is reflected instead of just the reads that happened to map
    # uniquely there -- and the fraction of redundant-covered bases in the
    # window is recorded as `pct_redundant`. Downstream,
    # mask_coverage_windows() turns `pct_redundant` into `is_redundant`,
    # which censors the window from GC/OTR bias-model FITTING, from the
    # origin/terminus peak-trough SEARCH (see otr_fit()) and from the
    # Viterbi observation sequence (see run_HMM()), while it still receives
    # a real bias-corrected coverage value and inherits the copy number of
    # the segment it sits in -- so a repeat neither invents an
    # amplification nor breaks a genuine deletion in two.
    while (i <= (genome_len - 1)) and (lst_win < genome_len):

        win_full_cov = df_b2c["unique_cov"].iloc[i:(i + win)].to_numpy()
        win_redundant_cov = df_b2c["redundant"].iloc[i:(i + win)].to_numpy()
        cov_type = df_b2c["cov_type"].iloc[i:(i + win)].to_numpy()

        winu = len(cov_type)  # bases actually available

        # Only windows backed by the full `win` bases are emitted. A trailing
        # partial window at the genome end would take its median over fewer
        # bases, and its short `win_end` would set the genome-end coordinate
        # used for the final CN segment and for the terminus fallback in
        # apply_otr_correction() -- so it is dropped.
        if winu < win:
            # ...unless no full window fits at all, i.e. the reference is
            # shorter than `win` (a small plasmid or contig). Then this
            # partial window is all there is, and returning an empty frame
            # would be worse than returning a short one.
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
        window.insert(i, i)
        win_end.insert(i, i + winu)
        lst_win = win_end[(len(win_end) - 1)]
        i_off = i + int(genome_len * 0.25)

        # If fragment size is greater than the window size calculate the
        # GC% of the entire fragment covering the coverage window
        if (frag > win):
            diff = int((frag - win) / 2)
            fragseq = genome_cyc[(i_off - diff):((i_off + win) + diff)]
            fragment.insert(i, ''.join(str(element) for element in fragseq))
            gcc = ''.join([nucleotide for nucleotide in fragseq
                           if nucleotide in ['C', 'G']])
            gccp = (len(gcc) / len(fragseq))
            gcp_s.insert(i, gccp)
        # Otherwise use the length of the window to calculate the GC%
        else:
            diff = int((win - frag) / 2)
            fragseq = list(
                genome_cyc[i_off - diff:(i_off + win) + diff]
            )
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
    frag=150,
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


def circular_arc(start, end, n):
    """
    Real genome positions and "unwrapped" x-coordinates for the arc walking
    forward from `start` to `end` around a circular genome of length `n`,
    wrapping past n-1 back to 0 if `end` < `start`. Unwrapped x keeps
    increasing past `n` on wraparound so a line can be fit against it
    without a false discontinuity at the FASTA coordinate 0/n boundary.
    """
    if end >= start:
        unwrapped = np.arange(start, end + 1)
    else:
        unwrapped = np.arange(start, end + 1 + n)
    real_positions = unwrapped % n
    return real_positions, unwrapped


def fit_arc_line(ux, y_arc, fit_mask, fallback_value):
    """OLS line through an arc's real (masked) points; falls back to a flat
    line at `fallback_value` if too few clean points remain to fit."""
    if fit_mask.sum() >= 2:
        slope, intercept = np.polyfit(ux[fit_mask], y_arc[fit_mask], 1)
    else:
        slope, intercept = 0.0, fallback_value
    return slope, intercept


#Fit the coverage based on the presence and the degree of origin and terminus biased read counts observed
#
# Detects the origin (coverage peak) and terminus (coverage trough) of
# replication from the median-filtered coverage profile, then fits a
# straight line through each of the two circular arcs connecting them
# (ordinary least squares, using every window in the arc). Windows
# flagged as deletions or redundant coverage (via mask_coverage_windows())
# are excluded from the line fits so they cannot distort the ramp.
def otr_fit(df, bias_threshold=1.0):

    x = df.index.to_numpy()
    y = df["gc_corr_norm_cov"].to_numpy(dtype=float)
    y_med_fil = df["gc_cor_med_fil"].to_numpy(dtype=float)
    n = len(x)

    # Windows to exclude from the arc line fits: genuine deletions and
    # redundant/repeat-coverage windows, if mask_coverage_windows() has
    # already flagged them.
    exclude = np.zeros(n, dtype=bool)
    if "is_deletion" in df.columns:
        exclude |= df["is_deletion"].to_numpy(dtype=bool)
    if "is_redundant" in df.columns:
        exclude |= df["is_redundant"].to_numpy(dtype=bool)

    # For the peak/trough SEARCH specifically, dilate the exclusion by one
    # window on each side (circularly -- np.roll wraps correctly around a
    # circular chromosome).
    exclude_for_search = exclude | np.roll(exclude, 1) | np.roll(exclude, -1)

    _yv = y[np.isfinite(y) & (y > 0)]
    _yref = np.median(_yv) if _yv.size else 1.0
    otr_floor = 0.1 * _yref if _yref > 0 else 1e-6

    # Peak/trough search excludes masked windows: -inf so they can never
    # win the argmax, +inf so they can never win the argmin. If EVERY
    # window happens to be masked (degenerate edge case), fall back to the
    # unmasked profile rather than crashing on an all-inf array.
    if exclude_for_search.all():
        y_for_max = y_med_fil
        y_for_min = y_med_fil
    else:
        y_for_max = np.where(exclude_for_search, -np.inf, y_med_fil)
        y_for_min = np.where(exclude_for_search, np.inf, y_med_fil)

    o_idx = int(np.nanargmax(y_for_max))
    t_idx = int(np.nanargmin(y_for_min))
    y_ori = y[o_idx]
    y_ter = y[t_idx]

    print(f'o_idx:{o_idx} and t_idx: {t_idx}')

    # Bias is only corrected if ori/ter are roughly opposite each other on
    # the circular genome (consistent with bidirectional replication) and
    # the peak/trough coverage ratio clears bias_threshold. bias_threshold
    # is left at 1.0 for now (any ori>ter passes); revisit if noise alone
    # is triggering false-positive corrections.
    circular_dist = min(abs(o_idx - t_idx), n - abs(o_idx - t_idx))
    separation_ok = (0.35 * n <= circular_dist <= 0.65 * n)
    magnitude_ok = (y_ter > 0) and ((y_ori / y_ter) > bias_threshold)

    if not (separation_ok and magnitude_ok):
        y_fit = np.repeat(np.mean(y), n)
        print("OTR bias not detected")
        return y, y_fit, o_idx, t_idx, False

    bias = True

    # Split the circle into the two arcs connecting origin and terminus.
    pos_a, ux_a = circular_arc(o_idx, t_idx, n)
    pos_b, ux_b = circular_arc(t_idx, o_idx, n)

    y_a = y[pos_a]
    y_b = y[pos_b]
    fit_mask_a = ~exclude[pos_a]
    fit_mask_b = ~exclude[pos_b]

    slope_a, intercept_a = fit_arc_line(ux_a, y_a, fit_mask_a, y_ori)
    slope_b, intercept_b = fit_arc_line(ux_b, y_b, fit_mask_b, y_ter)

    y_fit_a = slope_a * ux_a + intercept_a
    y_fit_b = slope_b * ux_b + intercept_b

    y_fit = np.empty(n, dtype=float)
    y_fit[pos_a] = y_fit_a
    # pos_a and pos_b share their two endpoints (o_idx, t_idx); both arcs'
    # lines are anchored near the true peak/trough there, so assigning
    # arc B's value last at the shared points is a negligible difference.
    y_fit[pos_b] = y_fit_b

    y_fit = np.clip(y_fit, otr_floor, None)
    y_corr = y / y_fit

    return y_corr, y_fit, o_idx, t_idx, bias


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# ---------------------------------------------------------------------------
# mask -> fit -> apply split for origin-to-terminus (OTR) bias.
# otr_fit() (above) now does its own masking internally (via the
# is_deletion/is_redundant columns), so this only needs to ensure those
# columns exist before calling it.
# ---------------------------------------------------------------------------
def fit_otr_bias(df, output):
    """
    Fit stage: runs the median-filter smoothing (if not already present),
    ensures deletion/redundant masking columns exist (running
    mask_coverage_windows() if they don't -- e.g. when OTR correction is
    run without a prior GC-correction pass), and runs otr_fit().

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


def apply_otr_correction(otr_fit_result, output, deletion_col="is_deletion"):
    """
    Windows flagged `deletion_col` are left un-scaled at their GC-corrected value. 
    Redundant coverage windows, DO receive the OTR scaling factor, so they are not
    zeroed or frozen and won't be miscalled as deletions by the HMM.
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
        json.dump(results, f, indent=4)

    return df, xori, xter


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


def log_emission_with_offsets(counts, offsets, mu, size, n_states, error_rate):
    """(n_obs, n_states + 1) log emission matrix with per-window bias offsets.

    State index == copy number: row 0 is the geometric zero state, and row k is
    NegBinom(mu = k * mu * offset_i, size = k * size). Scaling `size` with the
    state alongside the mean keeps variance proportional to copy number, which
    is what the old `variance * (state + 1)` did -- the offsets are the change
    here, not the across-state behaviour.

    A per-window matrix replaces the old (state, count) lookup table because
    the offsets vary per window, which no shared table can express. It also
    removes the table's `absmax` ceiling and the read-count clipping that fed it.
    """
    counts = np.asarray(counts, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    out = np.empty((counts.size, n_states + 1), dtype=float)
    out[:, 0] = geom.logpmf(counts + 1, 1 - error_rate)
    for state in range(1, n_states + 1):
        out[:, state] = _nb_logpmf_mu(counts, state * mu * offsets, state * size)

    out[~np.isfinite(out)] = -np.inf
    return out

#Emission Matrix
def setup_emission_matrix(n_states, mean, variance, absmax, error_rate):
    emission = np.full((n_states, absmax + 1), np.nan)
    
    for state in range(n_states):
        pr = solve_pr(mean * (state + 1), variance * (state + 1))
        p, r = pr[0], pr[1]
        
        for obs in range(absmax + 1):
            emission[state, obs] = calculate_prob(p, r, obs)
    
    # error rate offsets the probability threshold of 
    # predicting zero at erronous read alignments
    obs_range = np.arange(absmax + 1)
    zero_row = geom.pmf(obs_range + 1, 1 - error_rate)
    emission = np.vstack((zero_row, emission))
    # np.savetxt("emission.csv", emission, delimiter=",")  
    return emission

#: Prior probability per BASE that copy number changes. 1e-6 is one boundary per
#: ~1 Mb, against the ~25-40 segments these genomes actually carry.
DEFAULT_CHANGE_RATE = 1e-6


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


def run_HMM(df, output, error_rate=0.15, n_states=5, changeprob=None,
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
    inflates the variance -- pushing `n_states` up until pile-ups get their
    own spurious high-CN segments -- and lets a single repeat window break a
    genuine deletion in two.

    They are not dropped from the frame. Each still carries its
    bias-corrected coverage and inherits `prob_copy_number` from the segment
    it falls in, and `is_redundant` is written to the CNV.csv alongside so an
    inherited call can be told from a real one.

    `min_called_windows` is a floor: if censoring would leave fewer windows
    than this, every window is used instead. That keeps small references and
    the synthetic test fixtures -- which carry no `is_redundant` column at
    all -- behaving exactly as before. It governs the OBSERVATION SEQUENCE
    only: the emission fit excludes every window with any redundant coverage
    unconditionally, because a window that merely clips the edge of an IS
    element sits at 1.2-1.4x, inside the fit's censoring window, where it would
    inflate the dispersion precisely where sharpness matters most.
    """

    saveloc = os.path.join(output, "CNV_csv")
    genome_id = str(df["genome_id"].iloc[0])
    samplename = output.rstrip("/").split("/")[-1] + genome_id

    new_exp = df.copy()

    new_exp.loc[:, "otr_gc_corr_norm_cov"] = np.nan_to_num(new_exp["otr_gc_corr_norm_cov"].to_numpy())

    med = new_exp["read_count_cov"].median()

    rc_cap = int(max(1.0, max_copy_number) * med)

    # Back-converted read counts are computed for EVERY window -- the column
    # is part of the pipeline contract and feeds the diagnostic plots.
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
        n_states=n_states, error_rate=error_rate,
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

    # HMM_copy_number indexes win_st/win_end positionally, so the censored
    # subset can be passed straight through. chr_length stays the genome end
    # over ALL windows -- the last segment must reach it even if the final
    # window is censored.
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

    # Assign by window index rather than by appending to a flat list: once
    # censored windows are absent from the observation sequence a segment can
    # span windows that never voted for it, so the old
    # len(CN_HMM) == len(new_exp) invariant no longer holds.
    CN_HMM = pd.Series(np.nan, index=new_exp.index, dtype=float)

    for cnrow in copy_numbers.itertuples():
        in_segment = (
            (new_exp["win_st"] >= int(cnrow.Startpos))
            & (new_exp["win_st"] < int(cnrow.Endpos))
        )
        CN_HMM[in_segment] = int(cnrow.State)

    # Windows on a segment boundary can fall outside every half-open interval;
    # carry the neighbouring call across rather than leaving a hole.
    new_exp.loc[:, "prob_copy_number"] = CN_HMM.ffill().bfill().astype(int)

    csv_full_path = os.path.join(saveloc, f"{samplename}_CNV.csv")

    new_exp = new_exp.reset_index(drop=True)
    new_exp.to_csv(csv_full_path, index=False)

    print(f"{samplename}: Copy number prediction complete. .csv files saved.")

    return new_exp
