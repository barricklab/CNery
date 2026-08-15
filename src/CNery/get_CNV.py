#!/usr/bin/env python
# coding: utf-8
import argparse
from pathlib import Path

from .core import (
    DEFAULT_FILE_ENDINGS,
    parse_region,
    process_multi_genome,
    resolve_coverage_inputs,
    fit_otr_bias,
    apply_otr_correction,
    plot_otr_corr,
    run_HMM,
    plot_copy,
)


def main():
    from argparse import RawTextHelpFormatter
    import textwrap

    parser = argparse.ArgumentParser(
        description=(
            "CNery is python package extension to breseq that analyzes the "
            "sequencing coverage across the genome to predict copy number "
            "variation (CNV)"
        ),
        epilog=textwrap.dedent(
            "Inputs are breseq 'bam2cov' coverage tables (CSV or TSV). Run with no "
            "arguments in a folder that holds them, or name files and/or folders "
            "directly. \n"
        ),
        formatter_class=RawTextHelpFormatter,
    )

    # Define the command line arguments
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="INPUT",
        help=(
            "Coverage table files, and/or folders containing them. Folders are "
            "searched (top level only) for files ending in --file-ending. Every "
            "table given is analyzed together, sharing one GC-bias fit, so these "
            "should be the reference sequences of a single sample. "
            "Defaults to the current folder."
        ),
    )

    parser.add_argument(
        "--file-ending",
        action="append",
        dest="file_ending",
        metavar="ENDING",
        help=(
            "File ending that identifies a coverage table inside an input folder. "
            "Repeat the flag to accept more than one. Any --file-ending REPLACES "
            "the defaults ("
            + ", ".join(f"'{e}'" for e in DEFAULT_FILE_ENDINGS)
            + ") rather than adding to them. A file named directly on the command "
            "line is always used, whatever it is called."
        ),
    )

    parser.add_argument(
        "--region",
        action="append",
        dest="reg",
        required=False,
        type=str,
        metavar="SEQ_ID:START-END",
        help=(
            "Plot the CNV calls for one sequence over a genomic segment, e.g. "
            "'REL606:3497890-3955678'. The sequence ID is the one derived from "
            "the table's file name. Repeat the flag to plot several sequences, "
            "at most once each. 'SEQ_ID:' may be omitted when the run has only "
            "one input sequence. Open intervals are accepted: "
            "'REL606:3497890-' runs to the end of the sequence, "
            "'REL606:-3955678' from its start. Giving any --region also selects "
            "WHICH sequences are plotted: those not named get no CNV plot. This "
            "affects plotting only -- coverage, bias fitting and copy-number "
            "calling always cover every sequence, and the output CSVs always "
            "contain every window."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        dest="o",
        required=False,
        type=str,
        help=(
            "output file prefix / storage location. "
            "Defaults to the 'CNV_out' folder in current dir."
        ),
    )

    parser.add_argument(
        "-w",
        "--window",
        action="store",
        dest="w",
        required=False,
        default=200,
        type=int,
        help=(
            "Define window length to parse through the genome and calculate "
            "coverage and GC statistics."
        ),
    )

    parser.add_argument(
        "-s",
        "--step-size",
        action="store",
        dest="s",
        required=False,
        default=100,
        type=int,
        help=(
            "Define step size (<= window size) for each progression of the "
            "window across the genome sequence. Set step-size=window-size "
            "if non-overlapping windows."
        ),
    )

    parser.add_argument(
        "-f",
        "--frag-size",
        action="store",
        dest="f",
        default=150,
        required=False,
        type=int,
        help="Average fragment size of the sequencing reads.",
    )
    parser.add_argument(
        "-e",
        "--error-rate",
        action="store",
        dest="e",
        default=0.15,
        required=False,
        type=float,
        help=(
            "Approximate error rate in sequencing read coverage/reference "
            "alignment. Widens the negative-binomial emission distributions "
            "in the HMM. Default: 0.15."
        ),
    )
    parser.add_argument(
        "--bias",
        choices=["all", "none", "gc", "otr"],
        default="all",
        required=False,
        help=(
            "Select specific bias correction (only OTR or only GC) to run "
            "before CN prediction."
        ),
    )

    # Parse the command line arguments
    options = parser.parse_args()

    inputs = options.inputs if options.inputs else ["."]

    # Resolve and validate inputs BEFORE creating anything on disk, so a typo or a
    # folder with no tables in it fails without leaving an empty CNV_out/ tree behind.
    coverage_inputs = resolve_coverage_inputs(inputs, options.file_ending)

    # {genome_id: (start, end)}. Empty means "no --region given" -- plot everything
    # whole. Non-empty also selects which sequences get plotted at all.
    #
    # A bad --region used to `return` a message from main(), which exited 0 and did no
    # work -- indistinguishable from success. parser.error() prints usage and exits 2.
    regions = {}
    available = ", ".join(coverage_inputs)
    for text in (options.reg or []):
        try:
            seq, start, end = parse_region(text)
        except ValueError as exc:
            parser.error(f"--region: {exc}")

        # The sequence has to be named unless there is only one it could mean.
        # Applying one coordinate range to every reference is what made --region
        # crash on a chromosome-plus-plasmid run: the range falls outside the
        # shorter sequence, the plot slice comes out empty, and its median is NaN.
        if seq is None:
            if len(coverage_inputs) > 1:
                parser.error(
                    f"--region {text!r} does not say which sequence it refers to, "
                    f"and this run has {len(coverage_inputs)}: {available}. "
                    "Use SEQ_ID:START-END."
                )
            seq = next(iter(coverage_inputs))
        elif seq not in coverage_inputs:
            parser.error(
                f"--region names sequence {seq!r}, which is not among the "
                f"inputs: {available}."
            )

        if seq in regions:
            parser.error(
                f"--region given more than once for {seq!r}. Each sequence gets one "
                "plot, so it can carry only one region."
            )
        regions[seq] = (start, end)

    if options.o is not None:
        out_dir = options.o
    else:
        out_dir = "CNV_out/"

    out_subdirs = ["/CNV_plt", "/CNV_csv", "/GC_bias", "/OTR_corr"]
    for sub in out_subdirs:
        Path(out_dir + sub).mkdir(parents=True, exist_ok=True)

    # Origin and terminus of replication are always inferred from the coverage profile
    # print(
    #     "Origin/terminus of replication will be inferred from the "
    #     "coverage profile."
    # )

    # process single or multiple genomes in a unified way
    per_genome = process_multi_genome(
        coverage_inputs,
        output_prefix=out_dir,
        win=options.w,
        step=options.s,
        frag=options.f,
    )

    smpl = out_dir.strip().split("/")[-1]
    print(
        "Calculating coverage and GC% across sliding windows for each "
        "reference sequence"
    )
    def emit_cnv_plot(df_cnv, genome_id):
        """Plot this sequence's calls, unless --region selected other sequences."""
        if regions and genome_id not in regions:
            print(
                f"{smpl} ({genome_id}): not named in --region; CNV plot skipped."
            )
            return
        start, end = regions.get(genome_id, (0, 0))
        plot_copy(df_cnv, start, end, output=out_dir)
        print(f"{smpl} ({genome_id}): CNV prediction plots saved.")

    # Bias-correction and CNV calling per genome
    for genome_id, df_b2c in per_genome.items():
        print(f"Processing genome: {genome_id}")

        if options.bias == "gc":
            # df_b2c already GC-corrected by pooled LOWESS
            df_gc = df_b2c.copy()
            print(
                f"{smpl} ({genome_id}): GC bias vs coverage handled "
                f"(pooled fit)."
            )
            df_gc["otr_gc_corr_norm_cov"] = df_gc["gc_corr_norm_cov"]
            df_cnv = run_HMM(df_gc, out_dir, error_rate=options.e)
            emit_cnv_plot(df_cnv, genome_id)

        elif options.bias == "otr":
            # Use raw norm_raw_cov as baseline for OTR-only correction
            df_otr_in = df_b2c.copy()
            df_otr_in["gc_corr_norm_cov"] = df_otr_in["norm_raw_cov"]
            # fit_otr_bias() then apply_otr_correction() replaces the old
            # single-call otr_correction(df_otr_in, out_dir).
            otr_fit_result = fit_otr_bias(df_otr_in, out_dir)
            df_otr, ori_win, ter_win = apply_otr_correction(otr_fit_result, out_dir)
            print(
                f"{smpl} ({genome_id}): Corrected origin/terminus of "
                f"replication (OTR) bias in coverage."
            )
            plot_otr_corr(df_otr, output=out_dir, ori=ori_win, ter=ter_win)
            print(f"{smpl} ({genome_id}): OTR bias vs coverage plots saved.")
            df_cnv = run_HMM(df_otr, out_dir, error_rate=options.e)
            emit_cnv_plot(df_cnv, genome_id)

        elif options.bias == "none":
            df_none = df_b2c.copy()
            df_none["otr_gc_corr_norm_cov"] = df_none["norm_raw_cov"]
            df_cnv = run_HMM(df_none, out_dir, error_rate=options.e)
            emit_cnv_plot(df_cnv, genome_id)

        elif options.bias == "all":
            # df_b2c already has GC correction applied
            df_gc = df_b2c.copy()
            print(
                f"{smpl} ({genome_id}): GC bias vs coverage handled "
                f"(pooled fit)."
            )
            # Same fit -> apply split as the "otr" branch above, replacing
            # otr_correction(df_gc, out_dir).
            otr_fit_result = fit_otr_bias(df_gc, out_dir)
            df_otr, ori_win, ter_win = apply_otr_correction(otr_fit_result, out_dir)
            print(
                f"{smpl} ({genome_id}): Corrected origin/terminus of "
                f"replication (OTR) bias in coverage."
            )
            plot_otr_corr(df_otr, output=out_dir, ori=ori_win, ter=ter_win)
            print(f"{smpl} ({genome_id}): OTR bias vs coverage plots saved.")
            df_cnv = run_HMM(df_otr, out_dir, error_rate=options.e)
            emit_cnv_plot(df_cnv, genome_id)


if __name__ == "__main__":
    main()
