#!/usr/bin/env python
# coding: utf-8
import argparse
from pathlib import Path

from .core import (
    DEFAULT_FILE_ENDING,
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
            "Inputs are breseq 'bam2cov --format TSV' coverage tables. Run with no "
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
            f"the default ('{DEFAULT_FILE_ENDING}') rather than adding to it. "
            "A file named directly on the command line is always used, whatever "
            "it is called."
        ),
    )

    parser.add_argument(
        "-reg",
        action="store",
        dest="reg",
        required=False,
        type=str,
        help="select the region of the genome to evaluate",
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
        "--frag_size",
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
        default=0.05,
        required=False,
        type=float,
        help=(
            "Approximate error rate in sequencing read coverage/refrence "
            "alignment."
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

    if options.o is not None:
        out_dir = options.o
    else:
        out_dir = "CNV_out/"

    out_subdirs = ["/CNV_plt", "/CNV_csv", "/GC_bias", "/OTR_corr"]
    for sub in out_subdirs:
        Path(out_dir + sub).mkdir(parents=True, exist_ok=True)

    region = options.reg

    if region is not None:
        parts = region.split("-")
        if len(parts) == 2:
            if parts[0] == "":
                pltend = int(parts[1])
                pltstart = 0
            elif parts[1] == "":
                pltstart = int(parts[0])
                pltend = 0
            else:
                pltstart, pltend = int(parts[0]), int(parts[1])
        else:
            return (
                "Invalid region. Ensure the region is specified by int values "
                "of 2 genomic coordinates separated by a '-'."
            )
    else:
        pltstart, pltend = 0, 0

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
            df_cnv = run_HMM(df_gc, out_dir)
            plot_copy(df_cnv, pltstart, pltend, output=out_dir)
            print(f"{smpl} ({genome_id}): CNV prediction plots saved.")

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
            df_cnv = run_HMM(df_otr, out_dir)
            plot_copy(df_cnv, pltstart, pltend, output=out_dir)
            print(f"{smpl} ({genome_id}): CNV prediction plots saved.")

        elif options.bias == "none":
            df_none = df_b2c.copy()
            df_none["otr_gc_corr_norm_cov"] = df_none["norm_raw_cov"]
            df_cnv = run_HMM(df_none, out_dir)
            plot_copy(df_cnv, pltstart, pltend, output=out_dir)
            print(f"{smpl} ({genome_id}): CNV prediction plots saved.")

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
            df_cnv = run_HMM(df_otr, out_dir)
            plot_copy(df_cnv, pltstart, pltend, output=out_dir)
            print(f"{smpl} ({genome_id}): CNV prediction plots saved.")


if __name__ == "__main__":
    main()
