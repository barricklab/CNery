#!/usr/bin/env python
# coding: utf-8
import argparse
import os
from pathlib import Path

from .core import (
    DEFAULT_CHANGE_RATE,
    relative_copy_numbers,
    DEFAULT_DELETION_COVERAGE_FRACTION,
    DEFAULT_FILE_ENDINGS,
    parse_region,
    process_multi_genome,
    resolve_coverage_inputs,
    fit_otr_bias,
    apply_otr_correction,
    plot_correction_stages,
    plot_gc_passes,
    second_gc_pass,
    plot_otr_corr,
    predict_ori_ter_from_skew,
    write_gc_skew_results,
    plot_gc_skew,
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
        default=100,
        type=int,
        help=(
            "Define window length to parse through the genome and calculate "
            "coverage and GC statistics. Default: 100. Wider windows smooth "
            "the coverage but lose short events: the window statistic is a "
            "per-base median, whose precision grows sublinearly with width, so "
            "-w is a resolution knob."
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
            "if non-overlapping windows. Default: 100, i.e. non-overlapping. "
            "Copy-number calls are near-invariant to this: the state-change "
            "prior is per base (see --change-rate) and overlapping windows are "
            "down-weighted so they do not count the same bases twice."
        ),
    )

    parser.add_argument(
        "-f",
        "--frag-size",
        action="store",
        dest="f",
        default=400,
        required=False,
        type=int,
        help=(
            "Average fragment size of the sequencing library. GC%% is measured "
            "over this many bases centred on each window, because GC bias acts at "
            "fragment scale rather than at whatever window size was asked for. "
            "Ignored when it is smaller than -w, which is then used instead. "
            "Default: 400."
        ),
    )
    parser.add_argument(
        "-z",
        "--deletion-coverage-fraction",
        action="store",
        dest="deletion_coverage_fraction",
        default=DEFAULT_DELETION_COVERAGE_FRACTION,
        required=False,
        type=float,
        help=(
            "Coverage a deleted region still shows, as a fraction of the "
            "single-copy level. Sets the mean of the copy-number-0 emission. "
            "Real deletions are not empty -- mismapping and repeat spill leave "
            "a couple of percent behind. A fraction rather than an absolute "
            "depth so that what counts as a deletion does not change with how "
            "deeply the sample was sequenced. Default: %g."
            % DEFAULT_DELETION_COVERAGE_FRACTION
        ),
    )
    parser.add_argument(
        "--change-rate",
        action="store",
        dest="change_rate",
        default=DEFAULT_CHANGE_RATE,
        required=False,
        type=float,
        help=(
            "Prior probability PER BASE that copy number changes. The "
            "per-window probability is 1 - exp(-rate * step-size), so changing "
            "-w/-s no longer changes the implied biology. Read 1/rate as the "
            "expected segment length: the default 1e-06 is one copy-number "
            "boundary per megabase. Larger values give more, shorter segments."
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

    out_subdirs = ['/CNV_plt', '/CNV_csv', '/GC_bias', '/OTR_corr', '/GC_skew']
    for sub in out_subdirs:
        Path(out_dir + sub).mkdir(parents=True, exist_ok=True)

    # Origin and terminus of replication are always inferred from the coverage profile

    # process every coverage table given in one pass
    per_genome = process_multi_genome(
        coverage_inputs,
        output_prefix=out_dir,
        win=options.w,
        step=options.s,
        frag=options.f,
    )
    # process_multi_genome already:
    #   - reads and preprocesses each coverage table
    #   - pools all genomes, masks redundant/deletion windows, and does
    #     LOWESS GC correction (mask_coverage_windows -> fit_gc_bias ->
    #     apply_gc_correction)
    #   - plots pooled GC bias
    #   - returns {genome_id: df_gc_corrected_per_genome}, each already
    #     carrying is_deletion/is_redundant columns from the GC-stage mask

    smpl = out_dir.strip().split('/')[-1]
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

    def _report_otr(smpl, genome_id, otr_fit_result):
        """Say whether OTR fired, on whose coordinates, and on what evidence.

        Named separately from the fit because "corrected OTR bias" is no longer
        the only outcome worth printing: the correction can now come from the
        GC skew rather than the coverage, and a rejection carries a p-value that
        explains itself.
        """
        detail = otr_fit_result.get("detail") or {}
        p = detail.get("Coverage fit p-value")
        if not otr_fit_result["bias"]:
            print(
                f"{smpl} ({genome_id}): no origin/terminus bias corrected "
                f"(coverage fit p={p}); see OTR_corr/ for the evidence."
            )
            return
        source = detail.get("Breakpoint source", "coverage fit")
        p_lr = detail.get("Coverage vs skew likelihood-ratio p-value")
        extra = "" if p_lr is None else f", likelihood ratio p={p_lr}"
        print(
            f"{smpl} ({genome_id}): corrected origin/terminus of replication "
            f"(OTR) bias using the {source}{extra}."
        )

    # Bias-correction and CNV calling per genome
    # One number per sequence: its coverage relative to the longest sequence in this
    # run, which reads exactly 1.0. Computed here because it is the only place holding
    # every sequence at once -- apply_otr_correction() runs per sequence.
    relative_cn = relative_copy_numbers(per_genome)

    # {genome_id: corrected frame + its OTR result}, filled by pass A below.
    staged = {}

    for genome_id, df_b2c in per_genome.items():
        print(f"Processing genome: {genome_id}")

        # Origin/terminus from the reference's own cumulative GC skew. This sits
        # ahead of the --bias branch on purpose: it reads only the sequence, so
        # unlike the coverage-derived OTR results it is available in all four
        # modes, including the ones that never call apply_otr_correction().
        #
        # fit_otr_bias() takes it as a second candidate: when the coverage fit
        # cannot clear its own significance gate, or when it can but a
        # likelihood-ratio test says it is no better than a tent hinged here,
        # these coordinates supply the correction instead. Under --bias gc/none
        # nothing consumes it and it is still reported.
        skew_result = predict_ori_ter_from_skew(
            df_b2c, win=options.w, step=options.s
        )
        write_gc_skew_results(skew_result, out_dir, genome_id)
        plot_gc_skew(df_b2c, out_dir, skew_result)
        if skew_result["Prediction confident"]:
            print(
                f"{smpl} ({genome_id}): GC skew predicts origin "
                f"{skew_result['Origin (bp)']}, terminus "
                f"{skew_result['Terminus (bp)']}."
            )
        else:
            print(
                f"{smpl} ({genome_id}): GC skew origin/terminus prediction is "
                f"low confidence; see GC_skew/ for the values and curve."
            )

        # Pass A collects the bias-corrected frame for every sequence. The HMM
        # and the plots deliberately do NOT run here: the second GC pass below is
        # POOLED, so it cannot be fitted until every sequence has been
        # OTR-corrected, and nothing downstream should see coverage that is about
        # to change.
        if options.bias == "gc":
            df_corr = df_b2c.copy()
            df_corr["otr_gc_corr_norm_cov"] = df_corr["gc_corr_norm_cov"]
            otr_fit_result = ori_win = ter_win = None
            print(f"{smpl} ({genome_id}): GC bias vs coverage handled (pooled fit).")

        elif options.bias == "none":
            df_corr = df_b2c.copy()
            df_corr["otr_gc_corr_norm_cov"] = df_corr["norm_raw_cov"]
            otr_fit_result = ori_win = ter_win = None

        else:
            # "otr" and "all" differ only in what the OTR fit is given as its
            # baseline: --bias otr aliases gc_corr_norm_cov to the RAW coverage,
            # so the GC correction is not applied at all in that mode.
            df_in = df_b2c.copy()
            if options.bias == "otr":
                df_in["gc_corr_norm_cov"] = df_in["norm_raw_cov"]
            else:
                print(f"{smpl} ({genome_id}): GC bias vs coverage handled "
                      f"(pooled fit).")
            otr_fit_result = fit_otr_bias(df_in, out_dir, skew_result=skew_result)
            df_corr, ori_win, ter_win = apply_otr_correction(
                otr_fit_result, out_dir,
                relative_copy_number=relative_cn.get(genome_id, 1.0),
            )
            _report_otr(smpl, genome_id, otr_fit_result)

        staged[genome_id] = {"df": df_corr, "otr": otr_fit_result,
                             "ori": ori_win, "ter": ter_win}

    # ---- Pooled second GC pass ------------------------------------------------
    #
    # The OTR tent varies with position, and position correlates with GC, so
    # dividing by it puts a GC trend back into coverage the GC stage had removed
    # -- measured at 10.4% of coverage on ltee_ara_p5_75k_exp and 3.6% on
    # adp1_mgd06_lb. Refitting GC on the OTR-corrected coverage removes it
    # (1.0% and 0.9%), and because both passes are functions of GC alone they
    # compose exactly into one total curve.
    #
    # Pooled for the same reason the first pass is: GC bias belongs to the
    # sequencing chemistry, not to any one reference. That is what forces the
    # two-pass structure above -- the fit needs every sequence corrected first.
    #
    # Only under --bias all. Under gc/none there is no OTR stage to reintroduce
    # anything, and under otr the GC correction was explicitly opted out of, so
    # quietly applying one here would ignore the flag.
    if options.bias == "all" and staged:
        frames, _gc2 = second_gc_pass({g: st["df"] for g, st in staged.items()})
        for genome_id in staged:
            staged[genome_id]["df"] = frames[genome_id]
        gc_passes_path = plot_gc_passes(frames, out_dir)
        print(f"{smpl}: second GC pass fitted across {len(frames)} "
              f"sequence(s); total GC curve in "
              f"{os.path.basename(gc_passes_path)}.")

    # ---- Pass B: plots and copy-number calling, on the final coverage ---------
    for genome_id, st in staged.items():
        df_final, otr_fit_result = st["df"], st["otr"]
        if otr_fit_result is not None:
            plot_otr_corr(df_final, output=out_dir, ori=st["ori"], ter=st["ter"])
            print(f"{smpl} ({genome_id}): OTR bias vs coverage plots saved.")
        plot_correction_stages(df_final, out_dir, otr_fit_result, bias=options.bias)
        df_cnv = run_HMM(
            df_final,
            out_dir,
            deletion_coverage_fraction=options.deletion_coverage_fraction,
            bias=options.bias,
            change_rate=options.change_rate,
        )
        emit_cnv_plot(df_cnv, genome_id)


if __name__ == "__main__":
    main()
