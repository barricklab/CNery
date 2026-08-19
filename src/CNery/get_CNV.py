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
    DEFAULT_FRAG_SIZE,
    apply_frag_size,
    otr_ratio,
    pass1_summary,
    refit_gc_bias_pooled,
    select_frag_size,
    stage_pass1,
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
        default=None,
        required=False,
        type=int,
        help=(
            "Average fragment size of the sequencing library. GC%% is measured "
            "over this many bases centred on each window, because GC bias acts at "
            "fragment scale rather than at whatever window size was asked for. "
            "Ignored when it is smaller than -w, which is then used instead. "
            "LEAVE IT UNSET and the value is chosen from the data: candidate "
            "sizes are scored by how well the GC they imply predicts held-out "
            "coverage, on windows with the replication ramp divided out and the "
            "copy number called 1. That default is kept unless a candidate "
            "beats it by more than the measurement's own standard error, and "
            "the choice is reported. Pass a value to pin it. Default when the "
            "scan declines: 400."
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
    # `-f` unset means "choose it from the data". The scan needs a first pass to
    # control its confounds, so preprocessing starts at the default and the
    # fragment size is revisited once there is a tent and a set of calls.
    scan_frag = options.f is None
    frag_in_use = DEFAULT_FRAG_SIZE if scan_frag else options.f

    per_genome = process_multi_genome(
        coverage_inputs,
        output_prefix=out_dir,
        win=options.w,
        step=options.s,
        frag=frag_in_use,
        collect_gc_flags=scan_frag,
    )
    gc_flags = {}
    if scan_frag:
        per_genome, gc_flags = per_genome
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

    def _report_frag(smpl, detail):
        """Say which fragment size is in force, and on what evidence."""
        if not detail.get("Fragment size scanned"):
            print(f"{smpl}: fragment size {detail['Fragment size']} bp "
                  f"({detail.get('Fragment size reason', 'not scanned')}).")
            return
        chosen = detail["Fragment size"]
        best = detail.get("Fragment size selected")
        print(f"{smpl}: fragment size {chosen} bp -- scanned "
              f"{len(detail['Fragment size candidates'])} candidates, best "
              f"{best}; {detail.get('Fragment size reason', '')}")

    def _fmt_ratio(value):
        return "none" if value is None else f"{value:.4f}"

    def _report_otr(smpl, genome_id, otr_fit_result, pass_no):
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
                f"{smpl} ({genome_id}): pass {pass_no}: no origin/terminus bias "
                f"corrected (coverage fit p={p}); see OTR_corr/ for the evidence."
            )
            return
        source = detail.get("Breakpoint source", "coverage fit")
        p_lr = detail.get("Coverage vs skew likelihood-ratio p-value")
        extra = "" if p_lr is None else f", likelihood ratio p={p_lr}"
        print(
            f"{smpl} ({genome_id}): pass {pass_no}: corrected origin/terminus of "
            f"replication (OTR) bias using the {source}{extra}."
        )

    # ---- Two fitting passes, the first of them provisional -------------------
    #
    # Pass 1 is the pipeline as it always was, plus an HMM whose calls nobody
    # sees. Pass 2 runs the same fits again with everything that HMM did not call
    # CN = 1 censored out of them.
    #
    # The point is that mask_coverage_windows() censors on two crude proxies
    # computed once on UNCORRECTED coverage -- near-zero depth and repeat overlap
    # -- and an amplification is caught by neither, so it goes into the GC LOWESS
    # and the OTR tent at full weight. Measured: adp1_mgd06_lb fits its origin
    # inside its own CN-3 amplification and cwbi_ssym_ht04's chromosome inside
    # its CN-34 one. The HMM knows where those are; nothing else in the pipeline
    # does.
    #
    # One number per sequence: its coverage relative to the longest sequence in
    # this run, which reads exactly 1.0. Computed here because it is the only
    # place holding every sequence at once -- apply_otr_correction() runs per
    # sequence. Recomputed after the pooled GC refit below, which changes the
    # column it reads.
    relative_cn = relative_copy_numbers(per_genome)

    def correct_one(df_in, genome_id, skew_result, second):
        """One sequence through one pass's corrections. Returns (df, otr, ori, ter).

        `second` only changes what is already on the frame -- a wider censor and,
        under the GC modes, a coverage column that now means raw/G. The fits
        themselves are the same calls with the same gates in both passes.
        """
        if options.bias == "gc":
            df_out = df_in.copy()
            df_out["otr_gc_corr_norm_cov"] = df_out["gc_corr_norm_cov"]
            return df_out, None, None, None

        if options.bias == "none":
            df_out = df_in.copy()
            df_out["otr_gc_corr_norm_cov"] = df_out["norm_raw_cov"]
            return df_out, None, None, None

        # "otr" and "all" differ only in what the OTR fit is given as its
        # baseline: --bias otr aliases gc_corr_norm_cov to the RAW coverage, so
        # the GC correction is not applied at all in that mode.
        df_fit = df_in.copy()
        if options.bias == "otr":
            df_fit["gc_corr_norm_cov"] = df_fit["norm_raw_cov"]
        # The median filter seeds the ori/ter guess, so it has to be recomputed
        # from the coverage this pass is actually fitting rather than inherited
        # from the last one.
        df_fit = df_fit.drop(columns=["gc_cor_med_fil"], errors="ignore")

        otr = fit_otr_bias(df_fit, out_dir, skew_result=skew_result)
        extra = _pass1_keys(genome_id) if second else None
        df_out, ori, ter = apply_otr_correction(
            otr, out_dir,
            relative_copy_number=relative_cn.get(genome_id, 1.0),
            extra_results=extra,
        )
        return df_out, otr, ori, ter

    def _pass1_keys(genome_id):
        st = staged[genome_id]
        return pass1_summary(st["otr"], st["df"])

    # ---- Pass 1, and the fragment size it makes choosable --------------------
    skews = {}

    def run_pass1(frames, quiet=False):
        """Every sequence through pass 1. `quiet` is for the provisional run the
        fragment scan needs, whose output nobody should see."""
        out = {}
        for genome_id, df_b2c in frames.items():
            if not quiet:
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
            skew_result = skews.get(genome_id)
            if skew_result is None:
                skew_result = predict_ori_ter_from_skew(
                    df_b2c, win=options.w, step=options.s
                )
                skews[genome_id] = skew_result
            if not quiet:
                write_gc_skew_results(skew_result, out_dir, genome_id)
                plot_gc_skew(df_b2c, out_dir, skew_result)
            if not quiet:
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

            df_corr, otr_fit_result, _ori, _ter = correct_one(
                df_b2c, genome_id, skew_result, second=False
            )
            if not quiet:
                if otr_fit_result is not None:
                    _report_otr(smpl, genome_id, otr_fit_result, pass_no=1)
                elif options.bias == "gc":
                    print(f"{smpl} ({genome_id}): GC bias vs coverage handled "
                          f"(pooled fit).")

            # Provisional calls, written nowhere. Their only job is the censor below.
            df_called = run_HMM(
                df_corr, out_dir,
                deletion_coverage_fraction=options.deletion_coverage_fraction,
                bias=options.bias, change_rate=options.change_rate, write=False,
            )
            df_staged, cn_applied = stage_pass1(df_called)
            n_censored = int(df_staged["is_cn_variant"].sum())
            if quiet:
                pass
            elif cn_applied:
                print(f"{smpl} ({genome_id}): pass 1 complete; {n_censored} window(s) "
                      f"called CN != 1 will be censored from the pass-2 fits.")
            else:
                print(f"{smpl} ({genome_id}): pass 1 complete; CN censor NOT applied "
                      f"-- it would leave under half the windows. Pass 2 censors as "
                      f"pass 1 did.")

            out[genome_id] = {
                "df": df_staged, "otr": otr_fit_result, "skew": skew_result,
                "ratio": otr_ratio(otr_fit_result),
                "cn_applied": cn_applied, "cn_censored": n_censored,
            }
        return out

    staged = run_pass1(per_genome, quiet=scan_frag)

    if scan_frag:
        # The scan is scored on coverage with the ramp divided out and the copy
        # number called 1, which is why it cannot run before now -- and why a
        # changed fragment size invalidates the GC curve pass 1 just fitted, so
        # that pass runs again on the new axis.
        frag_in_use, frag_detail = select_frag_size(
            {g: st["df"] for g, st in staged.items()},
            gc_flags, options.w, DEFAULT_FRAG_SIZE,
        )
        _report_frag(smpl, frag_detail)
        if frag_in_use != DEFAULT_FRAG_SIZE:
            per_genome = apply_frag_size(per_genome, gc_flags, options.w,
                                         frag_in_use)
        staged = run_pass1(per_genome)

    # ---- Pooled GC refit, between the passes ---------------------------------
    #
    # The OTR tent varies with position, and position correlates with GC, so
    # dividing by it puts a GC trend back into coverage the GC stage had removed
    # -- measured at 10.4% of coverage on ltee_ara_p5_75k_exp and 3.6% on
    # adp1_mgd06_lb. Refitting GC on the OTR-corrected coverage removes it, and
    # because both passes are functions of GC alone they compose exactly into one
    # total curve G = g1 * g2.
    #
    # Pooled for the same reason the first pass is: GC bias belongs to the
    # sequencing chemistry, not to any one reference. That is what forces the
    # pass structure -- the fit needs every sequence corrected and called first.
    #
    # Skipped under --bias otr, where the GC correction was explicitly opted out
    # of, and under --bias none, which applies nothing at all.
    if options.bias in ("all", "gc") and staged:
        frames, _gc2 = refit_gc_bias_pooled({g: st["df"] for g, st in staged.items()})
        for genome_id in staged:
            staged[genome_id]["df"] = frames[genome_id]
        gc_passes_path = plot_gc_passes(frames, out_dir)
        print(f"{smpl}: GC bias refitted across {len(frames)} sequence(s) on "
              f"CN=1 windows; total GC curve in "
              f"{os.path.basename(gc_passes_path)}.")
        # gc_corr_norm_cov now means raw/G rather than raw/g1, and this reads it.
        relative_cn = relative_copy_numbers({g: st["df"] for g, st in staged.items()})

    # ---- Pass 2: the same fits, on CN=1 windows ------------------------------
    for genome_id, st in staged.items():
        df_final, otr_fit_result = st["df"], st["otr"]

        if options.bias != "none":
            df_final, otr_fit_result, ori_win, ter_win = correct_one(
                df_final, genome_id, st["skew"], second=True
            )
            if otr_fit_result is not None:
                _report_otr(smpl, genome_id, otr_fit_result, pass_no=2)
                ratio2 = otr_ratio(otr_fit_result)
                if st["ratio"] is not None or ratio2 is not None:
                    print(f"{smpl} ({genome_id}): origin-to-terminus ratio "
                          f"{_fmt_ratio(st['ratio'])} (pass 1) -> "
                          f"{_fmt_ratio(ratio2)} (pass 2).")
                plot_otr_corr(df_final, output=out_dir, ori=ori_win, ter=ter_win)
                print(f"{smpl} ({genome_id}): OTR bias vs coverage plots saved.")

        plot_correction_stages(df_final, out_dir, otr_fit_result, bias=options.bias)
        df_cnv = run_HMM(
            df_final,
            out_dir,
            deletion_coverage_fraction=options.deletion_coverage_fraction,
            bias=options.bias,
            change_rate=options.change_rate,
        )
        if "prob_copy_number_pass1" in df_cnv.columns:
            moved = int((df_cnv["prob_copy_number"].to_numpy()
                         != df_cnv["prob_copy_number_pass1"].to_numpy()).sum())
            print(f"{smpl} ({genome_id}): {moved} window(s) changed copy-number "
                  f"call between the two passes.")
        emit_cnv_plot(df_cnv, genome_id)


if __name__ == "__main__":
    main()
