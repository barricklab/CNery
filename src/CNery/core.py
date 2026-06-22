#!/usr/bin/env python
# coding: utf-8

import os
import json
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import ndimage
import matplotlib as mplt
from scipy.stats import geom
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import  gammaln
from scipy.optimize import minimize
from itertools import cycle, islice


def parse_fasta_records(fastafile):
    """
    Parse a (multi-)FASTA file and return a list of (header, seq_len)
    for each reference sequence.
    """
    records = []
    header = None
    seq_len = 0

    with open(fastafile) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                # Finish previous record
                if header is not None:
                    records.append((header, seq_len))
                header = line[1:]  # drop ">"
                seq_len = 0
            else:
                seq_len += len(line)

    # Last record
    if header is not None:
        records.append((header, seq_len))

    return records


def bam2cov_to_df(
    bamfile,          # path to BAM
    fastafile,        # path to FASTA
    output_prefix,    # output tab file: e.g. "coverage.tab"
    extra_args=None,  # optional, list for any extra CLI args
    region=None       # optional, "REF:1-12345" to avoid re-parsing FASTA
):
    # The tab output file name (may be output_prefix or output_prefix.tab)
    tab_file = output_prefix

    # If no region provided, fall back to original single-genome behavior
    if region is None:
        header = None
        seq_len = 0
        with open(fastafile) as fh:
            for line in fh:
                line = line.rstrip()
                if line.startswith(">"):
                    if header is None:
                        header = line[1:]      # drop leading ">"
                    else:
                        break                  # stop after first record
                else:
                    seq_len += len(line)       # add length of sequence line
        region = header + ":1-" + str(seq_len)

    # Construct command
    cmd = [
        "breseq", "bam2cov",
        "-t",  # request tab format
        "--region", region,
        "--resolution", "0",  # single-base resolution
        "--output", tab_file,
        "-b", bamfile,
        "-f", fastafile,
    ]
    if extra_args:
        cmd += extra_args

    # Run the command
    try:
        result = subprocess.run(
            cmd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"breseq bam2cov failed: {e.stderr}")
        raise

    # breseq appends ".tab" extension if not present; handle accordingly
    if not tab_file.endswith(".tab"):
        tab_file += ".tab"
    if not os.path.isfile(tab_file):
        raise FileNotFoundError(f"Coverage output file {tab_file} was not created.")

    # Load coverage as DataFrame
    try:
        df = pd.read_csv(
            tab_file,
            sep="\t",
            engine='python',
            header=0,
            index_col=0,
            skipfooter=4,
            comment="#"
        )
    finally:
        # Always remove the temp file, even if there was an error
        if os.path.exists(tab_file):
            os.remove(tab_file)

    return df


# Process the coverage per nucleotide pileup detected across the genome
# into normalized coverage across summary "windows" tiled across the genome
def preprocess(df, win=200, step=100, frag=350):

    if (step > win):
        return print(
            f'window size: {win} is smaller than step size: {step}. '
            f'Excluding segments of the genome for analysis.'
        )

    df_b2c = df.copy()
    df_b2c["unique_cov"] = df_b2c["unique_top_cov"] + df_b2c["unique_bot_cov"]
    df_b2c["redundant"] = df_b2c['redundant_top_cov'] + df_b2c['redundant_bot_cov']

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

    df_b2c["cov_type"] = df_b2c["redundant"].apply(lambda x: 'R' if x > 0 else 'U')
    df_gc = pd.DataFrame(columns=["window_num"])

    i = 0
    lst_win = 0

    # sliding window = win and increment size = step
    # summarizes GC% and median coverage
    while (i <= (genome_len - 1)) and (lst_win < genome_len):

        win_full_cov = df_b2c["unique_cov"].iloc[i:(i + win)].to_numpy()
        cov_type = df_b2c["cov_type"].iloc[i:(i + win)].to_numpy()
        win_cov = []

        # Filter the windows overlapping redundant coverage regions.
        # Ignores any coverage in and adjacent to repetitive/transposable changes
        winu = 0
        for j in range(len(cov_type)):
            if (cov_type[j] == 'U'):
                win_cov.insert(j, float(win_full_cov[j]))
                winu += 1
            else:
                break

        # If stretches of unique window does not exceed set window size,
        # move to the next step
        if (winu < win):
            i = i + step
        else:
            # Summarize the window coverage statistics
            window_med_cov.insert(i, float(np.nanmedian(win_cov)))
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

    gc_fit = np.poly1d(
        np.polyfit(df['gc_percent'].unique(),
                   df['gc_corr_fact'].unique(), 2)
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


# GC-bias correction
def gc_correction(df, zero_frac=0.1):
    # Corrects trends between GC% and coverage in windows
    # using locally weighted regression model

    df = df.copy()

    #Freeze genuinely low-coverage windows at zero
    zero_mask = df["read_count_cov"] <= (df["read_count_cov"].median()*zero_frac)
    df.loc[zero_mask, "norm_raw_cov"] = 0.0

    cov = df["norm_raw_cov"].values
    gc  = df["gc_percent"].values

    loess = sm.nonparametric.lowess
    gc_out = loess(
        cov, gc,
        frac=0.05,
        it=1,
        delta=0.0,
        is_sorted=False,
        missing='none',
        return_sorted=False
    )

    gc_corr = np.zeros_like(cov)
    nonzero = cov > 0
    gc_corr[nonzero] = cov[nonzero] / gc_out[nonzero]

    df["gc_corr_norm_cov"] = gc_corr
    df["gc_corr_fact"]     = gc_out

    return df


def process_multi_genome(
    bamfile,
    fastafile,
    output_prefix,
    win=200,
    step=100,
    frag=350,
    extra_args=None
):
    """
    Run bam2cov + preprocess per genome (FASTA header),
    pool for GC correction, plot pooled bias,
    then return per-genome GC-corrected DataFrames keyed by FASTA header.
    """

    # Discover all genomes in the FASTA [web:7]
    records = parse_fasta_records(fastafile)  # [(header, seq_len), ...]

    preprocessed = {}
    for (header, seq_len) in records:
        region = f"{header}:1-{seq_len}"
        tab_prefix = f"{output_prefix}_{header}"

        # per-genome bam2cov
        df_raw = bam2cov_to_df(
            bamfile,
            fastafile,
            tab_prefix,
            extra_args=extra_args,
            region=region
        )

        # per-genome preprocessing
        df_pre = preprocess(df_raw, win=win, step=step, frag=frag)
        df_pre["genome_id"] = header
        preprocessed[header] = df_pre

    # Pool all genomes for a shared LOWESS GC correction
    df_pooled = pd.concat(preprocessed.values(), ignore_index=True)

    # Global renormalization before LOWESS
    global_median = df_pooled["read_count_cov"].median()
    df_pooled["norm_raw_cov"] = df_pooled["read_count_cov"] / global_median

    # Pooled GC correction
    df_pooled = gc_correction(df_pooled)

    # Pooled GC-bias plot
    gc_cor_plots(df_pooled, output_prefix)

    # Re-separate corrected data by genome_id
    per_genome_corrected = {}
    for header, _ in records:
        df_g = df_pooled[df_pooled["genome_id"] == header].copy()
        df_g.reset_index(drop=True, inplace=True)   # <─ important
        per_genome_corrected[header] = df_g



    return per_genome_corrected

def plot_otr_corr(df, output, ori, ter):

    genome_id = str(df["genome_id"][0])
    samplename = output.strip().split('/')[-1] + genome_id
    saveplt = str(output+"/OTR_corr/")
  
    plt.figure(figsize=(10, 8))
    plt.scatter(df["win_st"],df["norm_raw_cov"], color="gray", label="Raw reads",s=8, alpha = 0.2)
    plt.scatter(df["win_st"],df["gc_corr_norm_cov"], color="gray", label="GC corrected", marker = '*', s=15, alpha = 0.5)
    plt.scatter(df["win_st"],df["otr_gc_corr_norm_cov"], color = 'orange', label="Ori/Ter bias corrected", s = 20, alpha = 0.85, 
                marker = mplt.markers.MarkerStyle(marker = 'o', fillstyle = 'full'))
    plt.plot(df["win_st"], df["otr_gc_corr_fact"], color = "white", label = "OTR-bias-fit-line")
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

#Function to fit lines between ori-ter-coordinates to determine the best fit to actual coverage across genomic windows
def fit_func(params, x, y):
    x1, x2, y1, y2, = params
    
    m = (y1-y2) / (x1-x2)
    c = y1 - m * (x1)
    error = 0
    for i in range(len(x)):
        y_pred = m*x[i] + c
        error += (y[i]-y_pred) ** 2
    return error

#Fit the coverage between the origin and terminus coordinates set by user to detect the presence and the degree of bias observed
# def otr_set(df, ter_idx, ori_idx):
    
#     cyc = False
#     bias = False
#     pt = "trough"
#     x = df.index
#     y = df["gc_corr_norm_cov"]
#     y_med_fil = df["gc_cor_med_fil"]
    
#     len_init = len(x)
    
#     x_cyc = list(islice(cycle(x), 0, len_init*3))
#     y_cyc = list(islice(cycle(y), 0, len_init*3))
    
#     xori_guess = ori_idx
#     xter_guess = ter_idx
#     o_idx = int(y_med_fil.to_numpy().argmax())
#     t_idx = int(y_med_fil.to_numpy().argmin())
#     yori_guess = y.iloc[o_idx]
#     yter_guess = y.iloc[t_idx]
#     # yori_guess = y[y_med_fil.argmax()]
#     # yter_guess = y[y_med_fil.argmin()] 
    
#     if (abs(xori_guess-xter_guess) > len_init * 0.3 ):
#         if (xori_guess < len_init*0.1) or (xori_guess > len_init * 0.9):
#             xori_guess = 0
#             y1 = y_cyc[:xter_guess]
#             y2 = y_cyc[xter_guess:len_init]
#             x1 = x_cyc[:xter_guess]
#             x2 = x_cyc[xter_guess:len_init]
#             initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
#             initial_guess2 = [xter_guess, xori_guess, yter_guess, yori_guess]
#         elif (xter_guess < len_init*0.1) or (xter_guess > len_init * 0.9):
#             xter_guess = 0
#             y1 = y_cyc[:xori_guess]
#             y2 = y_cyc[xori_guess:len_init]
#             x1 = x_cyc[:xori_guess]
#             x2 = x_cyc[xori_guess:len_init]
#             initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
#             initial_guess2 = [len_init, xori_guess, yter_guess, yori_guess]
#             pt = "peak"
#         else:
#             xi = xori_guess if (xori_guess < xter_guess) else xter_guess
#             xj = xori_guess if (xori_guess > xter_guess) else xter_guess
#             y1 = y_cyc[xi:xj]
#             y2 = y_cyc[xj:(xi+len_init)]
#             x1 = x_cyc[xi:xj]
#             x2 = x_cyc[xj:(xi+len_init)]
#             initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
#             initial_guess2 = [xter_guess, len_init, yter_guess, yori_guess]
#             cyc = True
#         bias = True
#     else:
#         y_corr = y
#         y_fit = np.repeat(np.mean(y), len_init)
#         print("Specified ori and ter too close to each other. No correction applied.")
#         return y_corr, y_fit, bias
    
#     result1 = minimize(fit_func, initial_guess1, args = (x1, y1))
#     result2 = minimize(fit_func, initial_guess2, args = (x2, y2))
    
#     xori_opt1, xter_opt1, yori_opt1, yter_opt1 = result1.x
#     xter_opt2, xori_opt2, yter_opt2, yori_opt2 = result2.x
    
#     xori_opt = np.mean([xori_opt1, xori_opt2])
#     xter_opt = np.mean([xter_opt1, xter_opt2])
#     yori_opt = np.mean([yori_opt1, yori_opt2])
#     yter_opt = np.mean([yter_opt1, yter_opt2])
    
#     y1_fit=[]
#     y2_fit=[]
    
#     if (yori_opt / yter_opt) > 1:
#         bias = True
#     else:
#         bias = False

#     if bias and cyc:
        
#         if (xori_opt > xter_opt):
#             m_opt = (yori_opt - yter_opt) / (xori_opt - xter_opt)
#             c_opt = yori_opt - m_opt * (xori_opt - xter_opt)
#             m_opt1 = (yori_opt-yter_opt) / (xori_guess-xter_guess)
#             m_opt2 = (yter_opt-yori_opt) / (len(x)-(xori_guess-xter_guess))
            
#             c_opt1 = yori_opt - m_opt * (xori_opt-xter_opt)
#             c_opt2 = yter_opt - m_opt * (xter_opt-xori_opt)
            
#             y1_fit = [m_opt1 * x + c_opt1 for x in range(len(x1))]
#             y2_fit = [m_opt2 * x + c_opt2 for x in range(len(x2))]
#             y_fit = y1_fit + y2_fit
#             y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xter_guess), (2*len(y_fit) - int(xter_guess)))))
#         else:
#             m_opt = (yori_opt - yter_opt) / (xori_opt - xter_opt)
#             c_opt = yori_opt - m_opt * (xori_opt - xter_opt)
#             m_opt1 = (yori_opt-yter_opt) / (xori_guess-xter_guess)
#             m_opt2 = (yter_opt-yori_opt) / (len(x)-(xori_guess-xter_guess))
            
#             c_opt1 = yori_opt - m_opt * (xori_opt-xter_opt)
#             c_opt2 = yter_opt - m_opt * (xter_opt-xori_opt)
            
#             y1_fit = [m_opt1 * x + c_opt1 for x in range(len(x1))]
#             y2_fit = [m_opt2 * x + c_opt2 for x in range(len(x2))]
#             y_fit = y1_fit + y2_fit
#             y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xori_guess), (2*len(y_fit) - int(xori_guess)))))
#         y_corr = y / y_fit
        
#     elif bias and not cyc:
#         if pt == "peak":
#             xter_opt = 0
            
#             m_opt1 = (yori_opt1 - yter_opt1) / (xori_opt1 - xter_opt1)
#             m_opt2 = (yter_opt2 - yori_opt2) / (xter_opt2 - xori_opt2)

#             c_opt1 = yori_opt2 - m_opt1 * (int(xori_opt1))
#             c_opt2 = yori_opt2 - m_opt2 * (int(xori_opt2))
            
#             y1_fit = [m_opt1 * x + c_opt1 for x in x1]
#             y2_fit = [m_opt2 * x + c_opt2 for x in x2]
#             y_fit = y1_fit + y2_fit
            
#         else:
#             xori_opt = 0
            
#             m_opt1 = (yori_opt1 - yter_opt1) / (xori_opt1 - xter_opt1)
#             m_opt2 = (yter_opt2 - yori_opt2) / (xter_opt2 - xori_opt2)
            
#             c_opt1 = yter_opt1 - m_opt1 * (int(xter_opt1))
#             c_opt2 = yter_opt1 - m_opt2 * (int(xori_opt2))
            
#             y1_fit = [m_opt1 * x + c_opt1 for x in x1]
#             y2_fit = [m_opt2 * x + c_opt2 for x in x2]
            
#             y_fit = y1_fit + y2_fit
            
#         y_corr = y / y_fit
#     else:
#         # y_corr = np.repeat(np.mean(y), len_init)
#         y_corr = y
#         y_fit = np.repeat(np.mean(y), len_init)
#         print("OTR bias not detected")
#         return y_corr, y_fit, xori_guess, xter_guess, bias

#     return y_corr, y_fit, bias

#Fit the coverage based on the presence and the degree of origin and terminus biased read counts observed
def otr_fit(df):
    
    cyc = False
    bias = False
    pt = "trough"
    x = df.index
    y = df["gc_corr_norm_cov"]
    y_med_fil = df["gc_cor_med_fil"]
    
    len_init = len(x)
    
    x_cyc = list(islice(cycle(x), 0, len_init*3))
    y_cyc = list(islice(cycle(y), 0, len_init*3))
    
    xori_guess = y_med_fil.argmax()
    xter_guess = y_med_fil.argmin()
    o_idx = int(y_med_fil.to_numpy().argmax())
    t_idx = int(y_med_fil.to_numpy().argmin())
    yori_guess = y.iloc[o_idx]
    yter_guess = y.iloc[t_idx]

    print(f'xori_guess:{xori_guess} and xter_guess: {xter_guess}')


    if (abs((xori_guess - xter_guess)) >= (len_init * 0.3)):

        bias = True
        
        if (xori_guess < len_init * 0.1) or (xori_guess > len_init * 0.9):
            xori_guess = 0
            y1 = y_cyc[:xter_guess]
            y2 = y_cyc[xter_guess:len_init]
            x1 = x_cyc[:xter_guess]
            x2 = x_cyc[xter_guess:len_init]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [xter_guess, xori_guess, yter_guess, yori_guess]

        elif (xter_guess < len_init*0.1) or (xter_guess > len_init * 0.9):
            xter_guess = 0
            y1 = y_cyc[:xori_guess]
            y2 = y_cyc[xori_guess:len_init]
            x1 = x_cyc[:xori_guess]
            x2 = x_cyc[xori_guess:len_init]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [len_init, xori_guess, yter_guess, yori_guess]
            pt = "peak"
        
        else:
            xi = xori_guess if (xori_guess < xter_guess) else xter_guess
            xj = xori_guess if (xori_guess > xter_guess) else xter_guess
            y1 = y_cyc[xi:xj]
            y2 = y_cyc[xj:(xi+len_init)]
            x1 = x_cyc[xi:xj]
            x2 = x_cyc[xj:(xi+len_init)]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [xter_guess, len_init, yter_guess, yori_guess]
            cyc = True

    else:
        y_corr = y
        y_fit = np.repeat(np.mean(y), len_init)
        print("OTR bias not detected")

        return y_corr, y_fit, xori_guess, xter_guess, bias
    
    result1 = minimize(fit_func, initial_guess1, args = (x1, y1))
    result2 = minimize(fit_func, initial_guess2, args = (x2, y2))
    
    xori_opt1, xter_opt1, yori_opt1, yter_opt1 = result1.x
    xter_opt2, xori_opt2, yter_opt2, yori_opt2 = result2.x
    
    xori_opt = np.mean([xori_opt1, xori_opt2])
    xter_opt = np.mean([xter_opt1, xter_opt2])
    yori_opt = np.mean([yori_opt1, yori_opt2])
    yter_opt = np.mean([yter_opt1, yter_opt2])
    
    print(f'guess1:{initial_guess1}, guess2:{initial_guess2}')
    # print(f'xori_opt1:{xori_opt1} and xter_opt1: {xter_opt1}')
    print(f'yori_opt1:{yori_opt1} and yter_opt1: {yter_opt1}')
    # print(f'xori_opt2:{xori_opt2} and xter_opt2: {xter_opt2}')
    print(f'yori_opt2:{yori_opt2} and yter_opt2: {yter_opt2}')


    y1_fit=[]
    y2_fit=[]

    if (yori_opt / yter_opt) > 1:
        bias = True
    else:
        bias = False
    
    if bias and cyc:
        
        if (xori_opt > xter_opt):
            m_opt = (yori_opt - yter_opt) / (xori_opt - xter_opt)
            c_opt = yori_opt - m_opt * (xori_opt - xter_opt)
            m_opt1 = (yori_opt-yter_opt) / (xori_guess-xter_guess)
            m_opt2 = (yter_opt-yori_opt) / (len(x)-(xori_guess-xter_guess))
            
            c_opt1 = yori_opt - m_opt * (xori_opt-xter_opt)
            c_opt2 = yter_opt - m_opt * (xter_opt-xori_opt)
            
            y1_fit = [m_opt1 * x + c_opt1 for x in range(len(x1))]
            y2_fit = [m_opt2 * x + c_opt2 for x in range(len(x2))]
            y_fit = y1_fit + y2_fit
            y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xter_guess), (2*len(y_fit) - int(xter_guess)))))
        else:
            m_opt = (yori_opt - yter_opt) / (xori_opt - xter_opt)
            c_opt = yori_opt - m_opt * (xori_opt - xter_opt)
            m_opt1 = (yori_opt-yter_opt) / (xori_guess-xter_guess)
            m_opt2 = (yter_opt-yori_opt) / (len(x)-(xori_guess-xter_guess))
            
            c_opt1 = yori_opt - m_opt * (xori_opt-xter_opt)
            c_opt2 = yter_opt - m_opt * (xter_opt-xori_opt)
            
            y1_fit = [m_opt1 * x + c_opt1 for x in range(len(x1))]
            y2_fit = [m_opt2 * x + c_opt2 for x in range(len(x2))]
            y_fit = y1_fit + y2_fit
            y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xori_guess), (2*len(y_fit) - int(xori_guess)))))
        y_corr = y / y_fit
        
    elif bias and not cyc:
        if pt == "peak":
            xter_opt = 0
            
            m_opt1 = (yori_opt1 - yter_opt1) / (xori_opt1 - xter_opt1)
            m_opt2 = (yter_opt2 - yori_opt2) / (xter_opt2 - xori_opt2)

            c_opt1 = yori_opt2 - m_opt1 * (int(xori_opt1))
            c_opt2 = yori_opt2 - m_opt2 * (int(xori_opt2))
            
            y1_fit = [m_opt1 * x + c_opt1 for x in x1]
            y2_fit = [m_opt2 * x + c_opt2 for x in x2]
            y_fit = y1_fit + y2_fit
            
        else:
            xori_opt = 0
            
            m_opt1 = (yori_opt1 - yter_opt1) / (xori_opt1 - xter_opt1)
            m_opt2 = (yter_opt2 - yori_opt2) / (xter_opt2 - xori_opt2)
            
            c_opt1 = yter_opt1 - m_opt1 * (int(xter_opt1))
            c_opt2 = yter_opt1 - m_opt2 * (int(xter_opt2))
            
            y1_fit = [m_opt1 * x + c_opt1 for x in x1]
            y2_fit = [m_opt2 * x + c_opt2 for x in x2]
            
            y_fit = y1_fit + y2_fit
            print(f'm_opt1:{m_opt1} and m_opt2:{m_opt2}')
            print(f'c_opt1:{c_opt1} and c_opt2:{c_opt2}')


        y_corr = y / y_fit

    else:
        # y_corr = np.repeat(np.mean(y), len_init)
        y_corr = y
        y_fit = np.repeat(np.mean(y), len_init)
        print("OTR bias not detected")
        return y_corr, y_fit, xori_guess, xter_guess, bias

    
    return y_corr, y_fit, int(xori_opt), int(xter_opt), bias


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#Correction of the normalized coverage based on the bias detected.
#Origin and terminus are always inferred from the coverage profile.
def otr_correction(df, output):

    if "gc_cor_med_fil" not in df.columns:
        n = len(df)
        # choose a safe window: at least 3, at most n, and odd
        win = max(3, int(n / 10))
        win = min(win, n)
        if win % 2 == 0:
            win -= 1
        if win < 1:
            # degenerate case: just copy the original series
            df["gc_cor_med_fil"] = df["gc_corr_norm_cov"].copy()
        else:
            df["gc_cor_med_fil"] = ndimage.median_filter(
                df["gc_corr_norm_cov"],
                size=win,
                mode="reflect",
            )

    genome_id = str(df["genome_id"][0])
    samplename = output.strip().split('/')[-1] + genome_id
    saveplt = str(output+"/OTR_corr/")

    # Fit the bias curve to the most probable location of ori/ter based on
    # coverage peaks and troughs respectively.
    h1, f1, ori_idx, ter_idx, bias = otr_fit(df)
    if bias:
        xori = df["win_st"].iloc[ori_idx]
        xter = df["win_st"].iloc[ter_idx]
        yori = f1[ori_idx]
        yter = f1[ter_idx]
        OTR = yori / (yter + 0.001)
    else:
        xori = df["win_st"].iloc[0]
        xter = df["win_end"].iloc[len(df)-1]
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
    # df["otr_gc_corr_norm_cov"] = h1
    # df["otr_gc_corr_fact"] = f1
    df["otr_gc_corr_norm_cov"] = df["gc_corr_norm_cov"].copy()
    low = df["read_count_cov"] <= df["read_count_cov"].median() * 0.1
    df.loc[~low, "otr_gc_corr_norm_cov"] = h1[~low]  # only apply scaling where coverage is non‑trivially positive
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

def calculate_prob(p, r, obs):
    # probabilities calculated by assuming negative binomial distribution (Poisson family), to account for 
    # a wide dispersion of coverage data points given noisy data and high copy number possibilities (amplifications)
    # gammaln function allows for calculation of log probabilities without computational over-flow. 
    
    probs = np.exp(gammaln(r + obs) - gammaln(obs + 1) - gammaln(r) + obs * np.log(p) + r * np.log(1 - p))
    return probs

#Emission Matrix
def setup_emission_matrix(n_states, mean, variance, absmax, error_rate):
    emission = np.full((n_states, absmax + 1), np.nan)
    
    for state in range(n_states):
        pr = solve_pr(mean * (state + 1), variance * (state + 1))
        p, r = pr[0], pr[1]
        
        for obs in range(absmax + 1):
            emission[state, obs] = calculate_prob(p, r, obs)
    
    # probability mass function determines the lower prbability of predicting zero as the readcounts approach absmax.
    # error rate offsets the probability threshold of predicting zero at higher readcounts accounting for the erronous read alignments
    zero_row = np.array([geom.pmf(i - 1, 1 - error_rate) for i in range(absmax + 1)])
    emission = np.vstack((zero_row, emission))
    # np.savetxt("emission.csv", emission, delimiter=",")  
    return emission

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

#Make Viterbi Matrtix
def make_viterbi_mat(obs, transition_matrix, emission_matrix):
    num_states = transition_matrix.shape[0]
    
    # Create a mask for the zero values
    mask = (emission_matrix == 0)
    # Take the logarithm of the non-zero values
    logemi = np.zeros_like(emission_matrix, dtype=float)
    logemi[~mask] = np.log(emission_matrix[~mask])

    # Handle the zero values separately, set to -inf
    logemi[mask] = -np.inf 

    logv = np.full((len(obs), num_states), np.nan)
    logtrans = np.log(transition_matrix)
    
    logv[0,:] = -np.inf
    
    #start prob of state = 1 when including zero state

    logv[0, 1] = np.log(1e-100)
    
    for i in range(1, len(obs)):
        for l in range(num_states):
            statelprobcounti = logemi[l, obs[i]]
            maxstate = max(logv[i - 1, :] + logtrans[l, :])
            logv[i, l] = statelprobcounti + maxstate
    # np.savetxt("viterbi.csv", logv, delimiter = ',')
    return logv


def HMM_copy_number(obs, transition_matrix, emission_matrix, win_st, win_end, chr_length):
    states = np.arange(0, emission_matrix.shape[0] + 1)  # Assuming state indices start from 1
    
    v = make_viterbi_mat(obs, transition_matrix, emission_matrix)
            
    # Go through each of the rows of the matrix v and find out which column has the maximum value for that row
    most_probable_state_path = np.argmax(v, axis=1)
    results = pd.DataFrame(columns=['Startpos', 'Endpos', 'State'])
    
    prev_obs = obs[0]
    prev_most_probable_state = most_probable_state_path[0]
    prev_most_probable_state_name = states[prev_most_probable_state]  # Adjust for 0-based indexing
    start_pos = 0

    
    for i in range(0, len(obs)-1):

        observation = obs[i]
        most_probable_state = most_probable_state_path[i]
        most_probable_state_name = states[most_probable_state]  # Adjust for 0-based indexing
        
        if most_probable_state_name != prev_most_probable_state_name:
            results = results._append({'Startpos': start_pos,'Endpos': win_end[i-1], 
                                      'State': prev_most_probable_state_name}, ignore_index=True)
            start_pos = win_st[i]
        
        prev_obs = observation
        prev_most_probable_state_name = most_probable_state_name

    results = results._append({'Startpos': start_pos, 'Endpos': chr_length, 
                              'State': prev_most_probable_state_name}, ignore_index=True)
    
    return results

def run_HMM(df, output, error_rate=0.15, n_states=5, changeprob=(1e-10)):
    
    saveloc = str(output+"/CNV_csv/")
    genome_id = str(df["genome_id"][0])
    samplename = output.strip().split('/')[-1] + genome_id
    # samplename = sample.strip().split('.')[0]
    
    df["otr_gc_corr_norm_cov"] = np.nan_to_num(df["otr_gc_corr_norm_cov"])
    # cor_cov = df["otr_gc_corr_norm_cov"]

    med = df["read_count_cov"].median()
    cor_rc = []
    
    for i in range(len(df)):
        # cor_rc.insert(i, int(np.nan_to_num(df["otr_gc_corr_norm_cov"].iloc[i]) * med))
        cor_rc.insert(i, int(df["otr_gc_corr_norm_cov"].iloc[i] * med))
    
    mean = np.mean(cor_rc)
    var = np.var(cor_rc)
    
    df["otr_gc_corr_rdcnt_cov"] = cor_rc
    
    # determines the number of expected states based on the normalized coverage range. Minimum number of expected states = 5
    n_states = int(df["otr_gc_corr_norm_cov"].max()) if int(df["otr_gc_corr_norm_cov"].max()) > 5 else 5

    new_exp = df.copy()
    
    rc_max = np.max(cor_rc)
    
    this_emission = setup_emission_matrix(n_states=n_states, mean=mean, variance=var, absmax=rc_max, error_rate=error_rate)
    this_transition = setup_transition_matrix(n_states, remain_prob=(1 - changeprob))
    
    # outputs consequtive segments of the genome where copy number changes based on HMM : Break points
    copy_numbers = HMM_copy_number(cor_rc, this_transition, this_emission, df["win_st"], df["win_end"], df['win_end'].max())
    
    brk_full_path = os.path.join(saveloc,'%s_break_pts.csv' % samplename)
    cn_brk = pd.DataFrame(copy_numbers,columns=['Startpos','Endpos', 'State'])
    cn_brk['Segment_Size'] = cn_brk['Endpos'] - cn_brk['Startpos']
    cn_brk.drop(columns='Endpos', inplace = True)
    cn_brk.to_csv(brk_full_path, index = False)
    
    CN_HMM = []
    # output copy number prediction for each instance of genomic sliding window
    for cnrow in range(len(copy_numbers)):

        state = int(copy_numbers['State'][cnrow])

        hmmstart = int(copy_numbers['Startpos'][cnrow])
        hmmend = int(copy_numbers['Endpos'][cnrow])
        
        CN_HMM_row = []
        # Index lists of windows between the breakpoints
        idx_list = df[(hmmstart <= df["win_st"]) & (df["win_st"] <= hmmend)].index


        if len(idx_list) == 0:
            continue  # skip if no matching start position found
    
        #Iterate over the select windows and append the copy number (state) to the window
        for idx in (idx_list):

            if ((df["win_st"].iloc[idx] >= hmmstart) and (df["win_end"].iloc[idx] <= hmmend)):
                CN_HMM_row.append(state)
            else:
                continue
        
        CN_HMM.extend(CN_HMM_row)
    
    new_exp['prob_copy_number'] = CN_HMM

    csv_full_path = os.path.join(saveloc,'%s_CNV.csv' % samplename)

    new_exp.reset_index(drop = True)
    new_exp.to_csv(csv_full_path, index = False)

    print(f"{samplename}: Copy number prediction complete. .csv files saved.")
    
    return new_exp