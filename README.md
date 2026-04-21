# CNery

*breseq* copy-number-variation extension. `CNery` reads the sequencing coverage output from [*breseq*](https://github.com/barricklab/breseq) and predicts copy-number variation (CNV) across the genome. Predictions are corrected for coverage biases introduced by sequencing chemistry (GC-content bias) and prokaryotic replication state during DNA isolation (origin-to-terminus / OTR bias).

Recent updates (latest commits):

- **Multi-genome CNV analysis** — `CNery` now processes *all* reference sequences found in the breseq BAM/FASTA in one pass. Each reference (chromosome, plasmid, contig, etc.) is preprocessed per genome, pooled for a shared LOWESS GC-bias fit, and then bias-corrected and CN-called independently.
- **Input/output flexibility** — inputs default to `<input>/data/reference.bam` and `<input>/data/reference.fasta`; output prefix defaults to `<input>/CNV_out/`. Output subfolders (`CNV_plt/`, `CNV_csv/`, `GC_bias/`, `OTR_corr/`) are created automatically.
- **Modular bias correction** — the `--bias` flag lets you choose `all` (GC + OTR), `gc`, `otr`, or `none`.
- **Pip-installable package** — `requirements.txt` and a fixed `pyproject.toml` allow install directly from GitHub via `pip install git+...`.
- **Origin/terminus coordinates are always inferred** — the previous `--ori` / `--ter` options have been removed. `CNery` now fits the OTR bias curve to the coverage profile automatically in every run.

---

## Installation

Recommended: create a conda/mamba environment from the provided spec.

```bash
mamba env create -f environment.yml
mamba activate CNery
```

Install `CNery` (a.k.a. `breseq-ext-cnv`) from GitHub:

```bash
pip install git+https://github.com/barricklab/breseq-ext-cnv.git
```

---

## Quick start

Run `CNery` inside a *breseq* output folder that contains the `data/` and `output/` subfolders:

```bash
CNery [-o <output folder>] [-w <window>] [-s <step size>] [-f <fragment length>]
```

To run from a different working directory, point `-i` at the breseq output folder (or supply `-ref` and the BAM path manually):

```bash
CNery -i <breseq output folder> \
      -ref <reference.fasta> \
      -o  <output folder> \
      -w  <window> \
      -s  <step size> \
      -f  <fragment length>
```

---

## Usage examples

Calculate coverage with a 500 bp window sliding in 250 bp steps; sequencing fragment length is 300 bp:

```bash
CNery -o <output folder> -w 500 -s 250 -f 300
```

Analyze coverage across the whole genome, but restrict CNV/coverage plots to a specific genomic segment:

```bash
CNery -o <output folder> --region 3497890-3955678 -w 1000 -s 500
```

The `--region` argument accepts open intervals too (`-reg 3497890-` from a start to end of genome, `-reg -3955678` from start of genome to an end position).

Control which bias correction is applied before CN prediction:

```bash
# Both GC + OTR corrections (default)
CNery -o <output folder> -w 500 -s 250 --bias all

# Only correct OTR (replication) bias
CNery -o <output folder> -w 500 -s 250 --bias otr

# Only correct GC-content bias
CNery -o <output folder> -w 500 -s 250 --bias gc

# No bias correction before CN prediction
CNery -o <output folder> -w 500 -s 250 --bias none
```

When OTR correction is applied, the origin and terminus of replication are automatically inferred from the coverage profile — no manual coordinates are required.

---

## Outputs

Given an output folder `CNV_out/`, `CNery` writes:

- `CNV_out/CNV_plt/` — per-reference CNV prediction plots.
- `CNV_out/CNV_csv/` — per-window coverage + CN calls as CSV.
- `CNV_out/GC_bias/` — pooled LOWESS GC-bias diagnostic plot.
- `CNV_out/OTR_corr/` — per-reference OTR bias plots and a JSON summary (`*_otr_results.json`) containing the inferred origin window, terminus window, normalized coverage at each, and the origin-to-terminus ratio.

Each reference sequence in the BAM/FASTA produces its own set of outputs, named with the reference / genome identifier.

---

## All command-line options

```
$ CNery -h

usage: CNery [-h] [-i I] [-ref REF] [-reg REG] [-o O] [-w W] [-s S] [-f F] [-e E]
             [--bias {all,none,gc,otr}]

CNery is a Python package extension to breseq that analyzes the sequencing
coverage across the genome to predict copy number variation (CNV).

options:
  -h, --help            show this help message and exit
  -i, --input I         input folder path (the breseq output folder with
                        'data' and 'output' folders). Defaults to the current
                        folder.
  -ref REF              select the reference file used for breseq. Defaults
                        to data/reference.fasta.
  -reg REG              select the region of the genome to evaluate
                        (format: START-END, e.g. 1000-50000).
  -o, --output O        output file prefix / storage location. Defaults to
                        the 'CNV_out' folder in the current dir.
  -w, --window W        Window length used to parse the genome and compute
                        coverage and GC statistics. Default: 200.
  -s, --step-size S     Step size (<= window size) for each progression of
                        the window across the genome. Set step-size = window
                        size for non-overlapping windows. Default: 100.
  -f, --frag_size F     Average fragment size of the sequencing reads.
                        Default: 500.
  -e, --error-rate E    Approximate error rate in sequencing read coverage /
                        reference alignment. Default: 0.05.
  --bias {all,none,gc,otr}
                        Select which bias correction to apply before CN
                        prediction. 'all' applies GC + OTR, 'gc' or 'otr'
                        applies only that one, 'none' skips bias correction.
                        Default: all.

Run this script in the breseq output folder that contains 'data' and 'output'
folders.
```

---

## Notes

- The origin/terminus of replication are inferred from the coverage profile using a two-slope fit around the genome-wide coverage peak (origin) and trough (terminus). Inferred coordinates and the fitted origin-to-terminus coverage ratio are written to `OTR_corr/*_otr_results.json` for every reference sequence.
- GC-bias correction uses a LOWESS fit pooled across *all* reference sequences in the BAM, so smaller replicons (e.g. plasmids) borrow strength from the main chromosome.
- CN calls are made by a hidden Markov model on the bias-corrected, normalized coverage.
