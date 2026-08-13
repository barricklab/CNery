# CNery

*breseq* copy-number-variation extension. `CNery` reads the sequencing coverage output from [*breseq*](https://github.com/barricklab/breseq) and predicts copy-number variation (CNV) across the genome. Predictions are corrected for coverage biases introduced by sequencing chemistry (GC-content bias) and prokaryotic replication state during DNA isolation (origin-to-terminus / OTR bias).

Recent updates (latest commits):

- **Multi-genome CNV analysis** — `CNery` now processes *all* reference sequences found in the breseq BAM/FASTA in one pass. Each reference (chromosome, plasmid, contig, etc.) is preprocessed per genome, pooled for a shared LOWESS GC-bias fit, and then bias-corrected and CN-called independently.
- **Input/output flexibility** — inputs default to `<input>/data/reference.bam` and `<input>/data/reference.fasta`; output prefix defaults to `<input>/CNV_out/`. Output subfolders (`CNV_plt/`, `CNV_csv/`, `GC_bias/`, `OTR_corr/`) are created automatically.
- **Modular bias correction** — the `--bias` flag lets you choose `all` (GC + OTR), `gc`, `otr`, or `none`.
- **Pip-installable package** — `requirements.txt` and a fixed `pyproject.toml` allow install directly from GitHub via `pip install git+...`.
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

## Generating a coverage table

`CNery` normally calls `breseq bam2cov` itself. You can also generate the per-reference coverage
tables ahead of time and point `CNery` at them with `--coverage-dir`, which skips `breseq` entirely
and needs **no BAM** — only the reference FASTA. Useful for archiving an analysis input, sharing a
reproducible test case, or running `CNery` where `breseq` is not installed.

```bash
CNery --coverage-dir coverage -ref reference.fasta -o CNV_out
```

`CNery` looks for one table per reference sequence, named `<seq_id>.coverage.tsv`, where `<seq_id>`
is the FASTA header up to its first space. A table must exist for every sequence in the FASTA; a
missing one is reported by name before any work starts.

```
coverage/
├── REL606.coverage.tsv
└── pPlasmid.coverage.tsv
```

**Requires [pre-release breseq](https://github.com/barricklab/conda)**, the Barrick lab channel of
development builds auto-built from `barricklab/breseq` master. Released versions on bioconda do not
include the current fixes to how coverage tables are written.

```bash
conda install -c https://barricklab.github.io/conda/ -c conda-forge -c bioconda breseq-prerelease
```

Omit `--region` and `breseq` writes one table per reference sequence automatically — no need to look
up sequence IDs or lengths. `--output` is then a directory:

```bash
mkdir -p coverage
breseq bam2cov \
  --format TSV \
  --resolution 0 \
  --output coverage \
  -b data/reference.bam \
  -f data/reference.fasta
```

To generate a single table instead, name the region using the sequence ID exactly as it appears in
the FASTA, and give `--output` the name `CNery` expects:

```bash
breseq bam2cov \
  --format TSV \
  --region REL606:1-4629812 \
  --resolution 0 \
  --output coverage/REL606.coverage.tsv \
  -b data/reference.bam \
  -f data/reference.fasta
```

Three options matter:

- **`--resolution 0`** outputs every position. The default is 600, which samples only 600 points
  across the whole region — far too sparse for windowed coverage, and it fails silently by producing
  a well-formed but nearly empty table.
- **`--format TSV`** produces a tab-separated table instead of a plot. (`-t` is accepted as a
  shorthand for the same thing.)
- **`--per-read-group`** *(optional)* repeats every coverage column once per read group (`@RG`) in
  the BAM, prefixed `RG-<n>_` where `<n>` is the read group's index in the BAM header. Requires a
  table format; it is rejected with `--format PNG`. A BAM with no read groups yields a single `RG-0`
  set.

```bash
breseq bam2cov \
  --format TSV \
  --per-read-group \
  --resolution 0 \
  --output coverage \
  -b data/reference.bam \
  -f data/reference.fasta
```

`CNery` reads such a table without any change. The aggregate columns keep their names and positions
and the per-read-group repeats are **appended**, so the columns `CNery` needs are still where it
expects them; it selects by name and ignores the rest. The per-read-group summary lines added to the
footer are stripped along with the others by their `#` prefix. Nothing in `CNery` consumes the
per-read-group columns today — they pass through — so the option is safe to enable now if you want
per-library coverage available in the same file for other tools.

The output carries a header row, one row per reference position, and a trailing `#`-commented summary
block:

```
position	ref_base	unique_top_cov	unique_bot_cov	redundant_top_cov	redundant_bot_cov	...
1	G	14	18	0	0	...
2	G	14	18	0	0	...
...
#	region_unique_average_cov	56.7267
#	region_repeat_average_cov	0
#	region_average_cov	56.7267
#	number_of_positions	4629812
```

That summary block is variable in length — `--show-average` adds a line, and `--per-read-group` adds
three per group (`#	RG-0_region_unique_average_cov	…`) — so anything parsing these tables should
skip lines by their `#` prefix rather than dropping a fixed number from the end.

For the same reason, do not assume a fixed column count: `position` must be the first column and the
named coverage columns must be present, but extra columns to the right are expected and should be
ignored rather than treated as an error.

If you do need the sequence IDs and lengths for a `--region` argument, read them from the FASTA
headers or the `.fai` index:

```bash
grep '^>' data/reference.fasta
cut -f1,2 data/reference.fasta.fai
```

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
