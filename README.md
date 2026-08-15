# CNery

*breseq* copy-number-variation extension. `CNery` reads per-reference coverage tables produced by [*breseq*](https://github.com/barricklab/breseq) `bam2cov` and predicts copy-number variation (CNV) across the genome. Predictions are corrected for coverage biases introduced by sequencing chemistry (GC-content bias) and prokaryotic replication state during DNA isolation (origin-to-terminus / OTR bias).

Recent updates (latest commits):

- **Coverage tables are the only input** — `CNery` reads *breseq* `bam2cov --format TSV` coverage tables and nothing else. It no longer needs a BAM or a reference FASTA, and never runs `breseq` itself. The reference sequence it needs for GC content is already in the table's own `ref_base` column.
- **Files and folders on the command line** — name coverage tables directly, or name folders and let `CNery` find them by file ending (`--file-ending`, default `coverage.tsv`). Files and folders can be mixed in one command.
- **Multi-genome CNV analysis** — `CNery` processes *all* the coverage tables given in one pass. Each reference (chromosome, plasmid, contig, etc.) is preprocessed separately, pooled for a shared LOWESS GC-bias fit, and then bias-corrected and CN-called independently.
- **Output flexibility** — output prefix defaults to `CNV_out/` in the current folder. Output subfolders (`CNV_plt/`, `CNV_csv/`, `GC_bias/`, `OTR_corr/`) are created automatically.
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

`CNery` takes coverage tables — see [Generating a coverage table](#generating-a-coverage-table) if
you do not have them yet. Point it at the folder holding them:

```bash
CNery <folder> [-o <output folder>] [-w <window>] [-s <step size>] [-f <fragment length>]
```

Run with no arguments at all and it reads the current folder. You can also name tables directly, or
mix files and folders in one command:

```bash
CNery REL606.coverage.tsv pPlasmid.coverage.tsv -o CNV_out
CNery coverage/ extra/pContig.coverage.tsv -o CNV_out
```

Folders are searched, top level only, for files ending in `coverage.tsv`. Use `--file-ending` if your
tables are named otherwise; repeat the flag to accept several. Note that any `--file-ending`
**replaces** the default rather than adding to it:

```bash
CNery coverage/ --file-ending cov.txt
CNery coverage/ --file-ending cov.txt --file-ending coverage.tsv
```

A table's sequence ID comes from its file name, with the matched ending and the `.` in front of it
removed — `REL606.coverage.tsv` becomes `REL606`, and `NC_012967.1.coverage.tsv` becomes
`NC_012967.1`. That ID names every output file, and no two inputs may share one.

**Everything passed in one command is analyzed together**, sharing a single GC-bias fit and one
global coverage median. That is what you want for the references of one sample — chromosome,
plasmids and contigs. Analyze separate samples with separate commands.

---

## Usage examples

Calculate coverage with a 500 bp window sliding in 250 bp steps; sequencing fragment length is 300 bp:

```bash
CNery <inputs> -o <output folder> -w 500 -s 250 -f 300
```

Analyze coverage across the whole genome, but restrict CNV/coverage plots to a specific genomic segment:

```bash
CNery <inputs> -o <output folder> --region 3497890-3955678 -w 1000 -s 500
```

The `--region` argument accepts open intervals too (`-reg 3497890-` from a start to end of genome, `-reg -3955678` from start of genome to an end position).

Control which bias correction is applied before CN prediction:

```bash
# Both GC + OTR corrections (default)
CNery <inputs> -o <output folder> -w 500 -s 250 --bias all

# Only correct OTR (replication) bias
CNery <inputs> -o <output folder> -w 500 -s 250 --bias otr

# Only correct GC-content bias
CNery <inputs> -o <output folder> -w 500 -s 250 --bias gc

# No bias correction before CN prediction
CNery <inputs> -o <output folder> -w 500 -s 250 --bias none
```

When OTR correction is applied, the origin and terminus of replication are automatically inferred from the coverage profile — no manual coordinates are required.

---

## Generating a coverage table

Coverage tables are `CNery`'s only input. Generate them once with `breseq bam2cov`, then run `CNery`
against them as often as you like — no BAM, no reference FASTA, and `breseq` need not be installed
on the machine that runs `CNery`. The reference sequence `CNery` needs for GC content is already in
each table's `ref_base` column.

The conventional layout is one table per reference sequence, named `<seq_id>.coverage.tsv`:

```
coverage/
├── REL606.coverage.tsv
└── pPlasmid.coverage.tsv
```

```bash
CNery coverage/ -o CNV_out
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
the FASTA, and give `--output` a name ending in `.coverage.tsv`:

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

`CNery` reads six of these columns — `position`, `ref_base`, and the four `unique_*`/`redundant_*`
coverage columns — and checks for them as soon as a table is opened, so a file with the wrong schema
is rejected by name instead of failing later. Note that *breseq*'s own
`08_mutation_identification/*.coverage.tab` files use a **different** schema (position last, no
`ref_base`) and are not usable as `CNery` input.

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

Each coverage table produces its own set of outputs, named with the sequence ID derived from its file name. The GC-bias plot is the exception: one pooled fit covers every table in the run.

---

## All command-line options

```
$ CNery -h

usage: CNery [-h] [--file-ending ENDING] [-reg REG] [-o O] [-w W] [-s S]
             [-f F] [-e E] [--bias {all,none,gc,otr}]
             [INPUT ...]

CNery is a Python package extension to breseq that analyzes the sequencing
coverage across the genome to predict copy number variation (CNV).

positional arguments:
  INPUT                 Coverage table files, and/or folders containing them.
                        Folders are searched (top level only) for files
                        ending in --file-ending. Every table given is
                        analyzed together, sharing one GC-bias fit, so these
                        should be the reference sequences of a single sample.
                        Defaults to the current folder.

options:
  -h, --help            show this help message and exit
  --file-ending ENDING  File ending that identifies a coverage table inside
                        an input folder. Repeat the flag to accept more than
                        one. Any --file-ending REPLACES the default
                        ('coverage.tsv') rather than adding to it. A file
                        named directly on the command line is always used,
                        whatever it is called.
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

Inputs are breseq 'bam2cov --format TSV' coverage tables. Run with no
arguments in a folder that holds them, or name files and/or folders directly.
```

---

## Testing

The test suite has two tiers, and **a bare `pytest` runs both**. Set up the development environment
first:

```bash
conda env create -f dev-environment.yml --prefix=$PWD/env
conda run -p $PWD/env pytest                 # everything
conda run -p $PWD/env pytest -m synthetic    # fast, offline
conda run -p $PWD/env pytest -m authentic    # real data only
```

**Synthetic tier** — DataFrames constructed in `tests/conftest.py`, staged to match the pipeline's
column contract at each stage. Fast, self-contained, no network. Use `-m synthetic` as the inner
loop while editing.

**Authentic tier** — real `breseq` coverage tables published as GitHub Release assets: genuine
coordinate gaps, repeat regions, per-read-group columns, and copy-number variation that synthetic
frames cannot reproduce. On a cold cache the first run downloads **~105 MB**, then caches it.

Running both by default is deliberate: an opt-in tier is one people forget, and real-data coverage
then lapses without anyone noticing. If you want the fast path, ask for it explicitly.

Each dataset is pinned by sha256 in `tests/data/registry.json`, so an asset replaced in place fails
the hash check rather than silently changing what the tests measure. Downloads are cached by
[pooch](https://www.fatiando.org/pooch/); set `CNERY_TESTDATA_DIR` to relocate that cache.

Datasets ship coverage tables but **no BAM** — coverage tables are all `CNery` reads, which keeps
them small.

An unavailable dataset causes those tests to *skip* rather than fail, so offline work stays possible.
That means a default run without network access reports passes and skips together — check for skips
rather than assuming green means everything ran.

Adding tests, publishing datasets, and updating golden files are covered in
[`DEVELOPER`](DEVELOPER).

---
