# CNery

*breseq* copy-number-variation extension. `CNery` reads per-reference coverage tables produced by [*breseq*](https://github.com/barricklab/breseq) `bam2cov` and predicts copy-number variation (CNV) across the genome. Predictions are corrected for coverage biases introduced by sequencing chemistry (GC-content bias) and prokaryotic replication state during DNA isolation (origin-to-terminus / OTR bias).

Recent updates (latest commits):

- **Coverage tables are the only input** — `CNery` reads *breseq* `bam2cov` coverage tables and nothing else. It no longer needs a BAM or a reference FASTA, and never runs `breseq` itself. The reference sequence it needs for GC content is already in the table's own `ref_base` column.
- **CSV or TSV, full or `--total-only`** — all four shapes of table are read without being declared. The delimiter is detected from the file's own header row, and the column schema from its column names. `--total-only` tables are the recommended input: they carry three coverage columns instead of eight, so they are markedly smaller, and CNery loses nothing by using them.
- **Files and folders on the command line** — name coverage tables directly, or name folders and let `CNery` find them by file ending (`--file-ending`, defaults `coverage.csv`, `coverage.tsv` and `coverage.tab`). Files and folders can be mixed in one command.
- **Multi-genome CNV analysis** — `CNery` processes *all* the coverage tables given in one pass. Each reference (chromosome, plasmid, contig, etc.) is preprocessed separately, pooled for a shared LOWESS GC-bias fit, and then bias-corrected and CN-called independently.
- **Cumulative GC skew origin/terminus prediction** — `CNery` now also locates the replication origin and terminus from the reference's own cumulative GC skew ([Grigoriev 1998](https://academic.oup.com/nar/article/26/10/2286/1030593)), independently of read depth. Reported in `GC_skew/` as a marked-up plot and a JSON summary, and used as the fallback origin/terminus when the coverage profile carries no usable gradient of its own.
- **Output flexibility** — output prefix defaults to `CNV_out/` in the current folder. Output subfolders (`CNV_plt/`, `CNV_csv/`, `GC_bias/`, `OTR_corr/`, `GC_skew/`) are created automatically.
- **Modular bias correction** — the `--bias` flag lets you choose `all` (GC + OTR), `gc`, `otr`, or `none`.
- **A per-base segment-length prior** — `--change-rate` is the probability per *base* that copy
  number changes, so re-tiling a genome with different `-w`/`-s` does not restate the biology.
  Read `1/rate` as the expected segment length.
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
CNery REL606.coverage.csv pPlasmid.coverage.csv -o CNV_out
CNery coverage/ extra/pContig.coverage.csv -o CNV_out
```

Folders are searched, top level only, for files ending in `coverage.csv`, `coverage.tsv` or
`coverage.tab` — all three are found by default, and they may sit side by side. (`.tab` is the
legacy extension breseq's deprecated `--table` flag writes; its contents are ordinary TSV.) Use
`--file-ending` if your tables are named otherwise; repeat the flag to accept several. Note that any `--file-ending` **replaces** the defaults
rather than adding to them:

```bash
CNery coverage/ --file-ending cov.txt
CNery coverage/ --file-ending cov.txt --file-ending coverage.csv
```

A table's sequence ID comes from its file name, with the matched ending and the `.` in front of it
removed — `REL606.coverage.csv` becomes `REL606`, and `NC_012967.1.coverage.tsv` becomes
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

Analyze coverage across the whole genome, but restrict the CNV plot to a specific genomic segment:

```bash
CNery <inputs> -o <output folder> --region REL606:3497890-3955678 -w 1000 -s 500
```

The sequence ID is the one derived from the table's file name. `SEQ_ID:` may be omitted when the run
has only one input sequence:

```bash
CNery REL606.coverage.csv -o CNV_out --region 3497890-3955678
```

Repeat the flag to plot several sequences, at most once each:

```bash
CNery coverage/ -o CNV_out --region REL606:3497890-3955678 --region pPlasmid:1-40000
```

Open intervals work too — `REL606:3497890-` runs to the end of that sequence, `REL606:-3955678` from
its start.

Two things to know. **Giving any `--region` also selects which sequences are plotted**: a sequence
not named gets no CNV plot. And `--region` affects **plotting only** — coverage, bias fitting and
copy-number calling always cover every sequence, and the output CSVs always contain every window for
every sequence, plotted or not.

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

When OTR correction is applied, the origin and terminus of replication are automatically inferred — no manual coordinates are required. CNery fits them from the coverage profile, falling back to the reference's own cumulative GC skew when the coverage carries no usable gradient. Either way the correction is applied only if it beats a bootstrap null, so a flat genome is left alone.

---

## Generating a coverage table

Coverage tables are `CNery`'s only input. Generate them once with `breseq bam2cov`, then run `CNery`
against them as often as you like — no BAM, no reference FASTA, and `breseq` need not be installed
on the machine that runs `CNery`. The reference sequence `CNery` needs for GC content is already in
each table's `ref_base` column.

The conventional layout is one table per reference sequence, named `<seq_id>.coverage.csv` (or
`.coverage.tsv` — both are found by default):

```
coverage/
├── REL606.coverage.csv
└── pPlasmid.coverage.csv
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
  --format CSV \
  --total-only \
  --resolution 0 \
  --output coverage \
  -b data/reference.bam \
  -f data/reference.fasta
```

To generate a single table instead, name the region using the sequence ID exactly as it appears in
the FASTA, and give `--output` a name ending in `.coverage.csv`:

```bash
breseq bam2cov \
  --format CSV \
  --total-only \
  --region REL606:1-4629812 \
  --resolution 0 \
  --output coverage/REL606.coverage.csv \
  -b data/reference.bam \
  -f data/reference.fasta
```

Four options matter:

- **`--resolution 0`** outputs every position. The default is 600, which samples only 600 points
  across the whole region — far too sparse for windowed coverage, and it fails silently by producing
  a well-formed but nearly empty table.
- **`--format CSV`** produces a comma-separated table instead of a plot; `TSV` gives the same
  columns tab-separated. `CNery` reads either and works out which from the file itself, so the
  choice is yours. (The old `-t` / `--table` flag is **deprecated**: it still selects TSV, but writes
  the legacy `.tab` extension. `CNery` matches that too, but prefer `--format TSV`.)
- **`--total-only`** (short flag `-1`) writes `unique_cov`, `redundant_cov` and `total_cov` in place
  of the eight strand-split coverage columns — roughly 2.5× smaller files. `breseq` sums the strands
  itself, and those sums are exactly what `CNery` computes from the wider table, so **nothing CNery
  uses is lost**: repeat detection via `redundant_cov` still works, and copy-number calls are
  unchanged. Recommended.
- **`--per-read-group`** *(optional)* repeats every coverage column once per read group (`@RG`) in
  the BAM, prefixed `RG-<n>_` where `<n>` is the read group's index in the BAM header. Requires a
  table format; it is rejected with `--format PNG`. A BAM with no read groups yields a single `RG-0`
  set.

```bash
breseq bam2cov \
  --format CSV \
  --total-only \
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
block. With `--total-only --format CSV`:

```
position,ref_base,unique_cov,redundant_cov,total_cov
1,G,32,0,32
2,G,32,0,32
...
#,region_unique_average_cov,56.7267
#,region_repeat_average_cov,0
#,region_average_cov,56.7267
#,number_of_positions,4629812
```

Without `--total-only`, each count is split by strand and six further columns follow:

```
position	ref_base	unique_top_cov	unique_bot_cov	redundant_top_cov	redundant_bot_cov	...
1	G	14	18	0	0	...
```

The delimiter is the only difference between `--format CSV` and `--format TSV`; it is used for the
header, the data and the footer alike.

That summary block is variable in length — `--show-average` adds a line, and `--per-read-group` adds
three per group (`#	RG-0_region_unique_average_cov	…`) — so anything parsing these tables should
skip lines by their `#` prefix rather than dropping a fixed number from the end.

For the same reason, do not assume a fixed column count: `position` must be the first column and the
named coverage columns must be present, but extra columns to the right are expected and should be
ignored rather than treated as an error.

`CNery` needs `position`, `ref_base`, and unique-versus-redundant coverage in one of the two shapes
above — `unique_cov` + `redundant_cov`, or the four strand-split `unique_*`/`redundant_*` columns. It
checks for them as soon as a table is opened, so a file with the wrong schema is rejected by name
instead of failing later. `total_cov` is ignored: it is `unique_cov + redundant_cov` by construction.
Note that *breseq*'s own `08_mutation_identification/*.coverage.tab` files use a **different** schema
(position last, no `ref_base`) and are not usable as `CNery` input.

If you do need the sequence IDs and lengths for a `--region` argument, read them from the FASTA
headers or the `.fai` index:

```bash
grep '^>' data/reference.fasta
cut -f1,2 data/reference.fasta.fai
```

---

## Outputs

Given an output folder `CNV_out/`, `CNery` writes:

The `-f` fragment size is optional. GC bias acts at the scale of the sequenced
fragment, so that size belongs to the library rather than to the analysis — and
it is not something a coverage table shows you. Left unset, CNery scores
candidate sizes by how well the GC each implies predicts held-out coverage (with
the replication ramp divided out and copy-number variants excluded, so neither
can be mistaken for a GC effect) and reports what it chose. The 400 bp default is
kept unless a candidate beats it by more than the measurement's own error. Pass
`-f` to pin it.

CNery corrects coverage and calls copy number in **two passes**. The first runs GC
correction, origin-to-terminus correction and the HMM as usual. The second repeats
both fits with every window the first pass did not call single-copy excluded from
them, then calls copy number again — only the second pass's results are written.

An amplification is invisible to the crude censoring the first pass has available
(near-zero depth, repeat overlap), so it otherwise sits in the GC and ramp fits at
full weight and distorts them. Measured on synthetic coverage where the truth is
known, a real 1.5x replication ramp with an amplification on top is detected 0 times
in 60 without this and 60 times in 60 with it, at no cost in false positives. If
excluding the non-single-copy windows would leave under half a sequence, it is
skipped for that sequence and reported as such.

- `CNV_out/CNV_plt/` — per-reference CNV prediction plots.
- `CNV_out/CNV_csv/` — per-window coverage + CN calls as CSV.
- The GC correction is a fitted curve, not an exact quantity, so how well it is determined at each window's GC is measured (by resampling the fit) and carried into the copy-number model as extra variance. The effect grows with copy number, because a correction factor's error is multiplied by the number of copies — which is why a window at an extreme GC inside an amplification is no longer able to earn its own copy-number segment on the strength of the correction alone.
- `CNV_out/GC_bias/` also holds `*_GC_passes.pdf` — the GC correction is fitted in two pooled passes, once on raw coverage and again after OTR correction (which reintroduces a GC trend, because the replication ramp varies with position and position correlates with GC). The second pass additionally excludes every window the first pass's copy-number calls did not put at CN=1. The plot shows both curves and their product, which is the total correction actually applied.
- `CNV_out/GC_bias/` — pooled LOWESS GC-bias diagnostic plot.
- `CNV_out/corr_plots/` — per-reference **before/after** diagnostic (`*_correction_stages.pdf`): one row for each correction step — GC and OTR, in each of the two passes — showing coverage before and after it, with the fitted curve overlaid, and directly beneath each row a strip of which windows that particular fit was allowed to see. Deletions are drawn as spans, repeats as a density track, and the second pass's strips additionally mark everything called CN≠1. One row does not continue from the one above it, and says so: the second OTR fit divides the first pass's ramp back out, because a ramp has to be fitted to coverage that still contains it. Each row is labelled with the fraction of windows within 20% of single copy before and after, reported both over all windows and over uncensored windows only — the two can differ a lot on a repeat-heavy replicon, and the strip below shows why. Produced in every `--bias` mode.

- `CNV_out/OTR_corr/` — per-reference OTR bias plots and a JSON summary (`*_otr_results.json`) containing the inferred origin window, terminus window, normalized coverage at each, the origin-to-terminus ratio, and the sequence's **relative copy number**.

  It also reports how well the applied ramp actually fits, beyond r²: `"Residual structure score"` asks whether what the tent *failed* to explain is systematically structured rather than just noisy — a fit can explain a lot of variance and still be the wrong shape over a long stretch. It is a z-score against a bootstrap null, so read it as roughly: below 1 unstructured, 1–2 mild, above 2 structured. `"Residual decorrelation length (bp)"` is the scale of correlation that would be present anyway, published alongside so the score can be judged in context. Both are diagnostic only — nothing in CNery acts on them.

  It also records the evidence behind the decision, whether or not a correction was applied: `"Coverage fit r-squared"` / `"Coverage fit p-value"`, the same pair for the GC-skew-anchored fit, the `"Coverage vs skew likelihood ratio"` and its p-value when both candidates were live, `"Bootstrap surrogates"`, and `"Breakpoint source"` (`coverage fit`, `GC skew`, or `not corrected`). A rejected fit is therefore diagnosable from the file alone. Note the p-values are floored at `1/(surrogates+1)`, so `0.001` is an upper bound rather than a measurement.

  `"Relative copy number"` is that sequence's coverage relative to the longest sequence in the run, which reads exactly `1.0`. It is not rounded to an integer: a plasmid at `2.95` is a measurement, and it is the only place plasmid copy number is reported — `prob_copy_number` in the CSVs is called per reference, so a uniformly multi-copy plasmid comes out as `1` there.
- `CNV_out/GC_skew/` — per-reference cumulative GC-skew plots with the predicted origin and terminus marked, and a JSON summary (`*_gc_skew_results.json`).

Each coverage table produces its own set of outputs, named with the sequence ID derived from its file name. The GC-bias plot is the exception: one pooled fit covers every table in the run.

### Origin and terminus from GC skew

Bacterial genomes are G-rich on the leading strand and C-rich on the lagging strand, so the sign of the GC skew `(G−C)/(G+C)` flips at the two points where the replication strands switch. Summing the skew along the genome turns those sign changes into extrema: following [Grigoriev 1998](https://academic.oup.com/nar/article/26/10/2286/1030593), the cumulative curve reaches its **minimum over the replication origin** and its **maximum at the terminus**.

`CNery` computes this from the coverage table's `ref_base` column — no FASTA and no read depth involved — and writes the result for every reference, in all four `--bias` modes:

```json
{
    "Origin (bp)": 3885501,
    "Terminus (bp)": 1526001,
    "Origin window index": 7771,
    "Terminus window index": 3052,
    "Windows": 9258,
    "Separation (fraction of genome)": 0.4903,
    "Cumulative skew amplitude": 151.2949,
    "Replichore skew t-statistic": 34.77,
    "Replichore skew p-value": 0.001,
    "Bootstrap surrogates": 1000,
    "Prediction confident": true,
    "Prediction method": "Ori-ter coordinates from cumulative GC skew (Grigoriev 1998)"
}
```

`Prediction confident` requires two things: the two extrema roughly antipodal (35–65% of the sequence apart, as bidirectional replication implies), and a p-value of 0.01 or better. **The coordinates are reported either way** — a low-confidence call stays diagnosable from the JSON and the plot rather than being reduced to a flag.

#### How the p-value is computed, and how to read it

Adjacent windows are not independent — genome composition varies on scales far longer than one window — so an ordinary *t*-test would badly overstate significance. `Replichore skew t-statistic` is therefore reported as an **effect size only**; its magnitude is inflated by an unknown factor and should not be converted to a p-value.

The p-value instead comes from a **circular block bootstrap**. Contiguous blocks of windows are resampled with replacement around the circle, which preserves local autocorrelation while destroying the long-range two-arm pattern being tested for; the null is "a sequence that wobbles like this one but has no single origin". The full procedure — locating the extrema, then scoring the two arms — is re-run on every surrogate, so choosing the breakpoints by looking at the data is paid for rather than ignored.

Two things to know when reading it:

- **It is floored at `1/(B+1)`.** A real chromosome beats all 1,000 surrogates and reads back exactly `0.001`. That is an upper bound, not a measurement — read it as "p < 0.001". `Bootstrap surrogates` is reported so the floor is visible.
- **Block length adapts to sequence length.** What governs power is the number of blocks rather than their size, so CNery targets ~20 blocks, bounded to 10–200 windows each. Sequences too short to give both long-enough and numerous-enough blocks simply do not reach significance, which is an honest reflection of how little evidence they carry.

The bootstrap is seeded, so repeated runs on the same input give the same p. It adds roughly 3% to the per-sequence cost (~0.15 s on a 4.6 Mb genome).

Expect `false` on plasmids. They have no bidirectional replication origin, so there is no sign change for the cumulative curve to turn on, and the two extrema you get back are noise — the two in the test data land at `p` = 0.36 and 0.44. That is the intended answer, not a failure, and chromosomes in the same run are unaffected since the prediction is made per reference.

Two properties worth knowing. The prediction depends only on the reference, so different samples aligned to the same reference give identical answers, whatever their depth or growth phase. And it is invariant to circular permutation: a reference whose coordinates start elsewhere predicts the same locus.

These values feed the OTR correction as its second candidate. CNery prefers the coverage-derived origin and terminus when they clear their own significance test *and* a likelihood-ratio test says they fit better than a ramp hinged at the GC-skew coordinates; otherwise it uses the skew's, provided the skew prediction is confident and the coverage does not contradict which end is the origin. `"Correction type"` and `"Breakpoint source"` in `*_otr_results.json` say which was used.

The practical consequence: a sequence with no replication gradient in its coverage but a confident skew prediction now receives a small correction where it previously received none. The magnitude is still fitted from the coverage, not imported — on a genuinely flat sequence the fitted ramp is close to 1.0 — but the evidence for such a correction is the reference sequence, not the reads. `"GC skew fit p-value"` reports what the coverage itself had to say about it.

---

## All command-line options

```
$ CNery -h

usage: CNery [-h] [--file-ending ENDING] [--region SEQ_ID:START-END] [-o O]
             [-w W] [-s S] [-f F]
             [-z DELETION_COVERAGE_FRACTION] [--change-rate CHANGE_RATE]
             [--bias {all,none,gc,otr}]
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
                        one. Any --file-ending REPLACES the defaults
                        ('coverage.csv', 'coverage.tsv', 'coverage.tab')
                        rather than adding to them. A file named directly on
                        the command line is always used, whatever it is
                        called.
  --region SEQ_ID:START-END
                        Plot the CNV calls for one sequence over a genomic
                        segment, e.g. 'REL606:3497890-3955678'. The sequence
                        ID is the one derived from the table's file name.
                        Repeat the flag to plot several sequences, at most
                        once each. 'SEQ_ID:' may be omitted when the run has
                        only one input sequence. Open intervals are accepted:
                        'REL606:3497890-' runs to the end of the sequence,
                        'REL606:-3955678' from its start. Giving any --region
                        also selects WHICH sequences are plotted: those not
                        named get no CNV plot. This affects plotting only --
                        coverage, bias fitting and copy-number calling always
                        cover every sequence, and the output CSVs always
                        contain every window.
  -o, --output O        output file prefix / storage location. Defaults to
                        the 'CNV_out' folder in the current dir.
  -w, --window W        Window length used to parse the genome and compute
                        coverage and GC statistics. Default: 100. Wider
                        windows smooth the coverage but lose short events: the
                        window statistic is a per-base median, whose precision
                        grows sublinearly with width, so -w is a resolution
                        knob.
  -s, --step-size S     Step size (<= window size) for each progression of
                        the window across the genome. Set step-size = window
                        size for non-overlapping windows. Default: 100, i.e.
                        non-overlapping. Copy-number calls are near-invariant
                        to this: the state-change prior is per base (see
                        --change-rate) and overlapping windows are
                        down-weighted so they do not count the same bases
                        twice.
  -f, --frag-size F     Average fragment size of the sequencing library. GC%
                        is measured over this many bases centred on each
                        window. Ignored when smaller than -w. Default: 400.
  -z, --deletion-coverage-fraction DELETION_COVERAGE_FRACTION
                        Coverage a deleted region still shows, as a fraction of
                        the single-copy level. Sets the mean of the
                        copy-number-0 emission. Real deletions are not empty --
                        mismapping and repeat spill leave a couple of percent
                        behind. A fraction rather than an absolute depth so
                        that what counts as a deletion does not change with how
                        deeply the sample was sequenced. Default: 0.02.
  --change-rate CHANGE_RATE
                        Prior probability PER BASE that copy number changes.
                        The per-window probability is 1 - exp(-rate * step-
                        size), so changing -w/-s no longer changes the
                        implied biology. Read 1/rate as the expected segment
                        length: the default 1e-06 is one copy-number boundary
                        per megabase. Larger values give more, shorter
                        segments.
  --bias {all,none,gc,otr}
                        Select which bias correction to apply before CN
                        prediction. 'all' applies GC + OTR, 'gc' or 'otr'
                        applies only that one, 'none' skips bias correction.
                        Default: all.

Inputs are breseq 'bam2cov' coverage tables (CSV or TSV). Run with no
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
