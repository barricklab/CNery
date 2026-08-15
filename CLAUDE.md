# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Environment.** Two specs, for two audiences:

- `environment.yml` — the user-facing install path, as documented in `README.md`.
- `dev-environment.yml` — the contributor path. Adds `pytest` (absent from the runtime manifests, so
  the suite cannot run without it), pins `pandas<3`, drops the `pathlib` stdlib-backport entry, and
  includes `breseq`. Following the same convention as the `breseq` repo, it builds into an untracked
  `env/` inside the repo (already covered by `.gitignore`):

```bash
conda env create -f dev-environment.yml --prefix=$PWD/env
conda run -p $PWD/env <command>          # or: conda activate $PWD/env
```

**breseq is deliberately not in this environment.** The pre-release builds CNery targets hard-depend
on `cnery-prerelease`, so installing breseq would place a packaged copy of CNery in site-packages,
shadowing the working tree anywhere `pythonpath=["src"]` does not reach. CNery never invokes breseq
anyway — it reads pre-generated coverage tables. To produce such a table, see "Generating a coverage
table" in `README.md` — it needs [pre-release breseq](https://github.com/barricklab/conda), not the
bioconda release.

**Tests.** Run from the repo root. `pyproject.toml` supplies `testpaths=["tests"]`,
`pythonpath=["src"]`, and `addopts="-q --tb=short"`, so tests import `CNery.core`
with no install step and no `PYTHONPATH` fiddling.

A bare `pytest` runs **both** tiers. `-m synthetic` and `-m authentic` opt *out* of one.

```bash
conda run -p $PWD/env pytest                    # all 221; ~105 MB download on a cold cache
conda run -p $PWD/env pytest -m synthetic       # 137, offline, ~14s -- the inner loop
conda run -p $PWD/env pytest -m authentic       # 84, real coverage tables
conda run -p $PWD/env pytest tests/test_hmm.py  # one file
conda run -p $PWD/env pytest tests/test_utils.py::TestFindNearest::test_exact_match
conda run -p $PWD/env pytest -k gc_correction   # by name
```

**Running the tool.** Console entry point is `CNery` → `CNery.get_CNV:main`. Full flag list is in
`README.md` and `CNery -h`.

**Coverage tables are the only input.** No BAM, no FASTA, and breseq is never invoked — it does not
need to be on `PATH`. Inputs are positional and may mix files and directories; with no arguments the
current directory is used.

- A **file** is read whatever it is named.
- A **directory** contributes the files inside it — top level only, no recursion — whose names end in
  one of `--file-ending` (defaults `coverage.csv`, `coverage.tsv` **and** `coverage.tab`, the last
  being the legacy extension the deprecated `--table` flag writes; repeatable, and any value
  given **replaces** the defaults rather than extending them). Because all three are defaults, a
  folder holding `REL606.coverage.csv` *and* `REL606.coverage.tsv` is a duplicate-ID error, not a
  silent preference.
- `genome_id` is the basename minus the matched ending *and the single `.` before it*, so
  `my.sample.1.coverage.tsv` → `my.sample.1`. Not `os.path.splitext`, and emphatically not a split on
  the first dot: sequence IDs routinely contain dots.

`resolve_coverage_inputs` (`core.py`) owns all of this and returns an ordered `{genome_id: path}`
mapping. It raises — naming the offending paths — for a missing path, a directory that matched
nothing, or two inputs resolving to the same `genome_id`. `get_CNV.main` calls it **before** creating
any output directory, so a bad invocation leaves nothing behind.

### Format and schema are detected, never declared

`bam2cov` writes four shapes of the same table and CNery accepts all four.

- **Delimiter** — `_detect_delimiter` reads the first non-comment line and compares tab against comma
  counts. Content, not extension: a file named on the command line may be called anything, and CSV
  and TSV differ *only* in this separator (`coverage_output.cpp:219-221`), footer included. Not
  `csv.Sniffer`, which guesses from a sample and misfires on single-column input.
- **Schema** — `normalize_coverage_columns` reduces either shape to the canonical `unique_cov` /
  `redundant` pair that `preprocess` has always built internally:
  - plain output splits by strand: `unique_top_cov + unique_bot_cov`, `redundant_top_cov +
    redundant_bot_cov`;
  - `--total-only` ships `unique_cov`, `redundant_cov`, `total_cov` — breseq has already done those
    sums (`coverage_output.cpp:467-472`), so **`--total-only` loses nothing CNery uses**.
    `pct_redundant` and repeat censoring keep working, and `total_cov` is ignored as derivable.

  It is idempotent, and `preprocess` calls it too, so `preprocess` still works on a raw frame handed
  to it directly (`tests/test_authentic.py` does exactly that).

`read_coverage_table` runs the schema check the moment a file is opened. Without it a wrong-schema
file parses cleanly (`_read_coverage_table` takes column 0 as the position index whatever it holds)
and fails later as a bare `KeyError`. breseq's own `08_mutation_identification/*.coverage.tab` files
are exactly that case — position last, no `ref_base` — and are **not** valid input.

Note `-t` is breseq's deprecated `--table` flag, *not* `--total-only` (which is `-1`); it writes
ordinary TSV under the legacy `.tab` extension, which is why `coverage.tab` is a default ending.
breseq's own `08_mutation_identification/*.coverage.tab` files share that extension but not the
schema, so pointing CNery at that directory now finds them and rejects them by name.

No linter, formatter, or CI is configured.

## Architecture

Two modules: `src/CNery/get_CNV.py` (argparse + stage orchestration) and `src/CNery/core.py`
(everything else). `src/CNery/` has no `__init__.py` and resolves as a namespace package, which is
why `pythonpath=["src"]` alone is sufficient.

### The pipeline is a column-name contract

One DataFrame row per genomic window. Each stage appends columns; the next stage reads them **by
name**, not through the function signature. Passing a frame that lacks the expected column produces
wrong numbers or a `KeyError` deep inside a stage, never a clear error at the boundary.

| Column | Written by | Meaning |
| --- | --- | --- |
| `read_count_cov` | `preprocess` | median total (unique + redundant) coverage in the window |
| `norm_raw_cov` | `preprocess` | `read_count_cov` / its median |
| `pct_redundant` | `preprocess` | fraction of the window's bases overlapping repeat coverage |
| `is_deletion` / `is_redundant` | `mask_coverage_windows` | the two censoring reasons, kept separate |
| `gc_corr_norm_cov` | `apply_gc_correction` | divided by the LOWESS fit at that window's GC |
| `otr_gc_corr_norm_cov` | `apply_otr_correction` | divided by the ori→ter ramp |
| `otr_gc_corr_rdcnt_cov` | `run_HMM` | back-converted to integer read counts |
| `prob_copy_number` | `run_HMM` | Viterbi state |

Supporting columns other stages depend on: `gc_cor_med_fil` (median filter of `gc_corr_norm_cov`;
seeds the ori/ter guess in `otr_fit`), `exclude_from_fit` / `censor_reason` (`is_deletion OR
is_redundant`, and which one it was) and `gc_corr_fact` / `otr_gc_corr_fact` (the divisors, retained
for the diagnostic plots).

Each correction is split **fit / apply**, with a shared masking step in front:
`mask_coverage_windows` → `fit_gc_bias` → `apply_gc_correction`, and
`fit_otr_bias` → `apply_otr_correction`. The fit stages see only uncensored windows; the apply
stages run over every window.

Adding a stage means reading the previous column name and writing the next one.

### `--bias` works by aliasing columns

`get_CNV.py:212-265` implements the four modes by *renaming data into the column the next stage
expects*, rather than by branching inside the correction functions. `run_HMM` unconditionally reads
`otr_gc_corr_norm_cov`; `fit_otr_bias` unconditionally reads `gc_corr_norm_cov`.

- `all` — no aliasing; both corrections run in sequence.
- `gc` — copy `gc_corr_norm_cov` → `otr_gc_corr_norm_cov`, skipping OTR entirely.
- `otr` — copy `norm_raw_cov` → `gc_corr_norm_cov` so `fit_otr_bias` sees uncorrected input.
- `none` — copy `norm_raw_cov` → `otr_gc_corr_norm_cov`.

Any new bias mode is another aliasing branch here, not a new parameter threaded through `core.py`.

Note that GC correction has *already run* by the time this dispatch executes — `process_multi_genome`
does it unconditionally, and the `otr`/`none` branches discard the result by overwriting the column.
That is also why `is_deletion` / `is_redundant` are present on the frame in all four modes:
`mask_coverage_windows` runs as part of that unconditional GC stage.

### Multi-genome flow

`process_multi_genome` is the top of the pipeline. It takes the `{genome_id: path}` mapping from
`resolve_coverage_inputs` and handles every table in one pass:

1. Per table: `read_coverage_table` → `preprocess` → tag rows with `genome_id`.
2. **Pool all tables into one frame**, renormalize `norm_raw_cov` against a single global median,
   and run one shared `mask_coverage_windows` → `fit_gc_bias` → `apply_gc_correction` across the
   pool.
3. Split back apart on `genome_id` and return `{genome_id: df}`.

GC bias is deliberately fitted globally across chromosome + plasmids + contigs (one pooled diagnostic
plot). OTR correction and CN calling are then per-reference. `genome_id` is both the split key and the
source of every output plot/CSV filename, so it must survive any transformation you add.

The shared global median in step 2 is why one invocation should carry the references of **one
sample**. Passing two samples' tables together cross-normalizes them against each other.

Note that `process_multi_genome` does no globbing: resolution is the caller's job, so that bad inputs
are rejected before `get_CNV.main` creates any output directory.

### Deliberate behavior that reads like a bug

- `preprocess` (`core.py:147`) **keeps** every window overlapping redundant (repeat) coverage,
  records the overlapping fraction as `pct_redundant`, and takes the window median over *total*
  (unique + redundant) coverage so a repeat's real depth is reflected. Window ordinal is therefore
  proportional to genomic coordinate. It does drop the **trailing partial window** at the genome
  end, so the count is exactly `(genome_len - win) // step + 1` — a short final window would take
  its median over fewer bases and would set the genome-end coordinate for the last CN segment and
  for the terminus fallback in `apply_otr_correction`.
- Censoring is one flag with three consumers. `mask_coverage_windows` (`core.py:337`) turns
  `pct_redundant > 0` into `is_redundant`, which excludes the window from the GC LOWESS fit
  (`fit_gc_bias`), from the OTR fit and its ori/ter peak-trough search (`fit_otr_bias`, `otr_fit`),
  and from the Viterbi observation sequence and emission-model estimate (`run_HMM`). It is never
  dropped from the frame: it still gets a bias-corrected coverage value and inherits
  `prob_copy_number` from the segment it sits in. Leaving repeat windows in the HMM made pile-ups
  invent their own high-CN segments and split genuine deletions in two — see
  `TestCopyNumber::test_no_segment_starts_on_a_repeat_window` in `tests/test_authentic.py`.
- `apply_gc_correction` (`core.py:448`) **freezes near-zero windows at exactly `0.0`**, so real
  deletions still get called CN=0 rather than being divided back up toward 1. Note this applies to
  `is_deletion` only, not `is_redundant` — the two censoring reasons are kept separate precisely
  because they are treated differently here. Asserted directly in `tests/test_gc_correction.py`
  and `tests/test_regression.py`.
- `apply_otr_correction` (`core.py:767`) applies the OTR factor everywhere except `is_deletion`
  windows, for the same reason.
- The HMM (`core.py:920-1021`) stacks a geometric zero-state row on top of one negative-binomial
  emission row per copy number, so **state index == copy number** and the matrices are
  `n_states + 1` square/rows. The negative binomial (not Poisson) is intentional: coverage is
  overdispersed, and `run_HMM` nudges `var` above `mean` when a synthetic-flat input would otherwise
  make `solve_pr` divide by a non-positive number.
- Output subdirectories (`CNV_plt/`, `CNV_csv/`, `GC_bias/`, `OTR_corr/`) are created once in
  `get_CNV.py`, *after* inputs are resolved so a bad invocation creates nothing; the writer
  functions in `core.py` assume they already exist.

## Testing conventions

Both tiers run by default. Tests needing real coverage tables are marked `authentic`; everything
else gets a `synthetic` marker applied automatically by `pytest_collection_modifyitems` in
`tests/conftest.py`, so the two are exhaustive and mutually exclusive and a new test cannot escape
both. Never hand-mark a test `synthetic`.

Synthetic tests construct windowed DataFrames staged to match the column
contract at each pipeline point — reuse the `tests/conftest.py` fixtures (`windowed_flat`,
`windowed_with_deletion`, `windowed_with_amplification`, `gc_corrected_flat`, `otr_corrected_flat`)
rather than hand-rolling frames, so a change to the contract surfaces in one place.
These fixtures deliberately carry **no** `pct_redundant` / `is_redundant` column, so they exercise
the uncensored path; `run_HMM` falls back to using every window when the column is absent or when
censoring would leave fewer than `min_called_windows`. Add the column explicitly when a test is
about censoring behavior.

Because the writers assume their output directories exist, any test that reaches
`apply_otr_correction`, `run_HMM`, or a plot function must `os.makedirs` the subdirs under
`tmp_path` first — see `_ensure_dirs` in `tests/test_integration.py` and
`tests/test_otr_correction.py`.

Nothing in the suite mocks a subprocess: CNery never runs breseq, so tests that need a coverage
table write one. Tables are read by `_read_coverage_table` (`core.py`), which strips the trailing
summary block by its `#` prefix — the block is **variable-length** (four lines by default, +1 under
`--show-average`, +3 per read group, and none at all in breseq's own
`08_mutation_identification/*.coverage.tab`), so never assume a fixed count. `TestFooterHandling` in
`tests/test_coverage_table_io.py` covers each of those shapes.

Delimiter, footer and schema detection are covered in `tests/test_coverage_table_io.py`, including
an equivalence class proving a `--total-only` table and a strand-split one of the same data give
identical windowed output. Input resolution and the file-ending/`genome_id` rules are in
`tests/test_inputs.py`;
`tests/test_cli.py` drives `main()` through argparse with `monkeypatch.setattr(sys, "argv", ...)`,
which is the only place the real command-line surface is exercised.

## Authentic datasets

Real coverage tables live as GitHub Release assets, not in the repo. They are fetched and cached by pooch,
verified against a sha256 pinned in `tests/data/registry.json`, and used only by tests marked
`@pytest.mark.authentic`.

```bash
conda run -p $PWD/env pytest -m authentic       # this tier only
CNERY_TESTDATA_DIR=/big/disk conda run -p $PWD/env pytest   # relocate the ~105 MB cache
```

Three datasets, differing where it matters — 0/1/2 read groups, 10/18/26 columns, and **4/7/10-line
footers**. That last spread is deliberate: none is 4, so they are real regression coverage for the
prefix-based footer stripping that replaced `skipfooter=4`.

Check the result says *passed*, not *skipped* — an unfetchable dataset skips (with the dataset, URL,
and error in the reason) rather than failing, which keeps offline work possible but can look like
success at a glance.

**Layout.** One release per dataset version, tagged `testdata-<name>-v<N>`, holding a single
`<name>.tar.gz`. Adding or revising a dataset touches only its own registry entry — nothing else is
regenerated or re-uploaded. This is why `tests/data/_fetch.py` passes pooch a per-file `urls=` map
instead of pooch's global `version=`, which would stamp one version across every dataset.

Identity rests on the **sha256, not the tag**: an asset replaced in place fails the hash check. Never
overwrite a published asset — cut a new version instead.

Heavy *inputs* go to Releases; expected *outputs* (golden CN calls) stay tracked in-repo under
`tests/data/expected/`, so a behavior change shows up as a reviewable diff. Prefer structural
assertions (breakpoint coordinates, CN state per segment, ori/ter window) over float-exact goldens —
LOWESS and `scipy.optimize.minimize` results drift across library versions.

**Adding a dataset:**

```bash
python tests/data/add_testdata.py <folder> --name lambda --description "..."   # dry run, next version
python tests/data/add_testdata.py <folder> --name lambda --publish             # cut the release
python tests/data/add_testdata.py <folder> --name lambda --version 3           # pin explicitly
```

It tars the folder, computes the sha256, creates the release, and prints the `registry.json` entry to
paste. It also drops a `dataset.json` provenance stub (breseq version, command line, accessions, what
the dataset exercises) into the folder — fill that in, since authentic data without provenance stops
being reproducible once whoever generated it moves on.

`--version` defaults to one past the highest already published, checking both the GitHub releases and
`registry.json` (they can disagree — a release cut but not yet pasted in, or an entry whose tag was
deleted). Reusing a published version is refused, and that check runs *before* the archive is built,
so a clash surfaces on the dry run. If `gh` can't be reached it refuses to guess a version rather than
risk picking v1 for a dataset that already exists; an explicit `--version` proceeds with a warning.

The fetch machinery itself is tested in `tests/test_testdata_registry.py`, which is deliberately
*not* marked `authentic`: it serves a synthetic archive over localhost, so registry and download
regressions surface in the default run rather than waiting for someone to opt in.
