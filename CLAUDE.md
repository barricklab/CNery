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
shadowing the working tree anywhere `pythonpath=["src"]` does not reach. Tests therefore run against
pre-generated coverage tables and never shell out to breseq. To produce such a table, see
"Generating a coverage table" in `README.md` — it needs
[pre-release breseq](https://github.com/barricklab/conda), not the bioconda release.

**Tests.** Run from the repo root. `pyproject.toml` supplies `testpaths=["tests"]`,
`pythonpath=["src"]`, and `addopts="-q --tb=short"`, so tests import `CNery.core`
with no install step and no `PYTHONPATH` fiddling.

A bare `pytest` runs **both** tiers. `-m synthetic` and `-m authentic` opt *out* of one.

```bash
conda run -p $PWD/env pytest                    # all 147; ~105 MB download on a cold cache
conda run -p $PWD/env pytest -m synthetic       # 75, offline, ~10s -- the inner loop
conda run -p $PWD/env pytest -m authentic       # 72, real coverage tables
conda run -p $PWD/env pytest tests/test_hmm.py  # one file
conda run -p $PWD/env pytest tests/test_utils.py::TestFindNearest::test_exact_match
conda run -p $PWD/env pytest -k gc_correction   # by name
```

**Running the tool.** Console entry point is `CNery` → `CNery.get_CNV:main`. Run it inside a breseq
output folder containing `data/`, or point at one with `CNery -i <breseq_output_dir>`. Full flag list
is in `README.md` and `CNery -h`.

`breseq` must be on `PATH` — `bam2cov_to_df` shells out to `breseq bam2cov` — unless coverage tables
are supplied, in which case breseq is never invoked and no BAM is needed. Two sources, explicit first:

- `--coverage-dir <dir>` → `<dir>/<seq_id>.coverage.**tsv**`. Required for every sequence in the
  FASTA; missing ones raise before any processing. `coverage_table_path` in `core.py` owns this name.
- otherwise `<breseq_dir>/08_mutation_identification/<seq_id>.coverage.**tab**` if present. Note the
  different suffix: these are breseq's own files, in a **different schema** (position last, no
  `ref_base`), and keep their historical name.

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
| `read_count_cov` | `preprocess` | median unique coverage in the window |
| `norm_raw_cov` | `preprocess` | `read_count_cov` / its median |
| `gc_corr_norm_cov` | `gc_correction` | divided by the LOWESS fit at that window's GC |
| `otr_gc_corr_norm_cov` | `otr_correction` | divided by the ori→ter ramp |
| `otr_gc_corr_rdcnt_cov` | `run_HMM` | back-converted to integer read counts |
| `prob_copy_number` | `run_HMM` | Viterbi state |

Supporting columns other stages depend on: `gc_cor_med_fil` (median filter of `gc_corr_norm_cov`;
seeds the ori/ter guess in `otr_fit`) and `gc_corr_fact` / `otr_gc_corr_fact` (the divisors, retained
for the diagnostic plots).

Adding a stage means reading the previous column name and writing the next one.

### `--bias` works by aliasing columns

`get_CNV.py:212-265` implements the four modes by *renaming data into the column the next stage
expects*, rather than by branching inside the correction functions. `run_HMM` unconditionally reads
`otr_gc_corr_norm_cov`; `otr_correction` unconditionally reads `gc_corr_norm_cov`.

- `all` — no aliasing; both corrections run in sequence.
- `gc` — copy `gc_corr_norm_cov` → `otr_gc_corr_norm_cov`, skipping OTR entirely.
- `otr` — copy `norm_raw_cov` → `gc_corr_norm_cov` so `otr_correction` sees uncorrected input.
- `none` — copy `norm_raw_cov` → `otr_gc_corr_norm_cov`.

Any new bias mode is another aliasing branch here, not a new parameter threaded through `core.py`.

Note that GC correction has *already run* by the time this dispatch executes — `process_multi_genome`
does it unconditionally, and the `otr`/`none` branches discard the result by overwriting the column.

### Multi-genome flow

`process_multi_genome` (`core.py:379`) is the top of the pipeline and handles every record in the
BAM/FASTA in one pass:

1. Per FASTA record: `bam2cov_to_df` → `preprocess` → tag rows with `genome_id = <fasta header>`.
2. **Pool all records into one frame**, renormalize `norm_raw_cov` against a single global median,
   and run one shared LOWESS `gc_correction` across the pool.
3. Split back apart on `genome_id` and return `{header: df}`.

GC bias is deliberately fitted globally across chromosome + plasmids + contigs (one pooled diagnostic
plot). OTR correction and CN calling are then per-reference. `genome_id` is both the split key and the
source of every output plot/CSV filename, so it must survive any transformation you add.

### Deliberate behavior that reads like a bug

- `preprocess` (`core.py:182`) **discards any window overlapping redundant (repeat) coverage**. Window
  ordinal is therefore not proportional to genomic coordinate, and the window count falls short of
  `genome_len / step`.
- `gc_correction` (`core.py:313`) excludes near-zero windows from the LOWESS fit and **freezes them at
  exactly `0.0`** in the output, so real deletions still get called CN=0 rather than being divided
  back up toward 1. Asserted directly in `tests/test_gc_correction.py` and `tests/test_regression.py`.
- `otr_correction` (`core.py:866`) applies the OTR factor only where coverage exceeds 10% of the
  median, for the same reason.
- The HMM (`core.py:955-1067`) stacks a geometric zero-state row on top of one negative-binomial
  emission row per copy number, so **state index == copy number** and the matrices are
  `n_states + 1` square/rows. The negative binomial (not Poisson) is intentional: coverage is
  overdispersed, and `run_HMM` nudges `var` above `mean` when a synthetic-flat input would otherwise
  make `solve_pr` divide by a non-positive number.
- Output subdirectories (`CNV_plt/`, `CNV_csv/`, `GC_bias/`, `OTR_corr/`) are created once in
  `get_CNV.py:158-160`; the writer functions in `core.py` assume they already exist.

## Testing conventions

Both tiers run by default. Tests needing real coverage tables are marked `authentic`; everything
else gets a `synthetic` marker applied automatically by `pytest_collection_modifyitems` in
`tests/conftest.py`, so the two are exhaustive and mutually exclusive and a new test cannot escape
both. Never hand-mark a test `synthetic`.

Synthetic tests construct windowed DataFrames staged to match the column
contract at each pipeline point — reuse the `tests/conftest.py` fixtures (`windowed_flat`,
`windowed_with_deletion`, `windowed_with_amplification`, `gc_corrected_flat`, `otr_corrected_flat`,
`single_fasta`) rather than hand-rolling frames, so a change to the contract surfaces in one place.

Because the writers assume their output directories exist, any test that reaches `otr_correction`,
`run_HMM`, or a plot function must `os.makedirs` the subdirs under `tmp_path` first — see
`_ensure_dirs` in `tests/test_integration.py` and `tests/test_otr_correction.py`.

For anything touching the breseq subprocess, follow `tests/test_bam2cov_io.py`: patch
`subprocess.run` with a side effect that writes a fake `.tab` file. Coverage tables are read by
`_read_coverage_tab` (`core.py`), which strips the trailing summary block by its `#` prefix — the
block is **variable-length** (four lines by default, +1 under `--show-average`, +3 per read group,
and none at all in breseq's own `08_mutation_identification/*.coverage.tab`), so never assume a fixed
count. `TestFooterHandling` in that file covers each of those shapes.

## Authentic datasets

Real BAM/FASTA live as GitHub Release assets, not in the repo. They are fetched and cached by pooch,
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
