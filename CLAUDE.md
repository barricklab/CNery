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
conda run -p $PWD/env pytest                    # all 560; ~105 MB download on a cold cache
conda run -p $PWD/env pytest -m synthetic       # 263, offline, ~35s -- the inner loop
conda run -p $PWD/env pytest -m authentic       # 297, real coverage tables
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
| `gc_skew` | `preprocess` | `(G-C)/(G+C)` over the window's `ref_base` letters |
| `cum_gc_skew` | `preprocess` | running sum of `gc_skew`, mean-subtracted first |
| `is_deletion` / `is_redundant` | `mask_coverage_windows` | the two censoring reasons, kept separate |
| `gc_corr_norm_cov` | `apply_gc_correction` | divided by the LOWESS fit at that window's GC |
| `otr_gc_corr_norm_cov` | `apply_otr_correction` | divided by the ori→ter ramp |
| `is_cn_variant` | `add_cn_censor` | pass-1 HMM did not call this window CN=1 |
| `gc_corr_tau` | `apply_gc_correction` | relative sd of the GC curve at this window's GC |
| `*_pass1` | `stage_pass1` | the first pass's value, before pass 2 overwrites it |
| `otr_gc_corr_rdcnt_cov` | `run_HMM` | back-converted to integer read counts (output/plots only — **not** the HMM's observation) |
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

`get_CNV.main`'s `correct_one()` implements the four modes by *renaming data into the column the
next stage expects*, rather than by branching inside the correction functions. It is called once per
pass, so a mode's aliasing is defined in exactly one place for both. `fit_otr_bias` unconditionally
reads `gc_corr_norm_cov`. The aliasing still drives the corrected-coverage columns written to
`CNV.csv` and the plots — but **not** the HMM, which now takes `bias` as an argument and composes its
emission offset from the correction-factor columns directly.

- `all` — no aliasing; both corrections run in sequence, and both are refitted in pass 2.
- `gc` — copy `gc_corr_norm_cov` → `otr_gc_corr_norm_cov`, skipping OTR entirely. The pooled GC
  refit still runs in pass 2, since the CN censor sharpens a GC fit whether or not an OTR stage
  exists.
- `otr` — copy `norm_raw_cov` → `gc_corr_norm_cov` so `fit_otr_bias` sees uncorrected input. No GC
  refit, because the GC correction was explicitly opted out of.
- `none` — copy `norm_raw_cov` → `otr_gc_corr_norm_cov`. No second pass at all.

Any new bias mode is another aliasing branch here, not a new parameter threaded through `core.py` —
*plus* an entry in `BIAS_OFFSET_COLUMNS`, since the HMM composes its emission offset from the
correction-factor columns rather than reading the aliased coverage column.

Note that GC correction has *already run* by the time this dispatch executes — `process_multi_genome`
does it unconditionally, and the `otr`/`none` branches discard the result by overwriting the column.
That is also why `is_deletion` / `is_redundant` are present on the frame in all four modes:
`mask_coverage_windows` runs as part of that unconditional GC stage.

### The pipeline runs TWICE, and the first HMM censors the second run

Every fit in CNery is contaminated by real copy-number variation.
`mask_coverage_windows` censors on two crude proxies computed once on
*uncorrected* coverage — `is_deletion` (≤10% of the global median) and
`is_redundant` (any repeat overlap) — and an **amplification is caught by
neither**, so it goes into the GC LOWESS and the OTR tent at full weight. The HMM
knows where the copy-number variation is; nothing else in the pipeline does.

```
PASS 1   censoring: is_deletion | is_redundant
  process_multi_genome:  preprocess -> pool -> mask -> pooled GC fit g1 -> apply
  per sequence:          GC skew -> otr_fit (gate + arbitration) -> tent T1 -> apply
  per sequence:          run_HMM(write=False)                              -> CN1

  is_cn_variant := (CN1 != 1)          <- add_cn_censor(), folded into exclude_from_fit

PASS 2   censoring: is_deletion | is_redundant | is_cn_variant
  pooled GC refit g2 on raw/(g1*T1)    ->  G = g1*g2,  gc_corr_norm_cov = raw/G
  per sequence: full otr_fit on raw/G  ->  T2          (the gate RE-RUNS)
  per sequence: run_HMM(write=True)    ->  the published calls, plots
```

This is alternating conditional fitting: `g2` is fitted on the residual with the
current tent removed, `T2` on the residual with the current GC curve removed.

- **The second OTR fit reads `raw/G`, never the residual `raw/(g1·T1·g2)`.** A
  tent has to be fitted to a series that still *contains* the ramp; fitting the
  residual would return a near-flat tent and the correction would collapse.
  `refit_gc_bias_pooled` therefore writes `gc_corr_norm_cov = raw/G` — the column
  and its own factor `gc_corr_fact = G` finally agree, which for one release they
  did not. The pass's own before/after, `raw/(g1·T1)` → `raw/(g1·T1·g2)`, is kept
  separately as `gc2_resid_cov` because the figure draws it and nothing else
  should consume it. `test_the_next_otr_fit_reads_raw_over_the_total_curve` pins
  the distinction.
- **`raw/G` is deliberately NOT flat against GC**, and reading its GC span as a
  regression is a mistake: `G` includes `g2`, which was fitted to flatten the
  *tent-corrected* series, so `raw/G` carries exactly the GC trend that the tent's
  own GC-projection implies (12.11% on `p5_75k_exp`). Dividing by `T2` removes it,
  and the FINAL column is what should be judged.

Span of a LOWESS of coverage against GC over the 1st–99th GC percentile, at each
stage — note `raw/G` is an intermediate, not a result:

| sequence | raw | `raw/g1` | `raw/(g1·T1)` | `raw/G` | **final** |
| --- | --- | --- | --- | --- | --- |
| `p5_75k_exp` | 18.27% | 1.20% | 10.27% | 12.11% | **0.93%** |
| `p1_shift` | 70.30% | 0.56% | 2.67% | 3.34% | **0.46%** |
| `adp1` | 12.85% | 1.03% | 1.74% | 2.44% | **0.81%** |
| `cwbi:chromosome` | 12.23% | 1.99% | 2.25% | 2.64% | **1.74%** |

- **Composition is exact and `gc_corr_fact` is the TOTAL.** Both passes are
  functions of GC alone, so `G = g1·g2` is a single curve. `gc_corr_fact` holds
  `G`, which is what `bias_offsets` feeds the HMM; the components stay on the
  frame as `gc_corr_fact_pass1` / `_pass2`, and all three are drawn in
  `GC_bias/*_GC_passes.pdf`. The two **oppose** each other at the extremes, so the
  second pass is mostly *removing* correction the first over-applied to the
  replication ramp.
- **The GC refit is pooled and applied to EVERY sequence**, including ones where
  no ramp was found — GC bias belongs to the sequencing chemistry, so one curve
  should describe the run rather than a different correction reaching each
  reference depending on whether its own OTR fit cleared a gate.
- **An undetected ramp again means bit-identical coverage.** Because
  `gc_corr_norm_cov` is now `raw/G`, the GC correction is fully accounted for
  before the OTR stage runs, so "no tent detected" means the OTR stage contributed
  exactly nothing. `test_no_tent_is_applied_when_no_bias_is_found` asserts that
  directly again, instead of having to divide `g2` back out first.
- **The pooled refit is what forces the pass structure.** It cannot run until
  every sequence is OTR-corrected *and* called, and the published HMM must not see
  coverage that is about to change.
  `tests/test_authentic.py::_run_pipeline` mirrors that split — that harness
  silently diverging from `main()` has cost real debugging time twice.

#### What the censor is worth, and what it costs

- **It changes no copy-number call on the corpus** — 0 windows on all 8
  sequences. What moves is the *evidence*: censoring the amplification takes
  `adp1` from r² 0.051 (p = 0.32) to r² 0.240 (p = 0.001) and `p5_75k_exp` from
  r² 0.473 to 0.943. The coverage arm now stands on its own where it previously
  needed the GC skew to carry it, and `adp1` acquires a live likelihood-ratio test
  for the first time (p = 0.167, still deferring to the skew).
- **Amplitudes barely move**: `adp1` 1.0673 → 1.0725, `cwbi:chromosome`
  1.1690 → 1.1699, `p5_75k_exp` 2.0500 → 2.0791, `p1_shift` 1.3464 → 1.3400. The
  earlier expectation that CWBI's ratio would fall toward 1.075 was wrong, and the
  reason is worth knowing: its breakpoints come from the GC skew, so only the OLS
  anchors could move, and the CN-34 amplification was never what those anchors
  were fitting.
- **`CN_CENSOR_MIN_KEEP = 0.5`.** If the censor would leave under half a
  sequence's windows it is declined and pass 2 censors exactly as pass 1 did — a
  genuinely duplicated replicon is every window at CN=2, and censoring it entirely
  is an empty fit, not a clean one. The floor is checked on what SURVIVES, so
  deletions and repeats already excluded count against it. CWBI's `plasmid_1`
  trips it on the corpus.
- **The gate re-runs in pass 2 and its p-values are the reported ones.** Pass 1's
  are kept beside them under `... (pass 1)` keys, so a verdict that changed under
  the censor is legible from the file alone.

#### The GC correction is an ESTIMATE, and the HMM is told how good a one

`bias_offsets` hands `run_HMM` a LOWESS fit evaluated at each window's GC.
Treating that fit as exact understates the emission variance, and understates it
**more at high copy number** — the offset's error is multiplied by `k`. Writing
`o = ô(1 + ε)` with `Var(ε) = τ²`, the law of total variance gives

```
Var(y | k) = m + m² · (1/(k·size) + τ²)          m = k · mu · ô
```

so the uncertainty adds in the reciprocal-size scale and the extra term `m²τ²`
grows as **k²**. The emission row for state `k` therefore uses
`k·size/(1 + k·size·τ²)`.

- **The k² scaling falls out; it is not imposed.** That is the whole reason this
  is a derivation rather than a knob, and it is why the correction is negligible
  at CN 1 and largest exactly where the offset is multiplied up.
  `test_the_added_variance_grows_as_k_squared` pins the exact form —
  `Var_with/Var_without − 1 = k·mu·τ²/(1 + mu/size)`, strictly linear in `k`.
- **The effective size must be formed PER STATE**, `k·r/(1 + k·r·τ²)`, not
  `k·(r/(1 + r·τ²))`. The second is the natural-looking refactor and it silently
  flattens the k² property to a constant.
- **τ is measured, never chosen.** `_gc_curve_tau` resamples the fit windows,
  refits the LOWESS on a 200-point GC grid `GC_TAU_SURROGATES = 100` times, and
  takes the sd of `log(curve)` — a relative sd, which is the form a
  multiplicative offset needs. Seeded, so goldens are stable.
- **Why no percentile or window-count rule works.** LOWESS uses a
  nearest-neighbour bandwidth (`frac=0.3`), so every fitted point averages the
  same ~0.3n windows — at GC 0.62 on `ltee_ara_m3_32k_2rg` there are still
  ~12,700 in the neighbourhood. The tails are not sparse; they are **one-sided**,
  so the local linear fit extrapolates within its own window. That is a boundary
  effect, invisible to any count-based rule, and resampling is what sees it.
  Measured τ is U-shaped: interior 0.0030, rising to 0.0095 at the 99.5th GC
  percentile and 0.0225 at the extreme.

**What it does, measured at the CLI defaults.** Exactly one thing:
`ltee_ara_m3_32k_2rg` goes 31 → 29 segments, dropping an 11-window **CN-4 sliver**
that sat inside a 263-window CN-3 amplification. Those windows are at the 99.5th
GC percentile, where the offset is divided down to 0.82 and the corrected level
reads 4.2 — a GC excursion wearing a copy-number costume. **Zero windows move on
the other seven sequences**, and no golden changed, because at the goldens'
`-w 1000 -s 500` the correction is a measured no-op. That is why
`tests/test_authentic.py::TestOffsetUncertaintyAtCliDefaults` runs its own two
datasets at `-w 100 -s 100 -f 400`; without it the whole mechanism could be
deleted with the suite still green.

- **`adp1_mgd06_lb` is the negative control and is not optional.** Its CN-3
  amplification contains 4 windows *below* the 0.25th GC percentile — the
  opposite excursion. Discounting evidence where the curve is uncertain is
  sign-blind, so it must leave that block whole, and the test asserts it does.
- **The fix has a working band, not unlimited headroom.** It holds from 1× to 2×
  the measured τ, but a **uniform** 4× inflation brings the sliver back at five
  times the size (55 windows) and loses the CN-12 call, because it also inflates
  the interior where the fit is sound. The *shape* of τ matters, not its scale.
  Do not "strengthen" this by scaling it.

**Two alternatives were measured and rejected. Do not retry them.**

- **Clamping the GC correction at the tails.** It changes the point estimate, so
  it necessarily helps one sign of excursion and hurts the other: at 0.5% tails
  it merges `m3_32k` (31 → 29) and *splits* `adp1` (7 → 9), a wash. Its threshold
  is also non-monotonic — at 1% and above the sliver returns **worse** (22 windows
  against 11 unclamped), because the clamp is a global refit that shifts `mu`,
  the state count and `g2`.
- **A fixed-width GC kernel.** Its point estimates agree with the rank-based one
  to ≤2% over the 1st–99th percentile and across a 12× bandwidth range, and both
  fix the artifact. It would cost a hand-rolled smoother, sparse-GC guards and a
  full golden regeneration for no measured gain in what is actually applied. It
  does report tail uncertainty better (τ 4× larger, 5–6× tail/interior contrast
  against 3×), which is the one real argument for it.

**`frac=0.3` is CV-validated, not an unexamined constant.** 5-fold
cross-validation cannot separate it from any bandwidth in 0.05–0.50 (within
0.03% MSE on both `m3_32k` and `adp1`); only 0.02 is clearly worse, at 49%. It is
also a sensible density-adaptive rule in disguise — measured at **0.26–0.44 × the
GC standard deviation** across six references, median 0.32. Its one weakness is
the tails, where the 30% neighbourhood spans 2.4–4.0 × the GC sd.

#### Why re-running the gate on censored data is safe HERE

CLAUDE.md used to record that an iterative fit–censor–refit loop took the OTR
false-positive rate from 0/8 to 4/8, because excising 1–2% of windows removes
87–98% of the variance the bootstrap was calibrated against. That measurement was
on a **residual-driven** censor: it excised the interval where the tent under test
fit worst, so the selection served the hypothesis. The HMM censor does not look at
the tent's residual at all.

Measured, on synthetic coverage built on `adp1`'s real geometry and GC with the
truth known — flat AR(1) trend at the measured 83-window scale, negative-binomial
counts, 60 seeds per arm, coverage-fit gate at a nominal 1%:

| arm | pass 1 clears | pass 2 clears |
| --- | --- | --- |
| flat, no amplification | 3% | **3%** |
| flat + CN-3 amplification | 0% | **3%** |
| 1.5× ramp + CN-3 amplification | **0%** | **100%** |

Two things to read off that. The censor does **not** inflate the false-positive
rate: it returns the gate to its own baseline (3% at a nominal 1% — the gate is
mildly anti-conservative on this synthetic, and was already), which the
amplification's variance had been artificially suppressing to 0%. And the last row
is the whole point: a real 1.5× ramp with an amplification on top is detected
**0/60 times** without the censor and **60/60 times** with it.

- **The variance-collapse effect is real and is not the failure it looks like.**
  Censoring 2.71% of `adp1`'s windows takes the weighted SST of the decimated
  series from 433.9 to 72.2 and its tent r² from 0.016 to 0.463. The bootstrap
  null resamples the same censored series, so it carries the same SST; what
  changed is that the amplification is no longer a block no tent can fit,
  dominating both numerator and denominator.
- **Do not build a ramp-free control by dividing out the best-fit tent.** It was
  tried and it is invalid: that tent is fitted on *contaminated* data — on `adp1`
  it sits inside the CN-3 amplification — so what is left is (real ramp − wrong
  tent), which still contains a ramp. Censoring then lets the fit find it, and the
  control reads 4/8 "false positives" that are nothing of the kind. The measured
  giveaway was a residual with τ = 1 cell over a series with τ = 223. Build the
  null synthetically, where the truth is known.

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
- The HMM stacks a geometric zero-state row on top of one negative-binomial emission row per copy
  number, so **state index == copy number** and the matrices are `n_states + 1` square/rows. The
  negative binomial (not Poisson) is intentional: coverage is overdispersed.
- **The zero state's mean is a fraction of the local baseline**, `deletion_coverage_fraction * mu *
  offset` (`-z`, default 0.02), which makes it the `k = 0.02` case of the emission contract below
  rather than a special case. It used to be `geom.pmf(count + 1, 1 - error_rate)` — a geometric of
  mean `error_rate / (1 - error_rate)` = 0.176 counts **absolute**, tied to nothing about the sample.
  As a fraction of baseline that is 0.28% at 60x and 0.018% at 1000x, so the largest residual coverage
  still called CN0 drifted from 19% to 4%: the same deletion was called on a shallow run and missed on
  a deep one. `tests/test_hmm.py::test_deletion_calls_do_not_depend_on_sequencing_depth` pins it.
  The default 0.02 is measured, not assumed — REL606's called deletions hold a mean 2.2% of baseline
  (90th percentile 3.2%), the residue of mismapping and repeat spill.
- **The HMM observes RAW counts, and the bias corrections enter as offsets on the emission mean** —
  `E[count | CN = k] = k * mu * gc_corr_fact * otr_gc_corr_fact`, not `corrected_coverage`. Both
  factor columns are already the divisors, so the same numbers multiply the expectation. Dividing
  them out of the data instead scales each window's variance by `1/factor²` while a single global
  variance is applied to all of them, and rounds after dividing. This is why the HMM reads
  `read_count_cov` and the factor columns rather than `otr_gc_corr_norm_cov`, which stays in the
  frame for the CSV and the plots. `bias_offsets()` composes the factors per `--bias` mode, and
  `bias` must be **passed in** to `run_HMM` — `gc_corr_fact` is on the frame even under `--bias none`.
- `fit_censored_negative_binomial` ports breseq's censoring (`coverage_distribution.cpp`): mode of a
  5-point moving average searched upward from `mean/4`, then `[0.5, 1.5] × mode`. Without it,
  `np.var` over every window measures the spread of a *mixture* — an amplification inflates the very
  dispersion meant to detect it (REL606: var/mean 6.03 against a true var/mu of 2.48). The objective
  is the truncated likelihood, not breseq's least-squares, so the fitted `size` is **not** comparable
  to breseq's `nbinom_size_parameter`. It returns `None` on degenerate input; `run_HMM` then falls
  back to moments plus the `var = mean * (1 + 1e-3)` guard.
- **No window with any redundant coverage enters that fit, unconditionally.** This is deliberately
  *not* softened by `min_called_windows` the way the observation mask is: a window clipping the edge
  of an IS element sits at 1.2–1.4x, *inside* the censoring band, where the bounds cannot catch it.
- `changeprob` is a per-*base* rate (`--change-rate`, `DEFAULT_CHANGE_RATE = 1e-6`), converted to a
  per-window probability as `1 - exp(-rate * step)`. A flat per-window probability implies a per-base
  rate of `changeprob/step`, so re-tiling the same genome silently restated the biology. Every
  observed transition is charged one `step` regardless of the gap censored windows left — pricing a
  wide repeat gap as a cheaper crossing would make a censored repeat a cheap place to break a segment.
- Log emissions are tempered by `step/window` (`overlap_weighting`). Under `-w 200 -s 100` every
  base sits in two windows, so the likelihood would otherwise count each base twice while the
  transition prior counts it once. It is a no-op at the default `-w 100 -s 100`, where
  `step == window`. Note this does **not** buy full `-w` invariance: the window statistic is a
  per-base *median*, whose precision grows sublinearly with window width (fitted `size` 43 at
  `w=100` against 54 at `w=200`), so `-w` remains a resolution knob. That is why the default is
  `-w 100 -s 100`: at `-w 200` the weakest two-window events stop being callable.
- `robust_state_count` sizes the state space from a 3-window rolling median, not `int(max(coverage))`
  — with a flat off-diagonal the switch cost carries `-log(n_states)`, so one outlier window would
  otherwise make every duplication call dearer genome-wide.
- The decode is a **real Viterbi backtrace**. `make_viterbi_mat` returns forward scores only; the
  path comes from `viterbi_path`. Taking `np.argmax(logv, axis=1)` per window is *not* a path — it
  can name a state no single path passes through, which is how a 3-window amplification came out
  labelled `1,1,3`, on its lowest window. `log_transition` is indexed **`[from, to]`**.
- `OTR_corr/<sample><seq_id>_otr_results.json` carries **`"Relative copy number"`**: this sequence's
  coverage relative to the longest sequence in the run, which reads exactly 1.0. Deliberately
  non-integral — 2.95 copies is a measurement. Computed by `relative_copy_numbers()` from the censored
  median of `gc_corr_norm_cov`, and passed *into* `apply_otr_correction` rather than carried on the
  frame: it is one scalar per sequence, and a constant column would add 226–407 kB to an 8.3 MB
  `CNV.csv`.
- `"Origin-to-Termius/Bias Ratio"` is plain `yori / yter`, so the file's own three numbers agree. It
  used to be `yori / (yter + 0.001)` — a divide-by-zero guard that put the reported ratio ~0.1% below
  what the two coverage values printed beside it give (1.06627 against 1.06733 on `adp1_mgd06_lb`),
  so a reader checking the arithmetic found it wrong. `yter` is a least-squares anchor on a curve
  already clipped at `otr_floor = 0.1 x` the median coverage, so it cannot be zero; the explicit
  `yter > 0` test remains and emits `null` rather than a fabricated number.
  `TestOriginTerminus::test_reported_ratio_reproduces_from_its_own_two_values` pins it.
- **That JSON must stay strict JSON.** breseq parses it with nlohmann, which has no `allow_nan`, so a
  single bare `NaN` makes the whole file unparseable and silently costs it all OTR reporting.
  `_json_safe` maps non-finite values to `null` on the way out. Two further constraints from the same
  reader: `"Origin-to-Termius/Bias Ratio"` is load-bearing **including the typo** (renaming it makes
  breseq's `j.count()` fail), and `"Origin window"` / `"Terminus window"` are not type-checked there,
  so they must never be null. Adding keys is safe; `_break_pts.csv` is not — breseq asserts exactly
  three columns and the assert is fatal.
- `corr_plots/<sample><seq_id>_correction_stages.pdf` is the per-sequence **before/after** diagnostic:
  one row per fitting *change* — GC and OTR — each with its own censoring strip beneath it. It is emitted in **all four `--bias` modes**, and **self-creates its directory** rather
  than being added to `out_subdirs`, which is what lets the self-creation test assert something
  instead of passing vacuously. Three things about it are load-bearing:
  - **The censoring strip is drawn per row even though the rows agree today.** Both fits exclude
    the same `is_deletion | is_redundant` set, so the two strips come out identical — which is the
    useful reading, not a redundancy: those fits saw the same data. A step that ever sees a
    different set then shows it where it applies rather than in one detached track.
    (`is_deletion`/`is_redundant` are computed once in `mask_coverage_windows` on *uncorrected*
    coverage and never revisited, so "≤10% of the global median" means something different near
    the origin than near the terminus; the strips make that frozen-ness legible rather than
    implicit.)
  - **The in-band metric is divided by `censored_median_coverage` first.** Without it the measure
    reads **0.000 for every stage and both denominators** on both CWBI plasmids — `norm_raw_cov` is
    normalised against the *pooled* median while they sit at 2.95× and 1.90× — and three empty bars
    would read as "the pipeline did nothing". The scaling is a no-op elsewhere (0.981–1.001 on all
    five chromosomes).
  - **Both denominators are reported because the LEVEL differs**, not the delta. On `plasmid_1` the
    uncensored figure reads a flat **1.000** against an honest **0.819**, since 52% of its windows
    are repeats sitting at 0.6–0.75. The censoring strip is what explains the gap.
- **The censoring strip needs two renderings.** Deletions are 1–15 runs of 12–55 windows and draw as
  true spans; repeats are 79–**332** runs of **2–4 windows**, which at 9,258 windows across ~10
  inches is ~0.001 inch each — spans there render as invisible stippling that reads as *nothing
  censored*, so they get a binned density lane at `min(400, n)` bins. Do not fold deletions into that
  density: a 12-window deletion in a 23-window bin reads 0.5 and understates it.
- **`OTR_STAGE_ROWS` is a table rather than "whatever columns are present"** because `--bias otr`
  aliases `gc_corr_norm_cov = norm_raw_cov` (`get_CNV.py`), so a GC row there would plot a
  bit-identical copy of the raw series under a label that is false.
- Output subdirectories (`CNV_plt/`, `CNV_csv/`, `GC_bias/`, `OTR_corr/`, `GC_skew/`) are created
  once in `get_CNV.py`, *after* inputs are resolved so a bad invocation creates nothing; the writer
  functions in `core.py` assume they already exist. `write_gc_skew_results` and `plot_gc_skew` are
  the exceptions, making their own like `apply_otr_correction` does.
- `predict_ori_ter_from_skew` (`core.py`) **does not censor** `is_deletion` / `is_redundant`
  windows, unlike every other fit stage. GC skew is a property of the *reference sequence*, and a
  deletion in the sample does not change the reference's base composition. Because `cum_gc_skew`
  is a running sum, dropping a window would not merely omit it — it would displace every point
  after it.
- `cum_gc_skew` subtracts the mean skew before cumulating (`preprocess`). That is **not cosmetic
  detrending**: it forces the sum to end at exactly zero, which is what makes `argmin`/`argmax`
  independent of where the reference's coordinate 1 falls. Rotating the start by `r` windows then
  maps the curve to `C'(k) = C((r+k) mod n) - C(r)`, a constant offset. Without it the sum ends at
  `n * mean(skew) != 0`, the wraparound adds a linear ramp, and a circularly permuted copy of the
  same genome predicts a different origin — which `TestGCSkewOriginTerminus` in
  `tests/test_authentic.py` checks against the `REL606_2314906bp_shift` dataset.
- Overlapping windows (`step < win`) count each base `win/step` times over in `cum_gc_skew`. That
  is one uniform factor across every window, so it rescales the curve without moving either
  extremum; no stride weighting is applied.
- The skew estimate's confidence gate is **two conditions**: ori/ter 35–65% apart (the same band
  `otr_fit` uses), and a circular block bootstrap p-value at or below 0.01. There is deliberately
  **no minimum window count** — the bootstrap subsumes it, since a short sequence cannot reach
  significance on its own, and one arbitrary constant beats two. `Sequence.skew_confident` in
  `tests/test_authentic.py` records which sequences pass.
- **The t-statistic is an effect size, not a test statistic.** Skew is spatially autocorrelated
  (REL606's ACF is still 0.22 at lag 10), so t's magnitude is inflated by an unknown factor and
  `t = 34.8` emphatically does not mean `p ≈ 1e-250`. `_replichore_t` deflates the effective sample
  size by the `win/step` overlap factor, which stops t growing as `sqrt(win/step)` from nothing but
  a finer stride — but that fixes double-counting only, not the autocorrelation. The p-value is
  what carries inferential weight; t is retained because it is the readable effect size.
- `_skew_bootstrap_p` re-runs the **whole** procedure — extrema search included — on every
  surrogate. The breakpoints are chosen by looking at the data, so a null holding them fixed would
  ignore that selection and be far too easy to beat.
- **What governs the bootstrap's power is the NUMBER of blocks, not their length.** Measured on a
  synthetic switch: 24 blocks puts p at its floor, 12 gives 0.004, 6 gives 0.035 and 3 gives 0.041,
  whatever the block length. Hence `_skew_block_length` targets `SKEW_TARGET_BLOCKS = 20` and
  adapts to `n`, rather than taking a fixed length. Bounded to 10–200 windows: at least the local
  autocorrelation length, and past 200 nothing is gained while blocks are lost.
- **p is floored at `1/(B+1)` and is an upper bound, not a measurement.** Every real chromosome
  exhausts all 1000 surrogates and reads back exactly 0.001. `"Bootstrap surrogates"` is in the
  JSON so that floor is legible from the file alone. Resolving further means `B ≈ 1e6` at ~90 s per
  sequence, to say something already obvious.
- The gate sets a flag; it never suppresses a coordinate. `GC_skew/*_gc_skew_results.json` always
  carries the measured origin, terminus, separation, amplitude, t and p, so a rejected prediction
  can be diagnosed from the file and the plot. Contrast the OTR JSON, which discards its
  coordinates behind `"Not detected"`. Like that file it is written with `allow_nan=False` and
  stays strict RFC JSON — every value in it is finite by construction, and the assertion is
  deliberate.
- The bootstrap is seeded (`seed=0`) so a given input always gives the same p and the goldens stay
  stable. It costs ~0.15 s on a 4.6 Mb genome against ~4 s for `preprocess`, so roughly 3%.

### Deciding whether the OTR tent is real, and whose breakpoints to use

- **Everything here rests on one identity.** `_otr_design_matrix`'s rows sum to exactly 1, so for
  fixed breakpoints the tent's column span is `span{1, u}` for a single phase vector `u` (0 at the
  origin, 1 at the terminus, back to 0). Hence `1 - RSS/SST == r²(u, y)` — a dot product, not a
  least-squares solve, which is what makes scoring a whole breakpoint grid across 1000 surrogates
  affordable. It is the *same* objective the Nelder–Mead search minimizes, not an approximation.
  Do not "simplify" `_otr_grid_scores` back into a loop over `minimize`.
- **The decimated series carries a WEIGHT per cell, and empty cells weigh nothing.** `_otr_decimate`
  returns `(values, weights)`, the weight being how many unmasked windows fell in that cell. Cells
  with none used to be filled by circular `np.interp` — fabricating coverage that supports whatever
  trend the neighbours imply, and not rarely: on CWBI's `plasmid_1`, 121 of 232 cells held no
  unmasked window, so **52% of the scored series was invented** and the statistic read r² 0.175
  (p = 0.034) against 0.120 (p = 0.099) once the fabrication was removed. The weighting also makes a
  cell holding three windows count three times one holding one, which is what the full-resolution
  objective does; equal-weighting cells silently reweighted the genome wherever censoring was uneven.
  The cell count is capped at the number of unmasked windows, since asking for more cells than
  observations only manufactures empty ones. In the bootstrap the **weights stay fixed at their
  lattice positions while values are resampled** — where the repeats and deletions sit is a property
  of the reference, and the null is "same censoring geometry, trend destroyed", not "the censoring
  happened elsewhere".
- **The free search is constrained to the band, and that constraint is load-bearing.** It is
  parametrised as `(x_ori, separation)` with `separation` a box bound, seeded from the grid's argmax
  and refined by Nelder–Mead. Unconstrained, this objective's global optimum is usually **not a tent
  at all**: as separation → 0 the short arc vanishes and the regressor tends to
  `1 - ((x - x_ori) mod L)/L`, a circular **sawtooth** — a straight line across the genome with one
  free discontinuity. Same two parameters, strictly larger shape class (any monotone drift *plus* one
  step), so RSS is monotone non-increasing as separation shrinks unless the coverage really is
  V-shaped. Measured: fitting the pure sawtooth family alone reproduces the unconstrained optimum
  (adp1 r² 0.1130 vs 0.1138, break at window 4820 vs 4821.2), and on adp1 that break sits on a real
  2.7× amplification edge — it was absorbing a copy-number step, not a replication ramp. Over 40 flat
  AR(1) nulls the unconstrained optimum landed below 5% separation 20 times and in-band 6.
- **This was never an optimiser weakness.** Brute force (500×500 grid + local refinement + 600 random
  restarts) confirms multi-start Nelder–Mead finds the exact global optimum on 6 of 8 authentic
  sequences and misses by ≤0.16% on the other two. Do not "fix" the search; the constraint is what
  matters. A fit pinned *at* the 0.35 bound is the honest signal that there is no interior optimum.
- Two seeding details that cost real work before: seed `k` and seed `k+4` used to be
  `(s, s+L/2)` and `(s+L/2, s)` — the same **unordered** pair, and the objective is symmetric under
  swap, so four of nine starts returned bit-identical results. And scipy perturbs an exactly-zero
  coordinate by an absolute `0.00025` instead of a relative 5%, which froze `x_ori` to a total
  excursion of 0.002 windows on two sequences — a 1-D search in disguise. Seeds are now offset by
  half a step so no coordinate is ever 0. The old masked argmax/argmin seed is gone: it converged to
  a strictly worse minimum on five of eight sequences and never uniquely won.
- The old gate was **separation plus a vacuous `bias_threshold`**: the label swap guarantees
  `y_ori >= y_ter`, so `ratio > 1.0` reduced to `y_ter > 0` and nothing ever tested the tent against
  a null. It is now separation **and** a circular block bootstrap at p ≤ 0.01, on a series decimated
  to ≤ `OTR_SCORE_CELLS` cells so cost is independent of genome size and of `-w`/`-s`.
- **Block length here governs validity, not just power** — the opposite of the GC-skew gate, where
  `test_verdict_is_insensitive_to_block_length` pins insensitivity as a virtue. Measured
  false-positive rate at a nominal 1%: 0.00 at block ≈ 5τ, but 0.10–0.33 at block ≈ τ. Hence the
  floor at `5 * tau` alongside the block-count target. **τ must be measured on the residual of the
  best tent**, never the raw series: the ramp is itself long-range structure, and measuring it raw
  pushed `p5_75k_exp` from p = 0.001 to 0.034 — the test destroying its own detection.
- **One bootstrap function covers both arms, and the shape of `phases` is the whole difference.**
  `R > 1` is the free fit, whose breakpoints were data-chosen, so the null re-runs that selection on
  every surrogate. `R == 1` is the GC-skew fit, whose breakpoints came from `ref_base` — the coverage
  resampling never touches it, so no selection occurred and taking a max would be a *conservative
  error*, not a safe default. That is why the skew arm is the more powerful of the two: on CWBI's
  chromosome it scores a smaller statistic (0.0152 vs 0.0255) and returns a smaller p (0.213 vs 0.278).
- **An orientation contradiction rejects the skew candidate; it never relabels it.** `otr_fit` swaps
  its own breakpoints when they come back inverted, which is legitimate because they are unlabelled
  and the antipodal seeds are blind. The skew's labels *are* the imported prior. Measured, 1 of 8
  sequences contradicts (`ltee_ara_m3_32k_2rg`, ratio 0.907) and its skew tent explains r² = 0.007,
  so what is rejected there is the sign of noise — on flat synthetic coverage the sign is a coin
  flip (49.3% over 300 trials).
- **When both arms are live, a bootstrap likelihood ratio decides.** The models are nested — both fit
  the anchors by OLS, the skew model additionally fixes the two breakpoint positions — so
  `Λ = m·ln(RSS_skew/RSS_free)` with 2 degrees of freedom. `RSS_free` is the minimum over the *same
  band-restricted grid*, never a fresh Nelder–Mead fit: fed the unrestricted optimum, adp1's
  degenerate 1.1%-separation spike (r² 0.1175 against a band-restricted 0.0524) wins at p = 0.001
  instead of losing at 0.181. Run the LRT only *after* both arms have been gated, never in place of
  the gates.
- **Λ's null is not χ²(2), and the reason is autocorrelation alone.** With *iid* residuals the null
  of Λ is χ²(2) to Monte-Carlo error on all eight sequences (q99 8.3–9.3 against 9.21), and stays so
  even with the separation band removed — neighbouring tents are near-collinear, so the max over
  change-point breakpoints costs essentially nothing. What breaks it is that real residuals stay
  spatially autocorrelated, inflating Λ's 99th percentile by 2.8×–128× (24×–51× on the *E. coli*
  chromosomes). χ²(2) would report p < 0.001 for **every** sequence, including `p5_75k_exp` where the
  two tents agree to 1.8% of the genome and the skew lands on REL606's *oriC* — bootstrap p is 0.80.
  `test_chi_square_would_reject_where_the_bootstrap_does_not` pins this.
- **The LRT resamples residuals; the detection gate resamples the raw series.** They disagree on
  purpose. A bootstrap for a composite null must simulate *from* the null model, and the LRT's null
  ("the skew tent is the truth") is fully specified. The detection gate's null has no fitted signal
  to hold fixed, so residual resampling there would bake the alternative into it.
- `OTR_LR_ALPHA = 0.05`, deliberately **not** the 0.01 the two detection gates share: those decide
  whether to correct at all, this only picks which of two credible ramps supplies the breakpoints.
  At 0.01 the synthetic true-null rejection rate is 0/30 and power halves (33% vs 70% at a
  5%-of-genome displacement). On the corpus every verdict is identical for any α in 0.01–0.10.
- **`OTR_SKEW_MAX_P = None` means the skew arm needs no coverage evidence of its own**, and the cost
  is explicit: on `ltee_ara_m3_38k` and `adp1_mgd06_lb` the applied ramps (1.069, 1.067) sit inside
  the flat-noise band and their own fixed-breakpoint p-values are 0.243 and 0.615. Those corrections
  are applied because the reference says an origin is there and the coverage does not contradict it.
  What bounds the damage is that the skew supplies only *where* — the amplitude is still solved by
  OLS against the observed coverage, so a flat sample yields a flat tent (300 synthetic flat series:
  ratio median 0.999, 95th pct 1.051, max 1.089). Setting it to 0.01 instead would rescue **zero**
  sequences; the fixed-breakpoint p-values run 0.001, 0.001, 0.213, 0.243, 0.328, 0.343, 0.615, 0.841
  with a clean gap, so nothing between 0.01 and 0.2 behaves differently.
- **`"Residual structure score"` is reported, never gating.** r² says how much of the variance the
  tent explained; this says whether what it *failed* to explain is structured — a tent can score a
  respectable r² and still be systematically wrong over a long stretch. It is the weighted circular
  CUSUM range (Kuiper's V) of the applied tent's residual, expressed as a z against a circular
  block-bootstrap null. Read it as **< 1 unstructured, 1–2 mild, > 2 structured, > 3 strongly so**.
  Measured on the corpus: `p1_50k_shift` 2.62, and `m3_38k` 1.03, `adp1` 1.03, `p5_75k_exp` −0.27,
  `cwbi:chromosome` −0.78. The one that fires is the case the terminus-region bullet below already
  documents, and it fires nowhere else.
- **Why none of the obvious statistics is used for it.** Coverage residuals are nowhere near white
  even under a perfect fit — measured decorrelation lengths are 2–50 cells (2–45 kb) on *good* fits.
  So lag-1 ρ / Durbin–Watson, τ, Ljung–Box and a runs test all rank `adp1`'s merely *weak* fit above
  `p1`'s genuinely *biased* one (ρ₁ 0.897 vs 0.736; Q 41,821 vs 24,671), and the runs test even gets
  the sign of the difference backwards. Bootstrap-normalising them is *identically* uninformative
  (z ≈ 0.6) because block resampling preserves exactly the short-range correlation they measure —
  **τ is the floor, not the signal**, which is why it is published beside the score rather than as
  it. The variance ratio V(b) reads z = 5–6 on the CWBI chromosome (r² = 0.005), and a
  fixed-bandwidth Bartlett correction reads 2.3–2.8 on the two plasmids and cannot separate a 2×
  amplification over 15% of the genome from a 10% terminus displacement.
- **The `_replichore_t` overlap-deflation precedent does NOT transfer here.** On the decimated
  lattice, cells are 719–1157 bp apart against 1000 bp windows, so adjacent cells share *zero* bases
  — the analytic `win/step` factor is exactly 1.00 on four of eight sequences and 1.20–1.32 on the
  rest, while the measured long-run-variance inflation of the same residuals is 2.5–46×. Overlap
  explains 0–13% of what it would have to. Only the block bootstrap reproduces the intrinsic floor.
- **The score is a detector, not a ruler.** Measured mean over 6 seeds at 0 / 2.5 / 5 / 10 / 20 / 30%
  breakpoint displacement: −0.02, 2.06, 3.67, 2.79, 2.67, 2.52. It climbs steeply, peaks near 5% and
  plateaus — because a worse fit leaves a longer-correlated residual (τ 7 → 171 cells across that
  range), which lengthens the bootstrap block, which raises the null with it. The same feedback is
  what keeps a real copy-number event from reading as misfit: a 2× amplification over 5% of the
  genome scores below a 5% displacement. The honest limit is that at 15% of the genome a 2×
  amplification does reach displacement-like values. The null itself is correctly centred (mean
  −0.02, sd 0.94), which is why the synthetic tests average over seeds rather than assert on one.
- The score's raw statistic grows as √m (4.21 → 9.84 as `OTR_SCORE_CELLS` goes 1000 → 8000) and is
  **unpublishable**; the calibrated z is flat to ±0.06 across the same range. And it must be
  **weighted**: unweighted, `plasmid_1` scores 3.37 — louder than the one real misfit — because 45%
  of its cells carry the constant fill value and a run of identical values is a maximal CUSUM
  excursion. Both plasmids fall below `OTR_MIN_CELLS` and report `null` instead, which is the
  honest answer for a series that is mostly fill.
- **Copy-number events contaminate the tent, and censoring them back out is harder than it looks.**
  Measured: `adp1_mgd06_lb`'s free-fit origin lands *inside* its own CN-3 amplification and
  `cwbi_ssym_ht04`'s chromosome inside its CN-34 one, and excising the amplification takes CWBI's
  applied ratio from 1.169 to 1.075 — roughly 40% of the ramp being divided out of that chromosome
  is copy number, not replication. The GC-skew arm can supply good *coordinates* in that situation;
  what it cannot fix is the *amplitude*, because the anchors are still solved on contaminated data.
  **Nothing in the pipeline corrects for this today**, and a fix must not be an iterative
  fit-censor-refit loop: one was prototyped and measured, and on ramp-free real sequences — each
  authentic sequence with its own best-fit tent divided out, so there is no ramp by construction —
  it took the OTR false-positive rate from **0/8 to 4/8** at a nominal 1%, inventing a ratio of
  **1.26** on `ltee_ara_m3_32k_2rg` from a starting p of 0.918. The mechanism: excising 1–2% of the
  *windows* removes **87–98% of the variance** the bootstrap was calibrated against, and a null
  whose surrogates come from the censored series cannot defend itself. A cap does not help — three
  of those four appeared after **one** round. Whatever replaces this, the detection decision must
  stay frozen on the uncensored series.
- **Inversions are deliberately not detected.** They are visible in coverage only *through* the
  replication ramp, which they mirror, and their mean residual is exactly zero — so the HMM calls
  CN = 1 straight through and only a slope-shaped statistic could see them. Detecting them was
  prototyped and dropped: the shape label was unreliable below the scan resolution, and on the
  authentic corpus it changed no result. Two structural blind spots would remain regardless, both
  physics rather than tuning: an
  **ori-symmetric** inversion produces a 3e-4 residual because replication timing is symmetric about
  the axis, and a **terminus-spanning** one is largely reparametrisable as a shifted terminus.
- **A single-kink tent cannot represent a terminus *region*.** Forks meet across the ter macrodomain,
  not at a point, so a symmetric V must choose between matching the descent and matching the trough.
  On `ltee_ara_p1_50k_shift` the observed trough is at 3.571 Mb while the coverage fit puts its
  terminus 221 kb early and the skew 270 kb late — the truth is between them. The LRT still picks the
  coverage fit there (p = 0.004), and 87% of that fit's advantage comes from *outside* the disputed
  stretch: the tent is a global two-line fit, so moving the terminus by 10.6% of the genome re-slopes
  both limbs everywhere. Note the skew is nonetheless the better *landmark* estimate on that sample
  (terminus 1.526 Mb against *dif* at ~1.55 Mb; the coverage fit is ~490 kb off). `otr_fit` optimises
  the coverage trend, not the annotation, and the skew's landmarks stay published in `GC_skew/`.
- The GC-skew coordinates are now consumed, so **`predict_ori_ter_from_skew` must run before
  `fit_otr_bias`** — it does in `get_CNV.main` and in `tests/test_authentic.py::_run_pipeline`.
  Computing it afterwards silently reverts to the coverage-only arm. `otr_fit` also refuses a
  prediction whose `"Windows"` disagrees with the frame, because the skew's indices are positional
  and `otr_fit` works in `df.index`; those agree only because `process_multi_genome` resets the index.

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
CNERY_TESTDATA_DIR=/big/disk conda run -p $PWD/env pytest   # relocate the ~152 MB cache
```

Six datasets, differing where it matters — 0/1/2 read groups, 5/10/18/26 columns, 4/7/10-line
footers, TSV and CSV, one and three sequences, and three different genome lengths. The footer
spread is real regression coverage for the prefix-based footer stripping that replaced
`skipfooter=4`.

Three of them carry the load for a specific area:

- `ltee_ara_p5_75k_exp` — **the only sample whose coverage carries a strong replication gradient**,
  an exponential-phase culture where it is real (1.95x peak to trough, terminus ~1.53 Mb, matching
  REL606's known ones). Since the GC-skew arm landed, four of the six chromosomes correct, but this
  is the only one where the *coverage fit* clears significance on its own strength — and even here
  the likelihood ratio cannot separate the two estimates (p = 0.80) and defers to the skew, whose
  origin lands on REL606's *oriC* at ~3.886 Mb where the coverage fit was 82 kb short. It is also the
  regression test for the ori/ter label swap: `_otr_concentrated_rss` is symmetric under exchanging
  the two breakpoints, so a mirrored fit divides by an inverted ramp and *spreads* coverage out —
  `TestOriginTerminus::test_correction_tightens_coverage` catches exactly that.
- `ltee_ara_p1_50k_shift` — **the only sequence where the coverage fit beats the GC skew**
  (likelihood ratio p = 0.004), and so the only live exercise of the arbitration's other branch.
  Also the case that shows the tent model's limit: neither estimate sits at the observed coverage
  trough. See the terminus-region bullet above.
- `cwbi_ssym_ht04` — **the only multi-sequence dataset and the only CSV one**. A chromosome plus
  two plasmids, so it is the only cover for `process_multi_genome`'s pooled GC fit and its shared
  global median. Note what it shows: the pooled median keeps the plasmids above the chromosome, but
  `run_HMM` refits the single-copy level from whichever sequence it is handed (fitted mu
  100.9 / 300.0 / 194.6), so **`prob_copy_number` is 1 for both plasmids** — that follows from "CN
  calling is per-reference". The multiple is not lost, though: it is published as
  **`"Relative copy number"`** in each sequence's `_otr_results.json`, at 2.953 and 1.898 against a
  chromosome pinned to exactly 1.0. Its chromosome is also the case that motivated the significance
  gate: the coverage fit there is a marginal 1.217 that fails at p = 0.28 and used to be applied
  anyway, moving windows within 20% of single-copy from 91.8% to 91.5%. It is now corrected on the
  skew's coordinates instead, at ratio 1.169, and finally helps (91.8% → 92.5%). Both plasmids are
  the negative control for the skew arm — neither clears the skew's own gate, which is the expected
  answer for a replicon with no bidirectional origin.
- `adp1_mgd06_lb` — the first non-REL606 genome, which is why sequence length and window count are
  per-`Sequence` rather than the module constants they used to be.

Tests that read a frame or an output file are parametrized **per sequence**, not per dataset.
Goldens are named `<dataset>_<seq_id>_break_pts.csv` for the same reason — CNery writes one file
per sequence.

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
