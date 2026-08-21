# Running the multi-seed paper grid on CENAPAD-SP "Lovelace" (OpenPBS)

Target: the normalised-AUC tables — one for accuracy, one for MCC — every entry
mean ± std over 5 seeds, with a compute-time column, for **2 architectures ×
2 datasets × 6 schemes**.

| | SimpleNN (2×64 MLP) | SimpleViT (d4/e128/h4/p4) |
|---|---|---|
| MNIST | `configs/nn_mnist.yaml` (55,050 par.) | `configs/vit_mnist.yaml` (539,914 par.) |
| CIFAR-10 | `configs/nn_cifar10.yaml` (201,482 par.) | `configs/vit_cifar10.yaml` (545,930 par.) |

All four use `sweep.step: 0.1`, `f_dist_avg_points: 3`, `fim_subset_size: 500`,
`warm_start: off`, all four `evaluation.metrics`, and `validation_split: 0`. One config serves all seeds — `FDIST_SEED` overrides
`experiment.seed`, and each seed gets its own checkpoint
(`models/{arch}_{dataset}_seed{N}.pth`) and results directory
(`…_seed{N}/`), so nothing collides.

## Submit

```bash
bash hpc/prep_data.sh                # once, on the LOGIN node: stage MNIST + CIFAR
qsub hpc/smoke_testegpu.pbs          # 30-min sanity check; look for "SMOKE OK"

bash hpc/submit_all.sh               # DRY RUN: prints every qsub it would issue
bash hpc/submit_all.sh --go          # actually submit
qstat -u "$USER"
```

`submit_all.sh` issues up to three phases per (config, seed) — 60 jobs in the
default grid (20 cheap + 20 exact + 20 K-sweep):

1. **`hpc/cheap_gpu.pbs`** on `umagpu` — trains the checkpoint and runs the five
   cheap schemes (magnitude, fim, f_dist_one_shot, f_dist_iterative,
   f_dist_global).
2. **`hpc/exact_fdist_par128.pbs`** on `par128` — exact `f_dist` only, reusing
   that checkpoint (`SKIP_TRAIN=true`), chained with `-W depend=afterok:`.
3. The same script again per `K` value, for the K-sweep table (SimpleNN/MNIST
   only by default).

That grid produces **140 pruning runs from 60 jobs**: all 6 schemes × 2
architectures × 2 datasets × 5 seeds (120), plus the K sweep on SimpleNN/MNIST
(20 more; K=3 is shared with the main table).

Then build the tables and the plotting data:

```bash
python -m src.aggregate_results --out paper_tables.md                 # main tables
python -m src.aggregate_results --format latex --out paper_tables.tex
python -m src.aggregate_results --by-k --out k_tables.md              # AUC and cost vs K
python -m src.aggregate_results --format json --out results.json      # for plots
```

Each of those renders **two tables per architecture/dataset**: AUC of normalised
accuracy, and AUC of normalised MCC. Accuracy is the headline number; MCC is the
chance-corrected check that a scheme preserves the decision structure instead of
collapsing onto the majority class. The accuracy tables also carry the
floor-corrected column `(AUC−floor)/(1−floor)`; the MCC tables do not, because a
collapsed one-class predictor scores MCC 0, making the correction the identity.
Precision and F1 are still recorded and exported to JSON, just not tabulated
(`--metrics acc_norm f1_norm` overrides the pair).

Runs are grouped by (dataset, architecture, scheme, **K**), so a K-sweep run is
never averaged into the main table's K=3 row, and a repeated seed within a group
is reported as a loud `!!` warning rather than silently averaged.

`--format json` writes everything needed to draw figures later without re-reading
the results tree (`schema: fdist-aggregate-v1`): per group, the `pruning_ratio`
grid, all four metric curves **per seed** plus their `mean`/`std` across seeds, AUC and
floor-corrected AUC (per seed, mean, std), and `compute_seconds` (per seed, mean,
std, device, worker count). Per-seed arrays are kept alongside the summaries so
error bands, individual traces and significance tests all remain possible.

## Why exact `f_dist` runs on a CPU queue, not a GPU

Two measured reasons.

**The GPU barely helps.** These models are far too small to saturate an A100 —
the Fisher evaluation is launch-bound. Backing the rate out of a previous run's
timestamps: 3.1 s per ViT Fisher evaluation on the A100 against 11.5 s on an
8-core laptop CPU. A 3.7× speedup is not what a GPU queue is for.

**The CPU queues are far bigger.** Per-user concurrent-job caps are the binding
constraint:

| queue | walltime | resources | max concurrent (user) |
|---|---|---|---|
| `umagpu` | 7 d | 1 A100, 16 cores | **2** |
| `miggpu` | 7 d | ½ A100, 8 cores | 2 |
| `duasgpus` | 3 d | 2 A100, 32 cores | 1 |
| **`par128`** | **7 d** | **128 cores, 488 GB** | **6** |
| `par16` | 10 d | 16 cores | 4 |
| `serial` | 10 d | 1 core | 10 |

Job arrays (`qsub -J`) are **not documented** for this cluster, which is why
`submit_all.sh` loops over individual `qsub` calls. The centre's docs confirm
**OpenPBS** (so `-v VAR=...` and `-W depend=afterok:` are available) and give
`#PBS -q par128` + `#PBS -l nodes=1:ppn=128` as the par128 form, which is what
`exact_fdist_par128.pbs` uses. Every `#PBS` directive sits above the first
executable line, since PBS ignores directives that appear after one. Job names
are kept to ≤ 9 characters (`ch_vc_s3`, `k17_nm_s0`) so they are safe under any
PBS `-N` length limit and make readable `$HOME/<name>.o<jobid>` files. The grid
scripts use `#PBS -m a` (abort only) — `-m abe` across 60 jobs would send ~180
e-mails; the one-off smoke test keeps `-m abe`.

## The forward-mode fast path

Exact `f_dist` scores each coordinate with `F_ii` evaluated at a singly-perturbed
parameter vector. The original implementation obtained that one number by
running a full reverse-mode Fisher (C backward passes per sample over **all** P
parameters) and discarding the other P−1 entries.

`fisher_entries_forward` (in `src/utils/fim_calculator.py`) uses the identity

```
F_ii = E_x Var_{c ~ p(·|x)} [ ∂logits_c / ∂θ_i ]
```

so a single JVP per (coordinate, sample) yields that entry for all classes at
once. Measured speedup per probe, single core, 500 Fisher samples:

| | reverse (old) | forward (new) | speedup |
|---|---|---|---|
| SimpleNN / MNIST | 0.612 s | 0.007 s | 89× |
| SimpleNN / CIFAR-10 | 2.046 s | 0.006 s | 325× |
| SimpleViT / MNIST | 13.05 s | 1.270 s | 10.3× |
| SimpleViT / CIFAR-10 | 15.13 s | 1.700 s | 8.9× |

This is exact, not an approximation. `tests/test_fisher_forward_equivalence.py`
(450 assertions) checks it against the production Fisher on a *literally
perturbed* model — float64 worst relative error ~1e-15 — and
`tests/test_fdist_fast_equivalence.py` checks that the pruner built on it selects
an **identical pruning mask** with bit-exact surviving weights. Set
`pruning.f_dist_fast: false` to fall back to the original loop.

Evaluation at each sweep point is shared the same way: accuracy, precision, F1
and MCC are reductions of one set of predictions, so `evaluate_metrics` makes a
single forward pass instead of four. That is ~4× off every sweep point (24.5 s →
6.0 s for the ViT on CIFAR-10) — negligible next to the Fisher probes for exact
`f_dist`, but it is most of the cost of the five cheap schemes.

## Probe sharding: how the ViT rows become affordable

A single process cannot use a 128-core node for this. Measured on the real ViT,
one probe costs ~1.35 s on a core and torch's intra-op threading saturates at
about 4 threads (1.40 s at 1 thread, 0.89 s at 4, 0.86 s at 8); neither
`probe_batch` nor `sample_chunk` moves it, and MPS was flat at 0.73 s/probe for
every `probe_batch`. The work is compute-bound per core with too little internal
parallelism — which is exactly the profile that shards perfectly across
*processes*.

`src/utils/fisher_parallel.py` splits the nonzero-coordinate list across
`FDIST_WORKERS` single-threaded worker processes (one `ProbePool` per pruning
step, so the model is inherited through the fork rather than pickled per task).
`hpc/exact_fdist_par128.pbs` sets `FDIST_WORKERS` to the node's core count and
pins `OMP_NUM_THREADS=1`.

**This does not change the numbers.** Sharding is by coordinate, never by
sample, so every coordinate still accumulates over the whole Fisher subset in
the original batch order; shard boundaries are additionally snapped to
`probe_batch` multiples so each shard replays the same vmap batch shapes (and
therefore the same BLAS blocking) the serial path would have used.
`tests/test_fisher_sharding_equivalence.py` asserts **bit-identical** Fisher
values, pruning masks, and surviving weights against the serial path at 2/3/4/5/7/8
workers, on both architectures, in float32 and float64, including a 50%
pre-pruned model. Measured scaling on this laptop's 4 performance cores: 2.96×
(~74% efficiency).

Cost per seed at Δ=0.1, K=3 (probes = N·(K−1)·(M+1)/2):

| | core-days/seed | at 64 eff. cores | at 95 eff. cores |
|---|---|---|---|
| SimpleNN / MNIST | 0.05 | minutes | minutes |
| SimpleNN / CIFAR-10 | 0.17 | minutes | minutes |
| SimpleViT / MNIST | 86.5 | 1.35 d | 21.9 h |
| SimpleViT / CIFAR-10 | 117.1 | 1.83 d | 1.23 d |

Every run fits the 7-day `par128` walltime with margin even at the pessimistic
end. Total for the exact-`f_dist` line: **~6,200 UA** (10 ViT runs ≈ 6,108;
10 NN runs ≈ 6; 20 K-sweep runs ≈ 60). Check your allocation before submitting.

## The K sweep

`K` (`pruning.f_dist_avg_points`) is the number of points in the exact-`f_dist`
path average, and cost is proportional to `K−1` probes per coordinate. `FDIST_K`
overrides the config and tags the results directory `_K{n}`, so one config drives
the whole sweep.

Phase 3 runs `K ∈ {2,5,9,17}` on SimpleNN/MNIST; together with the `K=3` that
phase 2 already produces, the ladder is **{2, 3, 5, 9, 17}**. That choice is
deliberate: `linspace(1,0,K)` then gives `{1,0} ⊂ {1,½,0} ⊂ quarters ⊂ eighths ⊂
sixteenths` — each grid a strict superset of the last, so the sweep is a genuine
dyadic refinement — while the cost, `K−1` probes per coordinate, doubles exactly
**1, 2, 4, 8, 16** with no gaps. `K=3` is not rerun: phase 2 produces it on the
same config, seeds, sweep grid and node type, so it is directly comparable.

Tune with `K_VALUES=...`, `K_CONFIGS=...` (add `configs/nn_cifar10.yaml` for the
CIFAR NN too), or `K_SWEEP=0` to skip.

## Notes

- `fim_device: auto` in all four configs, so the cheap sweeps use the allocated
  A100. `par128` jobs have no GPU and fall back to CPU automatically.
- `paths.dataset_path` decides where the datasets are read from and is what
  `prep_data.sh` stages into, so pointing it at node-local scratch needs no other
  change. `paths.log_path` is where `run_experiment.sh` tees each stage;
  `FDIST_LOG_DIR` overrides it per job.
- Live PBS output (`-k oe -j oe`) lands in `$HOME/<jobname>.o<jobid>` and is
  moved to `outputs/` when the job ends.
- Plot rendering is headless-safe (matplotlib Agg); no X needed.
- `par128` costs 32 UA/hour, i.e. 0.25 UA per core-hour.
- Job names are unique per (config, seed, K), so nothing collides in `outputs/`.
