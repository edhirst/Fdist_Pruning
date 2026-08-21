# Pruning Evaluation Guide

How a pruning experiment is run, what it records, and how the per-run outputs
become the paper's tables and figures.

## Architectures

Both testbeds run through the same train → prune → evaluate pipeline and produce
the same JSONs/plots (`metadata.model` and `metadata.dataset` distinguish runs):

- `model_type: "nn"` — SimpleNN MLP (flattened input)
- `model_type: "transformer"` — SimpleViT compact Vision Transformer

Either architecture runs on any `dataset.name` from {`mnist`, `fashion_mnist`,
`cifar10`} (`cifar` accepted as an alias). The four paper configs live in
`configs/`: `nn_mnist`, `nn_cifar10`, `vit_mnist`, `vit_cifar10`.

Every scheme, including exact per-coordinate `f_dist`, now runs at both scales.
The forward-mode Fisher kernel and process-level probe sharding are what made
that affordable; see [hpc/README.md](hpc/README.md) for the derivation, the
measured speedups and the cost per seed.

## Running an experiment

```bash
python -m src.train_model                 # train the checkpoint for the active config
python -m src.run_pruning                 # sweep one scheme over 0 → 100% pruning
```

`./run_experiment.sh` drives both across every scheme and cross-validation fold,
and is what the HPC job scripts call, so a local run and a cluster run execute the
same pipeline. `./try-script.sh [config.yaml]` is a thin wrapper that trains only
if the checkpoint is missing and then hands over to `run_experiment.sh`.

Behaviour is selected with environment variables documented at the top of
`run_experiment.sh` — `BASE_CONFIG`, `SCHEMES`, `EXACT_FDIST`, `SKIP_TRAIN` — and,
for `run_pruning.py` itself, `FDIST_RESULTS_DIR`, `FDIST_SEED`, `FDIST_K` and
`FDIST_WORKERS`.

### Development knobs

Four config settings exist to make experimenting cheaper. All four ship set to
**exactly what the paper runs**, so the defaults reproduce the published numbers
and changing one is an opt-in.

| setting | paper value | what raising/changing it does |
|---|---|---|
| `evaluation.metrics` | all four | Drops metrics from the sweep and from the results JSON. All of them share one forward pass, so this changes what is *recorded*, not the runtime. `accuracy` is required. |
| `evaluation.validation_split` | `0.0` | Holds out that fraction of the **train** split and reports validation loss/accuracy each epoch, so you can watch for overfitting without touching the test set. The held-out part is served without augmentation and the partition is seeded. |
| `paths.dataset_path` | `data` | Where torchvision datasets live. Point it at fast node-local scratch on a cluster. `hpc/prep_data.sh` stages into whatever the configs name. |
| `paths.log_path` | `logs` | Where `run_experiment.sh` tees each train/prune stage. Set to `null` for stdout only; `FDIST_LOG_DIR` overrides it per run. |

Trimming `evaluation.metrics` is safe for the tables: `src/aggregate_results.py`
omits a row whose run did not record the metric being tabulated, and says so,
rather than failing or averaging a partial group.

### Magnitude warm-start (`pruning.warm_start`)

For the f_dist family, `warm_start` prunes 0→`ratio` cheaply by magnitude and runs
the Fisher-based scheme only for the `ratio`→1.0 tail, while still producing the
full 0→100% curve. It is enabled in `src/config.yaml` (`ratio: 0.8`) but
**disabled in all four `configs/*.yaml` paper configs**, because the published
comparison needs f_dist to drive the entire curve. The active setting is recorded
in `metadata.warm_start` of every results JSON.

## What a run records

At each sweep point `run_pruning.py` evaluates the pruned model and stores the
metric **normalised by the unpruned baseline**, so 1.0 is "as good as the dense
model" for every architecture:

| key | metric |
|---|---|
| `acc_norm` | classification accuracy |
| `precision_norm` | macro-averaged precision |
| `f1_norm` | macro-averaged F1 |
| `mcc_norm` | Matthews correlation coefficient |

Which of these are computed is `evaluation.metrics`; the default is all four, and
`metadata.metrics` records what a given run actually evaluated.

The JSON has three blocks: `metadata` (scheme, dataset, model, timestamp, seed,
`f_dist_avg_points`, `warm_start`, `fim_subset_size`), `results` (the
`pruning_ratio` grid and the four curves above) and `timing`
(`sweep_seconds`, `step_seconds`, `device`, `probe_workers`).

Each run is written to

```
{paths.results_dir}/{arch}_{dataset}/{scheme}_pruning_result_figure_{timestamp}[_fold{N}][_seed{N}][_K{n}]/
```

so concurrent jobs for different architectures, seeds or K never collide.
`FDIST_RESULTS_DIR` overrides the root.

## Aggregating into the paper tables

`src/aggregate_results.py` walks a results tree, groups runs by
(dataset, architecture, scheme, K) and reports mean ± std over seeds:

```bash
python -m src.aggregate_results --out paper_tables.md                  # accuracy + MCC
python -m src.aggregate_results --format latex --out paper_tables.tex
python -m src.aggregate_results --by-k --out k_tables.md               # AUC and cost vs K
python -m src.aggregate_results --format json --out results.json       # for plots
```

**Two tables per architecture/dataset.** By default one table is rendered for
accuracy and one for MCC. Accuracy is the headline number; MCC is the
chance-corrected check that a scheme is genuinely preserving the decision
structure rather than collapsing onto the majority class. Precision and F1 are
still recorded and exported to JSON, they are simply not tabulated — override
with `--metrics acc_norm f1_norm` if you want them.

**AUC** is the mean normalised metric over the swept range (trapezoid ÷ range),
re-integrated here from the stored curves so every entry uses one quadrature rule.
Runs within a group that used different sweep grids are flagged, because their
AUCs are not comparable.

**Floor correction.** A fully destroyed model still scores chance accuracy, and
chance differs between architectures, so the accuracy table also reports
`(AUC − floor) / (1 − floor)` — the fair cross-architecture number. MCC needs no
such correction: a collapsed one-class predictor scores exactly 0, so the floor is
0 and the correction is the identity. That column is therefore omitted from the
MCC tables, with a note saying so.

**Integrity warnings** are printed once, below the tables: runs with no seed
recorded, groups whose sweep grids disagree, and — loudly, as `!!` — duplicate
seeds within a group, which would inflate *n* and shrink the std if averaged.

`--format json` (schema `fdist-aggregate-v1`) writes everything needed to draw
figures later without re-reading the results tree: per group, the `pruning_ratio`
grid, all four metric curves **per seed** plus `mean`/`std` across seeds, AUC and
floor-corrected AUC, and `compute_seconds`. Per-seed arrays are kept alongside the
summaries so error bands, individual traces and significance tests remain possible.

## Using the evaluation helpers directly

```python
from src.utils.evaluation import (
    evaluate_metrics, evaluate_accuracy, evaluate_mcc,
    calculate_auc, count_nonzero_params, get_model_size_kb,
)

# All requested metrics from ONE pass over the loader -- what the sweep uses.
vals = evaluate_metrics(model, test_loader, ["accuracy", "mcc"], device)

# Single-metric helpers, each doing its own pass.
acc = evaluate_accuracy(model, test_loader, device)
auc = calculate_auc(pruning_percentages, accuracies)
```

The four metrics are pure reductions of the same argmax predictions, so
`evaluate_metrics` collects those once and reduces them four ways. Measured on
the 10k test sets, that is **~4× faster** than calling the four helpers
(SimpleViT/CIFAR-10: 24.5 s → 6.0 s per sweep point) and the values are
bit-identical — `tests/test_single_pass_metrics.py` asserts exact equality,
including the degenerate one-class case.

`src/utils/evaluation.py` also carries the matplotlib figure helpers
(`plot_metric_curves` for mean ± std bands, `plot_auc_comparison`,
`plot_accuracy_comparison`, `plot_model_size_comparison`) and
`statistical_comparison`, which runs paired t-tests between schemes. These are
not called by the pipeline; they are there for building the paper figures from
the aggregated JSON.

## Adding a pruning scheme

1. Implement it against the `BasePruner` interface in `src/pruning/`.
2. Register it in `build_pruner` in `src/run_pruning.py`.
3. Add it to `SCHEME_ORDER` and `SCHEME_LABEL` in `src/aggregate_results.py` so it
   gets a row, in the right place, in the tables.

## Running on the cluster

`hpc/` holds the OpenPBS job scripts for CENAPAD-SP Lovelace and
`hpc/submit_all.sh` submits the whole multi-seed grid. See
[hpc/README.md](hpc/README.md) for the submission steps, queue choice and
walltime analysis.

## Requirements

```bash
pip install torch torchvision matplotlib numpy scikit-learn scipy pyyaml
```
