# Fisher-Distance Pruning

Reference implementation for **Fisher Information Distance Pruning for Neural
Architectures** (Berman, Fu, Hirst, Obirai).

Pruning a weight is a *finite* move in model space, from its trained value to
zero. Model space carries a natural metric, the Fisher information, so that move
has a length. Taking that length as the importance score gives

```
s_k  =  sqrt( (1/K) * sum_alpha  I_kk( theta | theta^k -> alpha*theta^k ) )  *  |theta^k|
```

where `alpha` runs over `K` points from 1 to 0 along the coordinate line. Holding
the metric constant collapses this to `sqrt(I_kk) * |theta^k|`, which is exactly
magnitude pruning multiplied by Fisher pruning: the two standard criteria fall
out as the leading term of one geometric quantity, rather than being combined by
hand. The rest of the family refines how faithfully the metric is resolved along
the path.

## Installation

```bash
git clone <repository-url> && cd Fdist_NNPruning
uv venv && source .venv/bin/activate
uv pip install -r environment/requirements.txt
```

See [environment/README.md](environment/README.md) for alternatives to `uv`.

## Quick start

```bash
./try-script.sh                      # train (if needed), then every scheme, on src/config.yaml
./try-script.sh configs/nn_mnist.yaml
```

Or drive the two stages directly:

```bash
python -m src.train_model            # writes models/{arch}_{dataset}.pth
python -m src.run_pruning            # sweeps 0 -> 100% sparsity, writes results/
```

Both read `src/config.yaml`; `model.common.model_type` (`nn` | `cnn` |
`transformer`) picks the architecture and `dataset.name` (`mnist` |
`fashion_mnist` | `cifar10`) the dataset. Any combination works, input shapes are
handled automatically, and checkpoints and results are namespaced per
architecture, dataset and seed so concurrent runs never collide.

## Pruning schemes

`pruning.pruning_scheme` selects one of six. The first two are the baselines; the
rest are the Fisher-distance family, in increasing order of how carefully they
resolve the metric along the pruning path.

| `pruning_scheme` | Score for weight `k` | Fisher work per sweep point |
|---|---|---|
| `magnitude` | `\|theta^k\|` | none |
| `fim` | `I_kk(theta)` | 1 evaluation |
| `f_dist_one_shot` | `sqrt(I_kk(theta*)) * \|theta^k\|` | 1 evaluation, always at the dense point `theta*` |
| `f_dist_iterative` | `sqrt(I_kk(theta)) * \|theta^k\|` | 1 evaluation, at the current pruned model |
| `f_dist_global` | `\|theta^k\| * mean_alpha sqrt(I_kk(alpha*theta))` | `K` evaluations |
| `f_dist` | `sqrt( mean_alpha I_kk(theta\|theta^k -> alpha*theta^k) ) * \|theta^k\|` | 1 evaluation + `K-1` probes **per surviving weight** |

`magnitude`, `fim` and `f_dist_one_shot` re-prune from a fresh copy of the dense
model at every sweep point, so zeros never accumulate ambiguously; the other
three advance one running model by `sweep.step` and recompute the metric on the
current pruned state.

`f_dist` is the exact per-coordinate measure the others approximate, and the
reason it is normally considered impractical: it perturbs one coordinate at a
time, so cost scales with the parameter count. Two things bring it within reach
even for the ~546k-parameter ViT:

- a **forward-mode (JVP) Fisher kernel** using
  `I_ii = E_x Var_{c~p(.|x)}[d logits_c / d theta_i]`, which reads one diagonal
  entry per forward-mode pass instead of running `C` backward passes over all
  parameters to keep one number (8.9x to 325x per probe, and exact, not
  approximate);
- **process-level probe sharding** across cores, bit-identical to the serial path.

`f_dist_global` is the cheap alternative: it shrinks all weights together along
the ray `alpha*theta` and reads every diagonal entry from each evaluation, so
cost is `K` Fisher evaluations per step regardless of model size.

## Reproducing the paper

The four paper testbeds are the ready-made configs, each pinned to the published
settings (`sweep.step: 0.1`, `f_dist_avg_points: 3`, `fim_subset_size: 500`,
warm-start off, all four metrics, no validation split):

| | SimpleNN (2x64 MLP) | SimpleViT (d4/e128/h4/p4) |
|---|---|---|
| MNIST | `configs/nn_mnist.yaml` (55,050 par.) | `configs/vit_mnist.yaml` (539,914 par.) |
| CIFAR-10 | `configs/nn_cifar10.yaml` (201,482 par.) | `configs/vit_cifar10.yaml` (545,930 par.) |

The full grid is 6 schemes x 2 architectures x 2 datasets x 5 seeds, plus a
K-ladder, and is submitted as 60 PBS jobs producing 140 runs. See
[hpc/README.md](hpc/README.md) for the cluster recipe, walltimes and the
verification checks.

Then aggregate the per-run JSONs and regenerate the paper's tables and figures:

```bash
python -m src.aggregate_results --root results --format json --out paper_data.json
python -m src.make_paper_tables > tables.tex
python -m src.make_paper_figures --out figures/          # PDFs + PNGs
```

Both generators read only `paper_data.json`, so a number can never drift between
a figure, a table and the prose. `make_paper_figures` writes to `../overleaf/figures`
by default, which assumes the paper source sits alongside this repository; pass
`--out` for anywhere else. `aggregate_results` also renders markdown or
LaTeX tables directly (`--format markdown`, `--by-k` for the K-ladder); see
[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for what each run records and how it
is aggregated.

## Repository layout

```
src/
  train_model.py            training loop, per-architecture recipes
  run_pruning.py            the sparsity sweep; scheme dispatch and logging
  aggregate_results.py      per-run JSONs -> multi-seed tables (markdown/LaTeX/JSON)
  make_paper_tables.py      paper_data.json -> overleaf/tables.tex
  make_paper_figures.py     paper_data.json -> overleaf/figures/
  subset_experiment.py      how far the Fisher subset size can be cut (rank correlation)
  config.yaml               the single source of settings, fully commented
  models/                   SimpleNN, SimpleCNN, SimpleViT
  pruning/                  one module per scheme, all subclassing BasePruner
  utils/                    Fisher calculators, data, metrics, seeding
configs/                    the four paper testbeds (+ a ViT smoke test)
hpc/                        PBS job scripts and the grid submitter
tests/                      equivalence tests for the fast paths
```

To add a scheme, subclass `BasePruner`, implement `apply_pruning`, and register
it in `src/pruning/__init__.py` and `build_pruner` in `src/run_pruning.py`
(details in [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#adding-a-pruning-scheme)).

## Tests

The optimised Fisher paths are justified by equivalence rather than by
benchmark, so each has a test asserting it reproduces the reference
implementation:

```bash
python -m tests.test_fisher_forward_equivalence    # JVP kernel   == backprop Fisher
python -m tests.test_fdist_fast_equivalence        # fast f_dist  == per-coordinate loop (same weights pruned)
python -m tests.test_fisher_sharding_equivalence   # sharded      == serial, bit-identical
python -m tests.test_single_pass_metrics           # one-pass metrics == per-metric passes
```

## Citation

```bibtex
@article{Berman:2026fdist,
    author  = {Berman, David S. and Fu, Yen-Yu and Hirst, Edward and Obirai, Thelma Chiwete},
    title   = {{Fisher Information Distance Pruning for Neural Architectures}},
    year    = {2026}
}
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any
improvements or bug fixes.

## License

MIT. See [LICENSE](LICENSE).
