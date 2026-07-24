# Fdist_NNPruning
Code for implementation of the novel Fisher-distance pruning scheme, with its application to Neural Networks and Transformers.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
   ...the `cd` into it locally.

2. Follow the instructions in the `environment` folder to set up the venv.

## Usage

1. Configure the parameters in the `config/config.yaml`, `config/datasets.yaml`, `config/models.yaml`, and `config/pruning.yaml` files according to your needs.

2. Run the `train_model` script to start training a new model:
   ```
   python -m src.train_model
   ```

3. Run the `run_pruning` script to start pruning the model:
   ```
   python -m src.run_pruning
   ```

   Alternatively, `./try-script.sh` trains the configured model if its checkpoint is
   missing and then runs every pruning scheme in sequence.

## Architectures & Datasets

`model.common.model_type` in `src/config.yaml` selects the architecture and
`dataset.name` the dataset (`mnist`, `fashion_mnist`, `cifar10` — `cifar` is
accepted as an alias). Any combination works; input shapes are handled
automatically, and checkpoints are stored per combination as
`models/{arch}_{dataset}.pth`. **The default for both architectures is CIFAR-10**,
so the two testbeds are directly comparable on the same task (and CIFAR-10
degrades gradually under pruning, letting schemes separate over the whole
0–100% range, whereas MNIST curves stay flat until ~70% sparsity).

- **`nn`** — `SimpleNN` MLP (input→64→64→10; images are flattened, so 784 inputs
  / ~55k params on MNIST-sized data and 3072 inputs / ~201k params on CIFAR-10).
- **`transformer`** — `SimpleViT`, a compact ViT-Lite-style Vision Transformer
  (conv patchify, learned positional embedding, pre-LN blocks with explicit
  softmax attention, mean pooling; ~546k params at the default
  patch 4 / dim 128 / depth 4 / 4 heads) trained from scratch on the
  standard compact-ViT benchmark, CIFAR-10.

Transformer notes:
- Fisher must use `fim_calculate_method: "backprop"` (nngeometry's KFAC has no
  LayerNorm/attention support).
- By default LayerNorm parameters and the positional embedding are excluded from
  pruning (`pruning.prunable_exclude`, standard practice in transformer sparsity
  work); reported pruning ratios remain fractions of ALL parameters.
- The exact per-coordinate `f_dist` scheme is computationally infeasible at
  transformer scale; `f_dist_global` is the path-averaged scheme to use there.

## Pruning Schemes

The project implements the following pruning schemes:

- **Magnitude Pruning** (`magnitude`): Removes weights based on their magnitude.
- **FIM Pruning** (`fim`): Utilizes the Fisher Information Matrix for pruning.
- **Magnitude x FIM Pruning (One Shot)** (`f_dist_one_shot`): Combines magnitude and FIM pruning in a single pass.
- **Magnitude x FIM Pruning (Iterative)** (`f_dist_iterative`): Applies magnitude and FIM pruning iteratively.
- **Square Root of Averaged Magnitude x FIM** (`f_dist`): Uses the square root of the averaged values for pruning (exact per-coordinate path average; feasible only for the NN on MNIST-sized data, ~55k params).
- **Fisher-distance, global path** (`f_dist_global`): batched α-scan — evaluates the
  Fisher diagonal at K models `α·θ` (all weights shrunk together) and scores
  `|w|·mean_α √F_ii(αθ)`; K Fisher evaluations per pruning step instead of
  #weights×(K−1), making full-range path-averaged sweeps feasible at any scale.

**Magnitude warm-start** (`pruning.warm_start`, on by default): the f_dist family can
be started by cheap magnitude pruning up to `ratio` (default 0.8) and only run the
Fisher-based scheme for the high-sparsity tail, while still producing the full
0→100% curve — this keeps exact `f_dist` within HPC walltimes. Set `enabled: false`
for the full-range comparison.

Result JSONs additionally contain per-metric AUC of the normalized sparsity–accuracy curves.
Each run is written to `{paths.results_dir}/{arch}_{dataset}/{scheme}_..._{timestamp}/`
(override the root with `FDIST_RESULTS_DIR`), so concurrent runs never collide. For
running the two architectures as parallel HPC jobs, see [hpc/README.md](hpc/README.md).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.