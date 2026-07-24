# Running on the CENAPAD-SP "Lovelace" cluster (OpenPBS)

Two self-contained jobs, one per architecture, designed to run **at the same time**
with **no shared output files**:

| Job script | Architecture | Config | Checkpoint | Results directory |
|---|---|---|---|---|
| `hpc/nn_cifar10.pbs`  | SimpleNN (2×64 MLP) | `configs/nn_cifar10.yaml`  | `models/SimpleNN_h2_n64_cifar10.pth`        | `results/SimpleNN_h2_n64_cifar10/` |
| `hpc/vit_cifar10.pbs` | SimpleViT (ViT)     | `configs/vit_cifar10.yaml` | `models/SimpleViT_d4_e128_h4_p4_cifar10.pth` | `results/SimpleViT_d4_e128_h4_p4_cifar10/` |

Each job selects its experiment with `export BASE_CONFIG=configs/<...>.yaml` and then
runs `./run_experiment.sh`, which trains the model and runs every pruning scheme over
0→100%. The `.pbs` files assume the repo is checked out as a directory named `github`
under the submit directory (they `cd "$PBS_O_WORKDIR"` then `cd github`) — rename that
line if your clone differs.

## Submit

Run everything **from the repo root** (that's what the `.pbs` scripts assume — they
`cd "$PBS_O_WORKDIR"` and expect `run_experiment.sh`, `configs/`, `.venv` right there).
If instead your repo lives in a subdirectory named `github`, uncomment the `cd github`
line in each script and submit from the parent.

```bash
# 0. one-time, on the LOGIN node (compute nodes have no internet):
#    stage the dataset + create dirs so the jobs don't race to download.
bash hpc/prep_data.sh

# 1. SMOKE TEST FIRST (testegpu, ~5 min, 30-min cap): confirms module + venv + GPU +
#    torch.func Fisher + the full pipeline actually work on the cluster. Isolated
#    outputs (models/smoke_*, results_smoke/). Check its log for "SMOKE OK".
qsub hpc/smoke_testegpu.pbs

# 2. once the smoke passes, submit the two real jobs — they run
#    independently and simultaneously:
qsub hpc/nn_cifar10.pbs
qsub hpc/vit_cifar10.pbs

qstat -u "$USER"        # watch the queue
```

The scripts load `miniconda3/22.11.1-gcc-9.4.0` and activate the repo's `.venv`
(`source .venv/bin/activate`) — adjust the module name / activation to your account, and
create that venv once (torch, torchvision, nngeometry, scikit-learn, scipy, matplotlib,
pyyaml). The GPU queues allocate resources automatically, so no `-l nodes=...` line is
needed. Live job output (`-k oe -j oe`) lands in `$HOME/<jobname>.o<jobid>` and is moved
to `outputs/` when the job finishes.

## Why there is no overwriting

Every run writes under a **per-(architecture, dataset) subdirectory**:

```
results/{arch}_{dataset}/{scheme}_pruning_result_figure_{timestamp}/
```

The two architectures have different `{arch}` (`SimpleNN_h2_n64` vs
`SimpleViT_d4_e128_h4_p4`), so their result trees never intersect — even if both
jobs run the same scheme in the same second. The same holds for the trained
**checkpoints** (distinct filenames) and the PBS **job logs**, which are named after
the distinct job names (`fdist_nn_cifar10.o<jobid>` vs `fdist_vit_cifar10.o<jobid>`)
and moved into `outputs/` at the end. The two jobs only ever *read* the shared
`data/CIFAR10`, which is why it is staged once up front.

> Extra isolation if you want it: set `paths.results_dir` in a config, or export
> `FDIST_RESULTS_DIR=/scratch/$USER/run_nn` before the job, to send a job's outputs
> to a completely separate root.

## Walltime — is exact `f_dist` the problem?

Short answer: **not for these CIFAR experiments.** On the `umagpu` queue (one full
A100, up to **7 days** walltime) each job finishes in well under an hour:

- ViT: ~15–30 min training + ~10–20 min for all pruning sweeps.
- NN: a few minutes end-to-end.

The only genuinely walltime-hostile scheme is the **exact per-coordinate `f_dist`**,
which needs ≈ (#active weights) × (K−1) full Fisher evaluations *per pruning step*.
That is why it is **automatically skipped** for anything larger than the small NN on
28×28 data — including *both* CIFAR architectures. So it never runs in these jobs,
and the rest of the schemes (magnitude, fim, f_dist_one_shot, f_dist_iterative,
f_dist_global) are cheap: `f_dist_global` is only K=3 Fisher evaluations per step.

Where exact `f_dist` *does* run (NN on MNIST/Fashion-MNIST) and would blow the
walltime, use the **magnitude warm-start** (next section) to make it tractable.

## Magnitude warm-start (`pruning.warm_start`, on by default)

To spend the Fisher-evaluation budget only where sparsity is high, the f_dist family
can be **warm-started by magnitude**: prune 0→`ratio` cheaply by magnitude, then run
the f_dist scheme only for the `ratio`→1.0 tail. The full 0→100% curve is still
produced. Configured in any config:

```yaml
pruning:
  warm_start:
    enabled: false  # applies to f_dist, f_dist_iterative, f_dist_global, f_dist_one_shot
    ratio: 0.8      # if enabled: magnitude up to 80% sparsity, then f_dist for 80%->100%
```

- The two CIFAR experiment configs (`configs/{nn,vit}_cifar10.yaml`) ship with it
  **OFF**, so `f_dist_global` and the rest of the f_dist family are tested over the
  **full 0->100% range** — the project's actual thesis. This is affordable: a
  full-range `f_dist_global` sweep is ~13 min on a CPU (measured on the 546k-param
  ViT at 500 Fisher samples) and faster on the A100; `f_dist_iterative` ~4 min.
- Turn it **on** (`enabled: true`) only to focus the Fisher-evaluation budget on the
  high-sparsity tail — e.g. to make exact per-coordinate `f_dist` tractable on MNIST.
  It remains on by default in the base `src/config.yaml`.

## Notes

- `fim_device` is `auto` in these configs, so Fisher runs on the allocated A100.
  (The repo default is `cpu`, which is specific to Apple-Silicon MPS where
  `torch.func` per-sample gradients are slow; CUDA does not have that problem.)
  If you ever see the FIM sweeps underperform on a node, set `fim_device: cpu`.
- Lighter queues work too and may schedule faster given how small these models are:
  swap `-q umagpu` for `-q miggpu` (½ A100, 7 days) or `-q miggpu24h` (½ A100, 1 day).
- Plot rendering is headless-safe (`matplotlib` uses the Agg backend); no X needed.
