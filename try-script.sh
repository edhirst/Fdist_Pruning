#!/usr/bin/env bash
set -euo pipefail

# fix in repo's root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python3}
# Optional first argument: config path (defaults to src/config.yaml). This lets
# the HPC job scripts point at a per-experiment config (e.g. configs/nn_cifar10.yaml)
# while sharing the same tested train -> prune-all-schemes pipeline.
CONFIG="${1:-src/config.yaml}"

echo "Running from: $ROOT_DIR"
echo "Config: $CONFIG"
echo

# Model type (lowercased), canonical dataset name, and the checkpoint path
# resolved the same way run_pruning.py does
# (explicit paths.pretrained_model_path, else models/{arch}_{dataset}.pth)
read -r MODEL_TYPE DATASET CKPT_PATH <<< "$($PYTHON - <<PY
import yaml
from src.utils.data_loader import normalize_dataset_name
from src.utils.model_builder import build_model_from_config, resolve_checkpoint_path
cfg = yaml.safe_load(open("${CONFIG}", "r"))
mt = str(((cfg.get("model", {}) or {}).get("common", {}) or {}).get("model_type", "NN")).strip().lower()
ds = normalize_dataset_name((cfg.get("dataset", {}) or {}).get("name", "mnist"))
_, arch = build_model_from_config(cfg, dataset_name=ds)
print(mt, ds, resolve_checkpoint_path(cfg, arch, ds))
PY
)"

echo "Model type: $MODEL_TYPE"
echo "Dataset: $DATASET"
echo "Checkpoint path: $CKPT_PATH"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found — training first."
  $PYTHON -m src.train_model "$CONFIG"
  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: training finished but checkpoint still not found at: $CKPT_PATH"
    exit 1
  fi
else
  echo "Checkpoint found. Skipping training."
fi
echo

make_prune_cfg () {
  local scheme="$1"
  local out_cfg
  out_cfg="$(mktemp "/tmp/prune_${scheme}_XXXX.yaml")"

  $PYTHON - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG}", "r"))

cfg["pruning"]["enable_pruning"] = True
cfg["pruning"]["pruning_scheme"] = "${scheme}"

with open("${out_cfg}", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("${out_cfg}")
PY
}

run_scheme () {
  local scheme="$1"
  echo "== Pruning: ${scheme} =="
  local cfg_file
  cfg_file="$(make_prune_cfg "$scheme")"
  $PYTHON -m src.run_pruning "$cfg_file"
  rm -f "$cfg_file"
  echo
}

run_scheme magnitude
run_scheme fim
run_scheme f_dist_one_shot
run_scheme f_dist_iterative
run_scheme f_dist_global

# Exact per-coordinate f_dist needs ~#active-weights full Fisher evaluations per
# pruning step: only feasible for the small NN on 28x28 datasets (~55k params).
# On cifar10 the NN grows to ~201k params (3072 inputs) — still infeasible.
if [[ "$MODEL_TYPE" == "nn" && ( "$DATASET" == "mnist" || "$DATASET" == "fashion_mnist" ) ]]; then
  run_scheme f_dist
else
  echo "== Skipping exact f_dist (model_type=${MODEL_TYPE}, dataset=${DATASET}; infeasible at this scale — f_dist_global covers the path-averaged scheme) =="
  echo
fi

echo "Done."
