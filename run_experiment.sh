#!/usr/bin/env bash
set -euo pipefail


PYTHON=${PYTHON:-python3}
BASE_CONFIG=${BASE_CONFIG:-src/config.yaml}

TRAIN_MODULE=${TRAIN_MODULE:-src.train_model}
PRUNE_MODULE=${PRUNE_MODULE:-src.run_pruning}

# Set PARALLEL_FOLDS=true to run folds in parallel (uses more RAM)
# Set PARALLEL_FOLDS=false or unset to run sequentially (default)
PARALLEL_FOLDS=${PARALLEL_FOLDS:-false}

# ========================

# Read num_folds from the active config (BASE_CONFIG, default src/config.yaml)
NUM_FOLDS=$($PYTHON - "$BASE_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r"))
print(cfg.get("cross_validation", {}).get("num_folds", 1))
PY
)

echo "Cross-validation folds: ${NUM_FOLDS}"
if [[ "$PARALLEL_FOLDS" == "true" ]]; then
  echo "Mode: PARALLEL (all folds run simultaneously - requires ${NUM_FOLDS}x RAM)"
else
  echo "Mode: SEQUENTIAL (folds run one at a time)"
fi
echo

run_fold() {
  local fold=$1
  echo "======================================================================"
  echo "==================== FOLD ${fold}/${NUM_FOLDS} ===================="
  echo "======================================================================"
  echo

  echo "==[1/7] Train model with config: ${BASE_CONFIG} =="
  $PYTHON -m "$TRAIN_MODULE" "$BASE_CONFIG"


  # Model type, canonical dataset, and the checkpoint path resolved the same
  # way run_pruning.py does
  read -r MODEL_TYPE DATASET CKPT_PATH <<< "$($PYTHON - "$BASE_CONFIG" <<'PY'
import sys, yaml
from src.utils.data_loader import normalize_dataset_name
from src.utils.model_builder import build_model_from_config, resolve_checkpoint_path
cfg = yaml.safe_load(open(sys.argv[1], "r"))
mt = str(((cfg.get("model", {}) or {}).get("common", {}) or {}).get("model_type", "NN")).strip().lower()
ds = normalize_dataset_name((cfg.get("dataset", {}) or {}).get("name", "mnist"))
_, arch = build_model_from_config(cfg, dataset_name=ds)
print(mt, ds, resolve_checkpoint_path(cfg, arch, ds))
PY
)"

  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    echo "Check train_model file name is same as config.paths.pretrained_model_path"
    exit 1
  fi

  echo "Checkpoint found: $CKPT_PATH"
  echo

  make_tmp_cfg () {
    local scheme="$1"
    local out_cfg
    out_cfg="$(mktemp -t cfg_${scheme}_XXXX.yaml)"

    $PYTHON - <<PY
import yaml
cfg = yaml.safe_load(open("${BASE_CONFIG}","r"))

# Ensure pruning enabled
cfg.setdefault("pruning", {})
cfg["pruning"]["enable_pruning"] = True
cfg["pruning"]["pruning_scheme"] = "${scheme}"

# Ensure pretrained_model_path using ckpt
cfg.setdefault("paths", {})
cfg["paths"]["pretrained_model_path"] = "${CKPT_PATH}"

# Add fold number to results
cfg.setdefault("cross_validation", {})
cfg["cross_validation"]["current_fold"] = ${fold}

with open("${out_cfg}","w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    echo "$out_cfg"
  }

  echo "==[2/7] Run MAGNITUDE pruning =="
  CFG_MAG=$(make_tmp_cfg "magnitude")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_MAG"
  rm -f "$CFG_MAG"
  echo

  echo "==[3/7] Run FIM (Fisher) pruning =="
  CFG_FIM=$(make_tmp_cfg "fim")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_FIM"
  rm -f "$CFG_FIM"
  echo

  echo "==[4/7] Run F_DIST_ONE_SHOT (FIM x Magnitude One-Shot) pruning =="
  CFG_FDIST_OS=$(make_tmp_cfg "f_dist_one_shot")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_FDIST_OS"
  rm -f "$CFG_FDIST_OS"
  echo

  echo "==[5/7] Run F_DIST_ITERATIVE (FIM x Magnitude Iterative) pruning =="
  CFG_FDIST_IT=$(make_tmp_cfg "f_dist_iterative")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_FDIST_IT"
  rm -f "$CFG_FDIST_IT"
  echo

  echo "==[6/7] Run F_DIST_GLOBAL (Fisher-distance, batched alpha-scan) pruning =="
  CFG_FDIST_GL=$(make_tmp_cfg "f_dist_global")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_FDIST_GL"
  rm -f "$CFG_FDIST_GL"
  echo

  # Exact per-coordinate f_dist: ~#active-weights full Fisher evals per step,
  # only feasible for the small NN on 28x28 datasets (~55k params; the cifar10
  # NN has ~201k and the ViT ~546k).
  if [[ "$MODEL_TYPE" == "nn" && ( "$DATASET" == "mnist" || "$DATASET" == "fashion_mnist" ) ]]; then
    echo "==[7/7] Run F_DIST (exact Fisher-distance) pruning =="
    CFG_FDIST=$(make_tmp_cfg "f_dist")
    $PYTHON -m "$PRUNE_MODULE" "$CFG_FDIST"
    rm -f "$CFG_FDIST"
    echo
  else
    echo "==[7/7] Skipping exact f_dist (model_type=${MODEL_TYPE}, dataset=${DATASET}; infeasible at this scale) =="
    echo
  fi

  echo "Fold ${fold}/${NUM_FOLDS} complete."
  echo
}

# Run folds either in parallel or sequentially
if [[ "$PARALLEL_FOLDS" == "true" ]]; then
  # Parallel execution: launch all folds as background jobs
  for fold in $(seq 1 "$NUM_FOLDS"); do
    run_fold "$fold" &
  done
  
  # Wait for all background jobs to complete
  wait
else
  # Sequential execution: run one fold at a time
  for fold in $(seq 1 "$NUM_FOLDS"); do
    run_fold "$fold"
  done
fi

echo "All done."
echo "Completed ${NUM_FOLDS} fold(s)."
echo "Results are under ./results/ (each run creates a timestamped folder)."

