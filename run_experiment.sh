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

# Read num_folds from config
NUM_FOLDS=$($PYTHON - <<'PY'
import yaml
cfg = yaml.safe_load(open("src/config.yaml","r"))
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

  echo "==[1/6] Train model with config: ${BASE_CONFIG} =="
  $PYTHON -m "$TRAIN_MODULE" "$BASE_CONFIG"


  CKPT_PATH=$($PYTHON - <<'PY'
import yaml
cfg = yaml.safe_load(open("src/config.yaml","r"))
print(cfg["paths"]["pretrained_model_path"])
PY
  )

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

  echo "==[2/6] Run MAGNITUDE pruning =="
  CFG_MAG=$(make_tmp_cfg "magnitude")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_MAG"
  rm -f "$CFG_MAG"
  echo

  echo "==[3/6] Run FIM (Fisher) pruning =="
  CFG_FIM=$(make_tmp_cfg "fim")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_FIM"
  rm -f "$CFG_FIM"
  echo

  echo "==[4/6] Run F_DIST_ONE_SHOT (FIM x Magnitude One-Shot) pruning =="
  CFG_FDIST_OS=$(make_tmp_cfg "f_dist_one_shot")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_FDIST_OS"
  rm -f "$CFG_FDIST_OS"
  echo

  echo "==[5/6] Run F_DIST_ITERATIVE (FIM x Magnitude Iterative) pruning =="
  CFG_FDIST_IT=$(make_tmp_cfg "f_dist_iterative")
  $PYTHON -m "$PRUNE_MODULE" "$CFG_FDIST_IT"
  rm -f "$CFG_FDIST_IT"
  echo

  # echo "==[6/6] Run F_DIST (Fisher-distance) pruning =="
  # CFG_FDIST=$(make_tmp_cfg "f_dist")
  # $PYTHON -m "$PRUNE_MODULE" "$CFG_FDIST"
  # rm -f "$CFG_FDIST"
  # echo

  # echo "Fold ${fold}/${NUM_FOLDS} complete."
  # echo
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

