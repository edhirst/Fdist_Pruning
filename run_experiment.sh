#!/usr/bin/env bash
set -euo pipefail


PYTHON=${PYTHON:-python3}
BASE_CONFIG=${BASE_CONFIG:-src/config.yaml}

TRAIN_MODULE=${TRAIN_MODULE:-src.train_model}
PRUNE_MODULE=${PRUNE_MODULE:-src.run_pruning}

# Set PARALLEL_FOLDS=true to run folds in parallel (uses more RAM)
# Set PARALLEL_FOLDS=false or unset to run sequentially (default)
PARALLEL_FOLDS=${PARALLEL_FOLDS:-false}

# SCHEMES: which pruning schemes to run, space separated. Default is every cheap
# scheme; exact "f_dist" is appended only when EXACT_FDIST=auto resolves to yes
# (see below) or when you name it explicitly. Splitting the schemes across jobs
# is what lets the cheap sweeps run on a GPU queue while exact f_dist runs on a
# big CPU node.
# NOTE: "-" not ":-" -- an explicitly EMPTY SCHEMES must mean "no cheap schemes"
# (the par128 exact-f_dist job sets SCHEMES=""). With ":-" bash would substitute
# the default for the empty string and the job would rerun every cheap scheme,
# silently duplicating results at the same seed.
SCHEMES=${SCHEMES-"magnitude fim f_dist_one_shot f_dist_iterative f_dist_global"}

# EXACT_FDIST: yes | no | auto. "auto" appends exact f_dist only for the small
# NN on 28x28 data, which was the only place it used to be affordable.
EXACT_FDIST=${EXACT_FDIST:-auto}

# SKIP_TRAIN=true reuses an existing checkpoint (so the exact-f_dist job does not
# retrain the model the cheap-scheme job already trained for this same seed).
SKIP_TRAIN=${SKIP_TRAIN:-false}

# Seed/K tag for log file names, so concurrent runs never share a log file.
RUN_TAG=""
[[ -n "${FDIST_SEED:-}" ]] && RUN_TAG="_seed${FDIST_SEED}"
[[ -n "${FDIST_K:-}" ]] && RUN_TAG="${RUN_TAG}_K${FDIST_K}"

# ========================

# Read the settings this script needs out of the active config in one go
# (BASE_CONFIG, default src/config.yaml).
CFG_INFO=$($PYTHON - "$BASE_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r"))
folds = (cfg.get("cross_validation", {}) or {}).get("num_folds", 1)
# paths.log_path: where per-stage logs are tee'd. null/"" turns logging off.
log = (cfg.get("paths", {}) or {}).get("log_path") or "-"
print(folds, log)
PY
)
NUM_FOLDS=${CFG_INFO%% *}
LOG_DIR=${CFG_INFO#* }
[[ "$LOG_DIR" == "-" ]] && LOG_DIR=""
# FDIST_LOG_DIR overrides the config, e.g. to send logs to node-local scratch.
LOG_DIR=${FDIST_LOG_DIR:-$LOG_DIR}
CFG_TAG=$(basename "$BASE_CONFIG" .yaml)

# Run one pipeline stage, tee'ing its output into paths.log_path so a local run
# leaves the same durable record the PBS jobs get from their queue output. With
# no log_path configured the command just writes to stdout, as it did before.
# `set -o pipefail` is in effect, so a failing python still fails the pipeline.
run_stage() {
  local label="$1"; shift
  if [[ -z "$LOG_DIR" ]]; then
    "$@"
    return
  fi
  mkdir -p "$LOG_DIR"
  local log_file="${LOG_DIR}/${CFG_TAG}_${label}_$(date +%Y%m%d_%H%M%S).log"
  echo "  log: ${log_file}"
  "$@" 2>&1 | tee "$log_file"
}

echo "Cross-validation folds: ${NUM_FOLDS}"
[[ -n "$LOG_DIR" ]] && echo "Stage logs: ${LOG_DIR}/"
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

  if [[ "$SKIP_TRAIN" == "true" ]]; then
    echo "==[train] SKIP_TRAIN=true -> reusing existing checkpoint =="
  else
    echo "==[train] Train model with config: ${BASE_CONFIG} =="
    run_stage "train_fold${fold}${RUN_TAG}" $PYTHON -m "$TRAIN_MODULE" "$BASE_CONFIG"
  fi


  # Model type, canonical dataset, and the checkpoint path resolved the same
  # way run_pruning.py does
  read -r MODEL_TYPE DATASET CKPT_PATH <<< "$($PYTHON - "$BASE_CONFIG" <<'PY'
import sys, yaml
from src.utils.data_loader import normalize_dataset_name
from src.utils.model_builder import build_model_from_config, resolve_checkpoint_path
from src.utils.seeding import resolve_seed
cfg = yaml.safe_load(open(sys.argv[1], "r"))
mt = str(((cfg.get("model", {}) or {}).get("common", {}) or {}).get("model_type", "NN")).strip().lower()
ds = normalize_dataset_name((cfg.get("dataset", {}) or {}).get("name", "mnist"))
_, arch = build_model_from_config(cfg, dataset_name=ds)
print(mt, ds, resolve_checkpoint_path(cfg, arch, ds, seed=resolve_seed(cfg)))
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

  # Resolve the scheme list for this job
  RUN_SCHEMES="$SCHEMES"
  case "$EXACT_FDIST" in
    yes) RUN_SCHEMES="$RUN_SCHEMES f_dist" ;;
    no)  ;;
    auto)
      if [[ "$MODEL_TYPE" == "nn" && ( "$DATASET" == "mnist" || "$DATASET" == "fashion_mnist" ) ]]; then
        RUN_SCHEMES="$RUN_SCHEMES f_dist"
      else
        echo "EXACT_FDIST=auto -> skipping exact f_dist (model_type=${MODEL_TYPE}, dataset=${DATASET})."
        echo "  Set EXACT_FDIST=yes to run it anyway (needs the forward-mode fast path; see hpc/README.md)."
      fi ;;
    *) echo "ERROR: EXACT_FDIST must be yes|no|auto, got '${EXACT_FDIST}'"; exit 1 ;;
  esac

  echo "Schemes to run: ${RUN_SCHEMES}"
  echo

  i=0
  total=$(wc -w <<< "$RUN_SCHEMES" | tr -d " ")
  for scheme in $RUN_SCHEMES; do
    i=$((i + 1))
    echo "==[${i}/${total}] Run ${scheme} pruning =="
    CFG_TMP=$(make_tmp_cfg "$scheme")
    run_stage "${scheme}_fold${fold}${RUN_TAG}" $PYTHON -m "$PRUNE_MODULE" "$CFG_TMP"
    rm -f "$CFG_TMP"
    echo
  done

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

