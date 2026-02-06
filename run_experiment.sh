#!/usr/bin/env bash
set -euo pipefail


PYTHON=${PYTHON:-python3}
BASE_CONFIG=${BASE_CONFIG:-src/config.yaml}

TRAIN_MODULE=${TRAIN_MODULE:-src.train_model}
PRUNE_MODULE=${PRUNE_MODULE:-src.run_pruning}

# ========================

echo "==[1/3] Train model with config: ${BASE_CONFIG} =="
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

with open("${out_cfg}","w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "$out_cfg"
}

echo "==[2/3] Run MAGNITUDE pruning =="
CFG_MAG=$(make_tmp_cfg "magnitude")
$PYTHON -m "$PRUNE_MODULE" "$CFG_MAG"
rm -f "$CFG_MAG"
echo

echo "==[3/3] Run FIM (Fisher) pruning =="
CFG_FIM=$(make_tmp_cfg "fim")
$PYTHON -m "$PRUNE_MODULE" "$CFG_FIM"
rm -f "$CFG_FIM"
echo

echo "All done."
echo "Results are under ./results/ (each run creates a timestamped folder)."

