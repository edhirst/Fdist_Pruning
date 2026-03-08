#!/usr/bin/env bash
set -euo pipefail

# 固定在 repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python3}
CONFIG="src/config.yaml"

echo "Running from: $ROOT_DIR"
echo

# 讀 checkpoint 路徑
CKPT_PATH=$($PYTHON - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG}", "r"))
print(cfg["paths"]["pretrained_model_path"])
PY
)

echo "Checkpoint path: $CKPT_PATH"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found."
  echo "Expected file: $CKPT_PATH"
  echo "請確認 models/ 下面真的有這個檔案"
  exit 1
fi

echo "Checkpoint found. Skipping training."
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

echo "== Pruning: magnitude =="
CFG_MAG="$(make_prune_cfg magnitude)"
$PYTHON -m src.run_pruning "$CFG_MAG"
rm -f "$CFG_MAG"
echo

echo "== Pruning: fim =="
CFG_FIM="$(make_prune_cfg fim)"
$PYTHON -m src.run_pruning "$CFG_FIM"
rm -f "$CFG_FIM"
echo

echo "== Pruning: f_dist_one_shot =="
CFG_FDOS="$(make_prune_cfg f_dist_one_shot)"
$PYTHON -m src.run_pruning "$CFG_FDOS"
rm -f "$CFG_FDOS"
echo

echo "== Pruning: f_dist_iterative =="
CFG_FDI="$(make_prune_cfg f_dist_iterative)"
$PYTHON -m src.run_pruning "$CFG_FDI"
rm -f "$CFG_FDI"
echo

# echo "== Pruning: fdist =="
# CFG_FD="$(make_prune_cfg f_dist)"
# $PYTHON -m src.run_pruning "$CFG_FD"
# rm -f "$CFG_FD"
# echo



echo "Done."
