#!/usr/bin/env bash
set -euo pipefail

# ====== 可自行調整 ======
PYTHON=${PYTHON:-python3}
BASE_CONFIG=${BASE_CONFIG:-src/config.yaml}

TRAIN_MODULE=${TRAIN_MODULE:-src.train_model}
PRUNE_MODULE=${PRUNE_MODULE:-src.run_pruning}

# 如果你想讓每次訓練的模型檔名固定好用 config 指定的 pretrained_model_path，
# 那你就讓 config.paths.pretrained_model_path 指向你期待的輸出檔名即可。
# 以你給的 config: models/SimpleNN_h2_n64.pth
# ========================

echo "==[1/3] Train model with config: ${BASE_CONFIG} =="
$PYTHON -m "$TRAIN_MODULE" "$BASE_CONFIG"

# 讀出訓練後要拿來 pruning 的 checkpoint path（沿用 config.paths.pretrained_model_path）
CKPT_PATH=$($PYTHON - <<'PY'
import yaml
cfg = yaml.safe_load(open("src/config.yaml","r"))
print(cfg["paths"]["pretrained_model_path"])
PY
)

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH"
  echo "你可能需要確認 train_model 存檔檔名是否跟 config.paths.pretrained_model_path 一致。"
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

# 強制確保 pruning enabled
cfg.setdefault("pruning", {})
cfg["pruning"]["enable_pruning"] = True
cfg["pruning"]["pruning_scheme"] = "${scheme}"

# 確保 pretrained_model_path 指向剛訓練完的 ckpt（避免你之後改了原 config）
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

