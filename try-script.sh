#!/usr/bin/env bash
# Convenience entry point for a single local experiment:
#
#     ./try-script.sh [config.yaml]        # defaults to src/config.yaml
#
# Trains the configured model only if its checkpoint is missing, then runs every
# pruning scheme the config supports. This is a thin wrapper around
# run_experiment.sh -- which is also what the HPC job scripts call -- so a local
# run and a cluster run execute exactly the same pipeline. Anything more
# specific (choosing schemes, seeds, K, worker counts) is done with the
# environment variables documented at the top of run_experiment.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python3}
CONFIG="${1:-src/config.yaml}"

echo "Running from: $ROOT_DIR"
echo "Config: $CONFIG"
echo

# Resolve the checkpoint exactly as run_pruning.py does (explicit
# paths.pretrained_model_path, else models/{arch}_{dataset}[_seed{N}].pth) so we
# can skip training when it already exists.
CKPT_PATH="$($PYTHON - "$CONFIG" <<'PY'
import sys, yaml
from src.utils.data_loader import normalize_dataset_name
from src.utils.model_builder import build_model_from_config, resolve_checkpoint_path
from src.utils.seeding import resolve_seed
cfg = yaml.safe_load(open(sys.argv[1], "r"))
ds = normalize_dataset_name((cfg.get("dataset", {}) or {}).get("name", "mnist"))
_, arch = build_model_from_config(cfg, dataset_name=ds)
print(resolve_checkpoint_path(cfg, arch, ds, seed=resolve_seed(cfg)))
PY
)"

if [[ -f "$CKPT_PATH" ]]; then
  echo "Checkpoint found, skipping training: $CKPT_PATH"
  export SKIP_TRAIN=true
else
  echo "Checkpoint not found, training first: $CKPT_PATH"
  export SKIP_TRAIN=false
fi
echo

export BASE_CONFIG="$CONFIG"
exec ./run_experiment.sh
