#!/usr/bin/env bash
# Pre-stage every dataset the grid needs, ONCE, before submitting any jobs.
#
# Why: hpc/submit_all.sh launches jobs for all four configs at the same time, and
# they share the dataset root. If a dataset is absent when several start, they
# race to download and extract into the same directory and corrupt it. Staging
# here removes that race -- and CENAPAD compute nodes generally have no internet,
# so the download has to happen on the login node in any case.
#
# The roots come from paths.dataset_path in the configs themselves, so changing
# that setting cannot leave staging pointed at the wrong directory.
#
# Usage (from the repository root, on a node with internet access):
#   bash hpc/prep_data.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python}"
mkdir -p models results outputs

"$PYTHON" - <<'PY'
import glob
import os

import yaml
from torchvision import datasets

# Every root and log directory any config asks for, deduplicated.
roots, logs = set(), set()
for path in ["src/config.yaml"] + sorted(glob.glob("configs/*.yaml")):
    with open(path) as fh:
        paths_cfg = (yaml.safe_load(fh).get("paths", {}) or {})
    roots.add(str(paths_cfg.get("dataset_path") or "data"))
    if paths_cfg.get("log_path"):
        logs.add(str(paths_cfg["log_path"]))

for d in sorted(logs):
    os.makedirs(d, exist_ok=True)

for root in sorted(roots):
    for name, ctor, sub in (
        ("MNIST", datasets.MNIST, "MNIST"),
        ("CIFAR-10", datasets.CIFAR10, "CIFAR10"),
    ):
        target = os.path.join(root, sub)
        for train in (True, False):
            ctor(root=target, train=train, download=True)
        print(f"{name} staged under {target}/")

print()
print("dataset roots:", ", ".join(sorted(roots)))
print("log dirs:     ", ", ".join(sorted(logs)) or "(none configured)")
PY

echo
echo "Also created: models/ results/ outputs/"
echo "Next:"
echo "  qsub hpc/smoke_testegpu.pbs     # 30-min sanity check, look for 'SMOKE OK'"
echo "  bash hpc/submit_all.sh          # dry run: prints every qsub it would issue"
echo "  bash hpc/submit_all.sh --go     # submit the grid"
