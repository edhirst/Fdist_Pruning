#!/usr/bin/env bash
# Pre-stage CIFAR-10 ONCE, before submitting the two jobs.
#
# Why: hpc/nn_cifar10.pbs and hpc/vit_cifar10.pbs both read data/CIFAR10. If the
# dataset is absent when both start, they would race to download+extract into the
# same directory and corrupt it. Staging it once here removes that race (and
# CENAPAD compute nodes usually have no internet anyway — run this on the login
# node, which does).
#
# Usage (from the repository root, on a node with internet access):
#   bash hpc/prep_data.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python}"
mkdir -p data models results logs

"$PYTHON" - <<'PY'
from torchvision import datasets
datasets.CIFAR10(root="data/CIFAR10", train=True, download=True)
datasets.CIFAR10(root="data/CIFAR10", train=False, download=True)
print("CIFAR-10 staged under data/CIFAR10/")
PY

echo "Prepared: data/CIFAR10 + logs/ models/ results/ directories exist."
echo "Now submit:  qsub hpc/nn_cifar10.pbs  &&  qsub hpc/vit_cifar10.pbs"
