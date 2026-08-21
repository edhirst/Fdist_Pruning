#!/usr/bin/env bash
# Submit the full multi-seed grid for the paper table.
#
#   bash hpc/submit_all.sh            # dry run: print the qsub commands only
#   bash hpc/submit_all.sh --go       # actually submit
#
# Phase 1 (GPU, umagpu) trains one checkpoint per (config, seed) and runs the
# five cheap schemes. Phase 2 (CPU, par128) runs exact f_dist on the SAME
# checkpoint, so it must not start until phase 1 for that pair has finished --
# the dependency is set with -W depend=afterok.
#
# The cluster caps concurrent jobs per user per queue (umagpu 2, par128 6); PBS
# simply queues the rest, so submitting the whole grid at once is fine.
#
# Phase 3 (optional, K_SWEEP=1) reruns exact f_dist on SimpleNN/MNIST over a range
# of f_dist_avg_points K, for the "how do accuracy and cost scale with K" table.
#
# K in {2,5,9,17} plus the K=3 that phase 2 already produced gives the full ladder
# {2,3,5,9,17}: the alpha grids linspace(1,0,K) are then {1,0} < {1,.5,0} <
# quarters < eighths < sixteenths, each a strict superset of the last, and the
# cost -- (K-1) probes per coordinate -- doubles exactly 1,2,4,8,16 with no gaps.
# K=3 is deliberately NOT repeated here: phase 2 runs it on the same config, the
# same seeds and the same node type, so it is directly comparable and rerunning
# it would only duplicate work.
# Add nn_cifar10 with K_CONFIGS="configs/nn_mnist.yaml configs/nn_cifar10.yaml".
set -euo pipefail

SEEDS=${SEEDS:-"0 1 2 3 4"}
CONFIGS=${CONFIGS:-"configs/nn_mnist.yaml configs/nn_cifar10.yaml configs/vit_mnist.yaml configs/vit_cifar10.yaml"}

# K sweep: which configs and which K values (the main table's K lives in the
# configs and is submitted by phase 2, so it is not repeated here).
K_SWEEP=${K_SWEEP:-1}
K_CONFIGS=${K_CONFIGS:-"configs/nn_mnist.yaml"}
K_VALUES=${K_VALUES:-"2 5 9 17"}

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

run() {
  if [[ $GO -eq 1 ]]; then
    eval "$@"
  else
    echo "$@"
  fi
}

n_cheap=0
n_exact=0
n_k=0
for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    # compact code per config so every job name stays <= 15 chars
    case "$(basename "$cfg" .yaml)" in
      nn_mnist)    code=nm ;;
      nn_cifar10)  code=nc ;;
      vit_mnist)   code=vm ;;
      vit_cifar10) code=vc ;;
      *)           code="$(basename "$cfg" .yaml | tr -cd 'a-z0-9' | cut -c1-4)" ;;
    esac
    tag="${code}_s${seed}"

    # ---- phase 1: train + cheap schemes, on the GPU queue ----
    cheap_cmd="qsub -N ch_${tag} -v CFG=${cfg},SEED=${seed} hpc/cheap_gpu.pbs"
    if [[ $GO -eq 1 ]]; then
      out=$(eval "$cheap_cmd")
      echo "submitted ch_${tag}: $out"
    else
      echo "$cheap_cmd"
      out="<jobid>"
    fi
    n_cheap=$((n_cheap + 1))

    # ---- phase 2: exact f_dist, on the big CPU node, after phase 1 ----
    run "qsub -N ex_${tag} -W depend=afterok:${out} -v CFG=${cfg},SEED=${seed} hpc/exact_fdist_par128.pbs"
    n_exact=$((n_exact + 1))

    # ---- phase 3: K sweep, same node type, also after phase 1 ----
    if [[ "$K_SWEEP" == "1" && " $K_CONFIGS " == *" $cfg "* ]]; then
      for k in $K_VALUES; do
        run "qsub -N k${k}_${tag} -W depend=afterok:${out} -v CFG=${cfg},SEED=${seed},K=${k} hpc/exact_fdist_par128.pbs"
        n_k=$((n_k + 1))
      done
    fi
  done
done

echo
echo "cheap (umagpu): ${n_cheap}    exact f_dist (par128): ${n_exact}    K-sweep (par128): ${n_k}"
echo "total jobs: $((n_cheap + n_exact + n_k))"
if [[ $GO -eq 0 ]]; then
  echo "(dry run -- rerun with --go to submit)"
fi
