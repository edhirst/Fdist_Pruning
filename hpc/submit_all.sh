#!/usr/bin/env bash
# Feed the full multi-seed grid into the queues, respecting the per-user caps.
#
#   bash hpc/submit_all.sh              # dry run: list every job and its state
#   bash hpc/submit_all.sh --go         # submit as many as the caps allow, then stop
#   bash hpc/submit_all.sh --watch      # keep submitting as slots free (run under nohup)
#   bash hpc/submit_all.sh --status     # what is submitted, running, still pending
#
# IMPORTANT: CENAPAD counts QUEUED jobs against the per-user limit, not just
# running ones -- qsub REJECTS with "would exceed queue <q>'s per-user limit"
# rather than queueing. The grid is 60 jobs against 2 umagpu + 6 par128 slots, so
# it has to be dripped in. That is what --watch does.
#
# Progress is recorded in hpc/.submit_state (one "<jobname> <jobid>" per line) and
# re-synced from qstat on every pass, so the script is safe to re-run, safe to
# interrupt, and picks up jobs submitted by an earlier invocation.
#
# Phase 1 (GPU, umagpu) trains one checkpoint per (config, seed) and runs the five
# cheap schemes. Phase 2 (CPU, par128) runs exact f_dist on that SAME checkpoint,
# so it is chained with -W depend=afterok. Phase 3 (optional, K_SWEEP=1) reruns
# exact f_dist on SimpleNN/MNIST over a range of f_dist_avg_points K.
#
# K in {2,5,9,17} plus the K=3 phase 2 already produces gives the ladder
# {2,3,5,9,17}: the alpha grids linspace(1,0,K) are then {1,0} < {1,.5,0} <
# quarters < eighths < sixteenths, each a strict superset of the last, and the
# cost -- (K-1) probes per coordinate -- doubles exactly 1,2,4,8,16 with no gaps.
# K=3 is deliberately NOT repeated: phase 2 runs it on the same config, seeds and
# node type, so it is directly comparable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=${SEEDS:-"0 1 2 3 4"}
CONFIGS=${CONFIGS:-"configs/nn_mnist.yaml configs/nn_cifar10.yaml configs/vit_mnist.yaml configs/vit_cifar10.yaml"}

K_SWEEP=${K_SWEEP:-1}
K_CONFIGS=${K_CONFIGS:-"configs/nn_mnist.yaml"}
K_VALUES=${K_VALUES:-"2 5 9 17"}

# Per-user concurrent-job caps, counting queued + held + running.
CAP_UMAGPU=${CAP_UMAGPU:-2}
CAP_PAR128=${CAP_PAR128:-6}

STATE=${STATE:-hpc/.submit_state}
INTERVAL=${INTERVAL:-300}          # --watch poll period, seconds

MODE=dry
for arg in "$@"; do
  case "$arg" in
    --go)     MODE=go ;;
    --watch)  MODE=watch ;;
    --status) MODE=status ;;
    *) echo "unknown argument: $arg" >&2
       echo "usage: $0 [--go|--watch|--status]" >&2; exit 2 ;;
  esac
done

touch "$STATE"

# ---- the full job list: tag|queue|script|qsub vars|parent tag ----------------
config_code() {
  case "$(basename "$1" .yaml)" in
    nn_mnist)    echo nm ;;
    nn_cifar10)  echo nc ;;
    vit_mnist)   echo vm ;;
    vit_cifar10) echo vc ;;
    *) basename "$1" .yaml | tr -cd 'a-z0-9' | cut -c1-4 ;;
  esac
}

build_jobs() {
  local cfg seed code tag k
  for cfg in $CONFIGS; do
    code=$(config_code "$cfg")
    for seed in $SEEDS; do
      tag="${code}_s${seed}"
      echo "ch_${tag}|umagpu|hpc/cheap_gpu.pbs|CFG=${cfg},SEED=${seed}|"
      echo "ex_${tag}|par128|hpc/exact_fdist_par128.pbs|CFG=${cfg},SEED=${seed}|ch_${tag}"
      if [[ "$K_SWEEP" == "1" && " $K_CONFIGS " == *" $cfg "* ]]; then
        for k in $K_VALUES; do
          echo "k${k}_${tag}|par128|hpc/exact_fdist_par128.pbs|CFG=${cfg},SEED=${seed},K=${k}|ch_${tag}"
        done
      fi
    done
  done
}

# ---- what PBS currently knows -----------------------------------------------
# qstat rows look like: <jobid> <user> <queue> <jobname> ... <state> <time>
# Job names are kept to <= 9 chars precisely so they are never truncated here.
qstat_rows() { qstat -u "$USER" 2>/dev/null | awk '$1 ~ /^[0-9]+\./ {print}'; }

sync_state() {
  local id name
  while read -r id _ _ name _; do
    [[ -z "${name:-}" ]] && continue
    grep -q "^${name} " "$STATE" 2>/dev/null || echo "${name} ${id}" >> "$STATE"
  done < <(qstat_rows)
}

queue_used() { qstat_rows | awk -v q="$1" '$3 == q' | wc -l | tr -d ' '; }
state_id()   { grep -m1 "^$1 " "$STATE" 2>/dev/null | awk '{print $2}'; }
submitted()  { [[ -n "$(state_id "$1")" ]]; }
in_queue()   { qstat_rows | awk -v i="$1" '$1 == i' | grep -q .; }

# A dependency on a job PBS has already purged can leave the child held forever,
# so only chain while the parent is still in the system. If it has finished, the
# checkpoint it produced is already on disk and the child can run unconditionally.
dep_flag() {
  local parent="$1" pid
  [[ -z "$parent" ]] && { echo ""; return; }
  pid=$(state_id "$parent")
  [[ -z "$pid" ]] && { echo "PENDING"; return; }
  if in_queue "$pid"; then echo "-W depend=afterok:${pid}"; else echo ""; fi
}

# ---- one submission pass -----------------------------------------------------
# Returns 0 if every job is submitted, 1 if work remains.
pass() {
  local tag queue script vars parent dep free_u free_p n_done=0 n_left=0 n_new=0
  sync_state
  free_u=$(( CAP_UMAGPU - $(queue_used umagpu) ))
  free_p=$(( CAP_PAR128 - $(queue_used par128) ))

  while IFS='|' read -r tag queue script vars parent; do
    if submitted "$tag"; then n_done=$((n_done + 1)); continue; fi
    n_left=$((n_left + 1))

    dep=$(dep_flag "$parent")
    [[ "$dep" == "PENDING" ]] && continue          # parent not submitted yet

    case "$queue" in
      umagpu)  (( free_u > 0 )) || continue ;;
      par128)  (( free_p > 0 )) || continue ;;
    esac

    local cmd="qsub -N ${tag} ${dep} -v ${vars} ${script}"
    if [[ "$MODE" == "dry" ]]; then
      echo "  would submit: $cmd"
    else
      local out
      if out=$(eval "$cmd" 2>&1); then
        echo "${tag} ${out}" >> "$STATE"
        echo "  submitted ${tag}: ${out}"
        n_new=$((n_new + 1))
        case "$queue" in umagpu) free_u=$((free_u - 1)) ;; par128) free_p=$((free_p - 1)) ;; esac
      else
        # Hitting the cap here is expected and not an error: stop filling that
        # queue this pass and try again next time round.
        echo "  deferred ${tag}: ${out}"
        case "$queue" in umagpu) free_u=0 ;; par128) free_p=0 ;; esac
      fi
    fi
  done < <(build_jobs)

  echo "  [submitted ${n_done}, pending ${n_left}, new this pass ${n_new}; " \
       "free slots: umagpu ${free_u}, par128 ${free_p}]"
  (( n_left - n_new <= 0 ))
}

total_jobs=$(build_jobs | wc -l | tr -d ' ')

case "$MODE" in
  status)
    sync_state
    echo "grid: ${total_jobs} jobs   state file: ${STATE}"
    printf '%-12s %-12s %s\n' TAG JOBID STATE
    while IFS='|' read -r tag _ _ _ _; do
      id=$(state_id "$tag")
      if [[ -z "$id" ]]; then printf '%-12s %-12s %s\n' "$tag" "-" "not submitted"
      elif in_queue "$id"; then
        printf '%-12s %-12s %s\n' "$tag" "$id" "$(qstat_rows | awk -v i="$id" '$1==i {print $(NF-1)}')"
      else printf '%-12s %-12s %s\n' "$tag" "$id" "finished/left queue"; fi
    done < <(build_jobs)
    ;;
  dry)
    echo "grid: ${total_jobs} jobs (caps: umagpu ${CAP_UMAGPU}, par128 ${CAP_PAR128})"
    pass || true
    echo
    echo "(dry run -- '--go' submits one pass, '--watch' keeps feeding until done)"
    ;;
  go)
    echo "grid: ${total_jobs} jobs"
    pass || true
    echo
    echo "Re-run with --go as slots free, or use --watch to do it automatically."
    ;;
  watch)
    echo "grid: ${total_jobs} jobs; feeding every ${INTERVAL}s until all are submitted"
    while :; do
      echo "== $(date '+%Y-%m-%d %H:%M:%S') =="
      if pass; then
        echo "All ${total_jobs} jobs submitted. Watch loop done."
        break
      fi
      sleep "$INTERVAL"
    done
    ;;
esac
