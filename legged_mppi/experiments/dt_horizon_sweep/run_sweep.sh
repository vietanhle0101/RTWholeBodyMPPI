#!/usr/bin/env bash
# dt/horizon sweep for the MINLP contact scheduler: does a longer lookahead
# (and/or coarser per-step discretization) improve box-push task success,
# beyond what the base 1.0s-lookahead config gets?
#
# Runs every (config, goal) pair below with simulate_minlp_box_push.py using
# an xargs -P worker pool (NOT shell job control -- `jobs`/`wait -n` silently
# fail to throttle anything when this script runs non-interactively/piped,
# which is exactly what happened the first time this was tried: it launched
# all 55 runs at once instead of 5 at a time), plus a hard per-run wall-clock
# timeout on top of the scheduler's own internal BONMIN watchdog.
#
# Logs go to ./logs; run summarize.py afterward (or anytime, on partial
# results) to get a table.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root
REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
SCRIPT="$REPO_ROOT/legged_mppi/scripts/simulate_minlp_box_push.py"
LOGDIR="$REPO_ROOT/legged_mppi/experiments/dt_horizon_sweep/logs"
CFGDIR="$REPO_ROOT/legged_mppi/whole_body_mppi/control/contact_scheduler/configs"
DURATION=30.0
MAX_CONCURRENT=5
PER_RUN_TIMEOUT=2700   # 45 min hard cap per run (well above any expected solve time)

mkdir -p "$LOGDIR"

# name:config_file
configs=(
  "1s_base:push_box_minlp.yml"
  "2s_coarse:push_box_minlp_2s_coarse.yml"
  "2s_fine:push_box_minlp_2s_fine.yml"
  "3s_coarse:push_box_minlp_long_horizon.yml"
  "3s_fine:push_box_minlp_3s_fine.yml"
)

# name:box_x:box_y:goal_x:goal_y
goals=(
  "fixed_forward:0.6:0.0:1.6:0.0"
  "fixed_lateral:0.6:0.0:0.6:1.0"
  "fixed_diag45:0.6:0.0:1.6:1.0"
  "fixed_shallow:0.6:0.0:1.9:0.5"
  "fixed_steep:0.6:0.0:1.0:1.2"
  "rand_0:0.810:-0.037:1.542:1.107"
  "rand_1:0.538:0.285:1.603:1.237"
  "rand_2:0.551:-0.030:1.993:-0.574"
  "rand_3:0.758:0.194:1.727:0.039"
  "rand_4:0.722:-0.262:1.518:0.773"
  "rand_5:0.803:-0.087:1.187:1.378"
)

run_one() {
  # Single positional arg: "cfg_name|cfg_file|goal_name|bx|by|gx|gy"
  local IFS='|'
  read -r cfg_name cfg_file goal_name bx by gx gy <<< "$1"
  local log="$LOGDIR/${cfg_name}__${goal_name}.log"
  if [ -f "$log" ] && grep -q "^RUN_DONE" "$log" 2>/dev/null; then
    echo "[skip, already done] $cfg_name / $goal_name"
    return
  fi
  echo "[start] $cfg_name / $goal_name -> $log"
  {
    echo "config=$cfg_name ($cfg_file) goal=$goal_name box=($bx,$by) target=($gx,$gy)"
    date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ"
  } > "$log"
  timeout "$PER_RUN_TIMEOUT" "$PY" -u "$SCRIPT" \
    --duration "$DURATION" \
    --box-x "$bx" --box-y "$by" \
    --goal-x "$gx" --goal-y "$gy" \
    --scheduler-config "$CFGDIR/$cfg_file" \
    >> "$log" 2>&1
  local rc=$?
  {
    date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ"
    echo "exit_code=$rc"
    echo "RUN_DONE"
  } >> "$log"
  echo "[done, rc=$rc] $cfg_name / $goal_name"
}
export -f run_one
export LOGDIR PY SCRIPT CFGDIR DURATION PER_RUN_TIMEOUT

tasks=()
for c in "${configs[@]}"; do
  IFS=':' read -r cfg_name cfg_file <<< "$c"
  for g in "${goals[@]}"; do
    IFS=':' read -r goal_name bx by gx gy <<< "$g"
    tasks+=("${cfg_name}|${cfg_file}|${goal_name}|${bx}|${by}|${gx}|${gy}")
  done
done

echo "Launching ${#tasks[@]} tasks, $MAX_CONCURRENT at a time."
printf '%s\n' "${tasks[@]}" | xargs -P "$MAX_CONCURRENT" -I {} bash -c 'run_one "$@"' _ {}

echo "SWEEP_ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
