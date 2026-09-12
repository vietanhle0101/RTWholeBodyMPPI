#!/usr/bin/env bash
# Re-run the 15-goal comparison set (same goals as logs_baseline) against the
# MINLP scheduler on top of the current code state: velocity-frame fix +
# multi-rate discretization (multirate_2s config) + the yawfix hold-last-yaw
# reference-adapter fix, with the in-MINLP heading-state extension reverted
# out. Logs go to ./logs_yawfix; run summarize.py logs_yawfix afterward.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root
REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
SCRIPT="$REPO_ROOT/legged_mppi/scripts/simulate_minlp_box_push.py"
LOGDIR="$REPO_ROOT/legged_mppi/experiments/dt_horizon_sweep/logs_yawfix"
CFGDIR="$REPO_ROOT/legged_mppi/whole_body_mppi/control/contact_scheduler/configs"
DURATION=30.0
MAX_CONCURRENT=5
PER_RUN_TIMEOUT=2700

mkdir -p "$LOGDIR"

configs=(
  "multirate_2s:push_box_minlp_multirate_2s.yml"
)

# Same 15 goals used for logs_baseline, for an apples-to-apples comparison.
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
  "rand_6:0.761:-0.274:1.097:-1.706"
  "rand_7:0.735:-0.165:1.506:0.489"
  "rand_8:0.668:-0.029:1.114:1.417"
  "rand_9:0.611:-0.133:1.396:-0.306"
)

run_one() {
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
