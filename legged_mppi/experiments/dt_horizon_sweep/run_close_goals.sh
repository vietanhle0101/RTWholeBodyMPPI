#!/usr/bin/env bash
# Same 11 directions as the main sweep, but each goal moved to 0.5m from the
# box's start instead of the original 1.0-1.5m. Isolates "can this config
# solve this geometry at all" from "did it have enough time/distance budget"
# -- several main-sweep "not reached" results turned out to be steady
# progress that simply ran out of runway in 30s, not real failures.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root
REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
SCRIPT="$REPO_ROOT/legged_mppi/scripts/simulate_minlp_box_push.py"
LOGDIR="$REPO_ROOT/legged_mppi/experiments/dt_horizon_sweep/logs_close"
CFGDIR="$REPO_ROOT/legged_mppi/whole_body_mppi/control/contact_scheduler/configs"
DURATION=30.0
MAX_CONCURRENT=4
PER_RUN_TIMEOUT=2700

mkdir -p "$LOGDIR"

# name:config_file -- start with 1s_base only; extend to other configs later
# if this proves useful.
configs=(
  "1s_base:push_box_minlp.yml"
)

# name:box_x:box_y:goal_x:goal_y (same direction as main sweep, 0.5m out)
goals=(
  "fixed_forward:0.600:0.000:1.100:0.000"
  "fixed_lateral:0.600:0.000:0.600:0.500"
  "fixed_diag45:0.600:0.000:0.954:0.354"
  "fixed_shallow:0.600:0.000:1.067:0.179"
  "fixed_steep:0.600:0.000:0.758:0.474"
  "rand_0:0.810:-0.037:1.079:0.384"
  "rand_1:0.538:0.285:0.911:0.618"
  "rand_2:0.551:-0.030:1.019:-0.206"
  "rand_3:0.758:0.194:1.252:0.115"
  "rand_4:0.722:-0.262:1.027:0.134"
  "rand_5:0.803:-0.087:0.930:0.397"
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

echo "Launching ${#tasks[@]} close-goal tasks, $MAX_CONCURRENT at a time."
printf '%s\n' "${tasks[@]}" | xargs -P "$MAX_CONCURRENT" -I {} bash -c 'run_one "$@"' _ {}

echo "SWEEP_ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
