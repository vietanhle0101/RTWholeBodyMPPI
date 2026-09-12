#!/usr/bin/env bash
# 20-goal comparison: current best MINLP setup (multirate_2s config, with
# the yaw slew-rate limiter and cross-replan face-persistence cost) vs. true
# no-MINLP baseline, same box-start/goal pairs for both. Reports success and
# (via summarize.py) time-to-reach for each. Pass "minlp" or "baseline" as
# $1 to run just one side; with no arg, runs both.
#
# Goals: 5 fixed + 10 random goals from earlier sweeps, plus 5 new random
# ones (rand_10..rand_14) to bring the set to 20.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root
REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
CFGDIR="$REPO_ROOT/legged_mppi/whole_body_mppi/control/contact_scheduler/configs"
DURATION=30.0
MAX_CONCURRENT=5
PER_RUN_TIMEOUT=2700

MODE="${1:-both}"

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
  "rand_10:0.620:-0.022:1.122:0.016"
  "rand_11:0.652:-0.240:1.099:1.360"
  "rand_12:0.724:-0.029:1.640:-0.716"
  "rand_13:0.625:-0.239:1.053:0.842"
  "rand_14:0.753:-0.277:1.330:-1.242"
)

run_one() {
  # Single positional arg: "kind|cfg_name|goal_name|bx|by|gx|gy"
  local IFS='|'
  read -r kind cfg_name goal_name bx by gx gy <<< "$1"
  local logdir="$REPO_ROOT/legged_mppi/experiments/dt_horizon_sweep/logs_20goals_${kind}"
  mkdir -p "$logdir"
  local log="$logdir/${cfg_name}__${goal_name}.log"
  if [ -f "$log" ] && grep -q "^RUN_DONE" "$log" 2>/dev/null; then
    echo "[skip, already done] $kind / $goal_name"
    return
  fi
  echo "[start] $kind / $goal_name -> $log"
  {
    echo "config=$cfg_name goal=$goal_name box=($bx,$by) target=($gx,$gy)"
    date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ"
  } > "$log"
  if [ "$kind" == "minlp" ]; then
    timeout "$PER_RUN_TIMEOUT" "$PY" -u "$REPO_ROOT/legged_mppi/scripts/simulate_minlp_box_push.py" \
      --duration "$DURATION" --box-x "$bx" --box-y "$by" --goal-x "$gx" --goal-y "$gy" \
      --scheduler-config "$CFGDIR/push_box_minlp_multirate_2s.yml" \
      >> "$log" 2>&1
  else
    timeout "$PER_RUN_TIMEOUT" "$PY" -u "$REPO_ROOT/legged_mppi/scripts/simulate_baseline_box_push.py" \
      --duration "$DURATION" --box-x "$bx" --box-y "$by" --goal-x "$gx" --goal-y "$gy" \
      >> "$log" 2>&1
  fi
  local rc=$?
  { date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ"; echo "exit_code=$rc"; echo "RUN_DONE"; } >> "$log"
  echo "[done, rc=$rc] $kind / $goal_name"
}
export -f run_one
export REPO_ROOT PY CFGDIR DURATION PER_RUN_TIMEOUT

tasks=()
for g in "${goals[@]}"; do
  IFS=':' read -r goal_name bx by gx gy <<< "$g"
  if [ "$MODE" == "minlp" ] || [ "$MODE" == "both" ]; then
    tasks+=("minlp|multirate_2s|${goal_name}|${bx}|${by}|${gx}|${gy}")
  fi
  if [ "$MODE" == "baseline" ] || [ "$MODE" == "both" ]; then
    tasks+=("baseline|baseline|${goal_name}|${bx}|${by}|${gx}|${gy}")
  fi
done

echo "Launching ${#tasks[@]} tasks, $MAX_CONCURRENT at a time."
printf '%s\n' "${tasks[@]}" | xargs -P "$MAX_CONCURRENT" -I {} bash -c 'run_one "$@"' _ {}

echo "SWEEP_ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
