#!/usr/bin/env python3
"""Summarize the dt/horizon sweep logs into a results table.

Usage: .venv/bin/python3 legged_mppi/experiments/dt_horizon_sweep/summarize.py
Safe to run any time, including while the sweep is still in progress -- it
just reports "running" for logs without a final position yet.
"""

import glob
import os
import re

import numpy as np

GOAL_THRESH = 0.2
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

FINAL_RE = re.compile(r"final box position:\s*\[\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\]")
GOAL_RE = re.compile(r"target=\(([-\d.]+),([-\d.]+)\)")
CONFIG_RE = re.compile(r"^config=(\S+)", re.MULTILINE)
DONE_RE = re.compile(r"^RUN_DONE", re.MULTILINE)
RC_RE = re.compile(r"^exit_code=(\d+)", re.MULTILINE)
# simulate_minlp_box_push.py's early-stop print, and
# simulate_baseline_box_push.py's final "reached goal: True at t=..." line --
# whichever the log actually has.
REACH_TIME_RE = re.compile(
    r"^t=([\d.]+)s: box reached goal|reached goal: True at t=([\d.]+)s", re.MULTILINE)


def parse_log(path):
    text = open(path).read()
    m = GOAL_RE.search(text)
    goal = (float(m.group(1)), float(m.group(2))) if m else None
    finals = FINAL_RE.findall(text)
    done = bool(DONE_RE.search(text))
    rc_match = RC_RE.search(text)
    rc = int(rc_match.group(1)) if rc_match else None
    reach_time = None
    m = REACH_TIME_RE.search(text)
    if m:
        reach_time = float(m.group(1) or m.group(2))
    if not finals or goal is None:
        return {"status": "running" if not done else "no_result", "rc": rc}
    fx, fy = float(finals[-1][0]), float(finals[-1][1])
    dist = float(np.hypot(fx - goal[0], fy - goal[1]))
    return {"status": "done" if done else "unclear", "final": (fx, fy),
            "goal": goal, "distance": dist, "reached": dist < GOAL_THRESH, "rc": rc,
            "reach_time": reach_time}


def main():
    import sys
    logdir = sys.argv[1] if len(sys.argv) > 1 else LOGDIR
    rows = []
    for path in sorted(glob.glob(os.path.join(logdir, "*.log"))):
        name = os.path.basename(path)[:-4]
        config, goal_name = name.split("__", 1)
        result = parse_log(path)
        result.update(config=config, goal=goal_name, name=name)
        rows.append(result)

    configs = sorted(set(r["config"] for r in rows))
    goals = sorted(set(r["goal"] for r in rows))

    print(f"{'config':<14}" + "".join(f"{g:<14}" for g in goals) + "reached/done")
    for cfg in configs:
        line = f"{cfg:<14}"
        reached = 0
        done = 0
        for g in goals:
            match = next((r for r in rows if r["config"] == cfg and r["goal"] == g), None)
            if match is None:
                cell = "-"
            elif match["status"] == "running":
                cell = "running"
            elif "distance" not in match:
                cell = f"FAIL(rc={match.get('rc')})"
            else:
                done += 1
                mark = "*" if match["reached"] else ""
                cell = f"{match['distance']:.3f}{mark}"
                if match["reached"]:
                    reached += 1
                    if match.get("reach_time") is not None:
                        cell += f"@{match['reach_time']:.1f}s"
            line += f"{cell:<14}"
        line += f"{reached}/{done}"
        print(line)

    print()
    print("* = reached goal (distance < 0.2m); @Ns = time to reach it")
    print()
    total_done = sum(1 for r in rows if "distance" in r)
    total_reached = sum(1 for r in rows if r.get("reached"))
    total_running = sum(1 for r in rows if r["status"] == "running")
    total_failed = sum(1 for r in rows if r["status"] not in ("running",) and "distance" not in r)
    expected = len(configs) * len(goals)
    print(f"Overall: {total_reached}/{total_done} reached, {total_running} still running, "
          f"{total_failed} failed/no-result, {len(rows)} logs total (expect {expected})")
    reach_times = [r["reach_time"] for r in rows if r.get("reached") and r.get("reach_time") is not None]
    if reach_times:
        arr = np.array(reach_times)
        print(f"Reach time over {len(arr)} successes: mean={arr.mean():.2f}s, "
              f"median={np.median(arr):.2f}s, min={arr.min():.2f}s, max={arr.max():.2f}s")


if __name__ == "__main__":
    main()
