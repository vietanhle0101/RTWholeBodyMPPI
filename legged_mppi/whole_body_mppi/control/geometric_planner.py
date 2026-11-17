"""Geometric push planner: a simple global trunk plan for box pushing, rebuilt from the current state.

Push direction d points from the box to the goal. If the trunk is behind the box on that line it walks
through the box towards E = goal - contact * d (where its center ends when the box sits on the goal).
Otherwise it first walks to a standoff point behind the box, going around the box if needed.
Returns a time-indexed plan (t, P, V) for BodyReference.set_plan.
"""
import numpy as np

DT = 0.01


def polyline(t0, points, speeds, tail=2.0):
    """Constant-speed segments through points starting at time t0, holding the last point for `tail` s."""
    t, P, V = [t0], [points[0]], []
    for a, b, v in zip(points[:-1], points[1:], speeds):
        seg = np.linalg.norm(b - a)
        if seg < 1e-6:
            continue
        n = max(int(np.ceil(seg / v / DT)), 1)
        vel = (b - a) / seg * v
        for i in range(1, n + 1):
            V.append(vel); t.append(t[-1] + DT); P.append(a + (b - a) * i / n)
    V.append(np.zeros(2))
    t.append(t[-1] + tail); P.append(points[-1]); V.append(np.zeros(2))
    return np.array(t), np.array(P), np.array(V)


def seg_dist(c, a, b):
    """Distance from point c to segment ab."""
    ab = b - a
    s = np.clip((c - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(a + s * ab - c)


class GeometricPushPlanner:
    def __init__(self, goal, contact=0.47, standoff=0.75, clearance=0.6, push_speed=0.1, approach_speed=0.2,
                 align_tol=0.12, keep_tol=0.3, done_tol=0.1, lead=0.2):
        self.goal = np.asarray(goal, float)
        self.lead = lead
        self.contact, self.standoff, self.clearance = contact, standoff, clearance
        self.push_speed, self.approach_speed = push_speed, approach_speed
        self.align_tol, self.keep_tol, self.done_tol = align_tol, keep_tol, done_tol
        self.mode = "approach"

    def plan(self, t, trunk_xy, box_xy):
        p, b, g = np.asarray(trunk_xy, float), np.asarray(box_xy, float), self.goal
        if np.linalg.norm(g - b) < self.done_tol:
            self.mode = "done"
            return polyline(t, [p], [])
        d = (g - b) / np.linalg.norm(g - b)
        n = np.array([-d[1], d[0]])
        E = g - self.contact * d
        A = b - self.standoff * d

        along, lateral = (p - b) @ d, (p - b) @ n
        # enter push mode only when lined up; once pushing, re-aiming at the moved box each replan
        # corrects small offsets, so only fall back to approach when well off the line
        tol = self.keep_tol if self.mode == "push" else self.align_tol
        if abs(lateral) < tol and -self.standoff - 0.15 < along < 0:
            self.mode = "push"
            # steer through a point just ahead of the box on the box->goal line, so lateral drift is
            # corrected within ~0.7 m instead of over the whole remaining distance
            if np.linalg.norm(g - b) - self.contact > self.lead:
                return polyline(t, [p, b + self.lead * d, E], [self.push_speed, self.push_speed])
            return polyline(t, [p, E], [self.push_speed])  # E is already close: aim at it directly

        self.mode = "approach"
        points = [p]
        # go around only if the trunk is level with or ahead of the box (when behind it, the way to A is free)
        if along > -0.25 and seg_dist(b, p, A) < self.clearance:
            side = n if lateral >= 0 else -n
            W = b + self.clearance * side - self.standoff * d     # beside and behind the box
            if seg_dist(b, p, W) < self.clearance:
                points.append(b + self.clearance * side)          # beside the box first
            points.append(W)
        points += [A, E]
        speeds = [self.approach_speed] * (len(points) - 2) + [self.push_speed]
        return polyline(t, points, speeds)
