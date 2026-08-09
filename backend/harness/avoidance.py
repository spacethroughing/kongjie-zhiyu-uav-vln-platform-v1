from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .models import Vec3


@dataclass(frozen=True)
class CorridorAssessment:
    blocked: bool
    minimum_clearance_m: float | None
    nearest_point: Vec3 | None


@dataclass(frozen=True)
class Detour:
    waypoint: Vec3
    side: int
    angle_degrees: float
    minimum_clearance_m: float | None


def decode_point_cloud(values: Iterable[float]) -> list[Vec3]:
    """Convert the bridge's compact flat xyz array into finite NED points."""
    raw = list(values)
    points: list[Vec3] = []
    for index in range(0, len(raw) - 2, 3):
        x, y, z = raw[index], raw[index + 1], raw[index + 2]
        if all(math.isfinite(value) for value in (x, y, z)):
            points.append(Vec3(x=x, y=y, z=z))
    return points


def point_to_segment_distance(point: Vec3, start: Vec3, end: Vec3) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z
    length_squared = dx * dx + dy * dy + dz * dz
    if length_squared <= 1e-12:
        return math.sqrt(
            (point.x - start.x) ** 2
            + (point.y - start.y) ** 2
            + (point.z - start.z) ** 2
        )
    projection = (
        (point.x - start.x) * dx
        + (point.y - start.y) * dy
        + (point.z - start.z) * dz
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest = Vec3(
        x=start.x + projection * dx,
        y=start.y + projection * dy,
        z=start.z + projection * dz,
    )
    return math.sqrt(
        (point.x - closest.x) ** 2
        + (point.y - closest.y) ** 2
        + (point.z - closest.z) ** 2
    )


def assess_corridor(
    start: Vec3,
    end: Vec3,
    points: Iterable[Vec3],
    required_clearance_m: float,
) -> CorridorAssessment:
    minimum: float | None = None
    nearest: Vec3 | None = None
    for point in points:
        clearance = point_to_segment_distance(point, start, end)
        if minimum is None or clearance < minimum:
            minimum = clearance
            nearest = point
    return CorridorAssessment(
        blocked=minimum is not None and minimum < required_clearance_m,
        minimum_clearance_m=minimum,
        nearest_point=nearest,
    )


def choose_local_detour(
    start: Vec3,
    goal: Vec3,
    points: list[Vec3],
    required_clearance_m: float,
    step_m: float,
    is_segment_allowed: Callable[[Vec3, Vec3], bool],
    preferred_side: int | None = None,
    previous_heading_rad: float | None = None,
) -> Detour | None:
    """Choose a short collision-free sidestep; the caller rescans after it.

    ``preferred_side`` and ``previous_heading_rad`` provide local-planner
    hysteresis across rolling replans.  They prevent equally safe candidates
    from alternating left/right or producing a sharp heading reversal.
    """
    dx = goal.x - start.x
    dy = goal.y - start.y
    heading = math.atan2(dy, dx)
    step = max(2.5, step_m)
    distance_to_goal = math.sqrt(
        dx * dx + dy * dy + (goal.z - start.z) * (goal.z - start.z)
    )
    altitude_ratio = min(1.0, step / max(distance_to_goal, 1e-9))
    waypoint_z = start.z + (goal.z - start.z) * altitude_ratio
    angle_magnitudes = (40.0, 55.0, 70.0, 90.0, 110.0)
    sides = (
        (preferred_side, -preferred_side)
        if preferred_side in (-1, 1)
        else (1, -1)
    )
    candidates: list[tuple[float, Detour]] = []
    for side_rank, side in enumerate(sides):
        for angle_degrees in angle_magnitudes:
            angle = heading + side * math.radians(angle_degrees)
            waypoint = Vec3(
                x=start.x + math.cos(angle) * step,
                y=start.y + math.sin(angle) * step,
                z=waypoint_z,
            )
            if not is_segment_allowed(start, waypoint):
                continue
            assessment = assess_corridor(
                start, waypoint, points, required_clearance_m
            )
            if assessment.blocked:
                continue
            remaining = math.hypot(goal.x - waypoint.x, goal.y - waypoint.y)
            clearance_bonus = min(
                assessment.minimum_clearance_m or required_clearance_m * 2,
                required_clearance_m * 3,
            )
            heading_change = 0.0
            if previous_heading_rad is not None:
                heading_change = abs(
                    (angle - previous_heading_rad + math.pi) % (2 * math.pi)
                    - math.pi
                )
            # Prefer forward progress, keep an established wall-following side,
            # retain heading continuity, and use greater observed clearance as
            # the tie breaker. This is a lightweight rolling local-planner cost,
            # not a global route replacement.
            score = (
                remaining
                + abs(angle_degrees) * 0.008
                + side_rank * 0.35
                + heading_change * 0.55
                - clearance_bonus * 0.03
            )
            candidates.append(
                (
                    score,
                    Detour(
                        waypoint=waypoint,
                        side=side,
                        angle_degrees=side * angle_degrees,
                        minimum_clearance_m=assessment.minimum_clearance_m,
                    ),
                )
            )
    return min(candidates, key=lambda item: item[0])[1] if candidates else None
