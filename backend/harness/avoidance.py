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
    revisit_ratio: float = 0.0


def decode_point_cloud(values: Iterable[float]) -> list[Vec3]:
    """Convert the bridge's compact flat xyz array into finite NED points."""
    raw = list(values)
    points: list[Vec3] = []
    for index in range(0, len(raw) - 2, 3):
        x, y, z = raw[index], raw[index + 1], raw[index + 2]
        if all(math.isfinite(value) for value in (x, y, z)):
            points.append(Vec3(x=x, y=y, z=z))
    return points


def point_cloud_preview_payload(
    points: Iterable[Vec3],
    vehicle_position: Vec3,
    *,
    max_points: int = 900,
    max_range_m: float = 60.0,
) -> dict:
    """Create a bounded world-NED point-cloud payload for live visualization.

    Collision checking continues to use the complete decoded scan.  This
    deterministic spatially ordered sample exists only to keep WebSocket and
    browser rendering costs bounded.
    """
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    visible = [
        point
        for point in points
        if 0.3
        <= math.sqrt(
            (point.x - vehicle_position.x) ** 2
            + (point.y - vehicle_position.y) ** 2
            + (point.z - vehicle_position.z) ** 2
        )
        <= max_range_m
    ]
    stride = max(1, math.ceil(len(visible) / max_points))
    sampled = visible[::stride][:max_points]
    return {
        "data_frame": "VehicleInertialFrame",
        "point_count": len(visible),
        "sampled_point_count": len(sampled),
        "vehicle_position": vehicle_position.model_dump(mode="json"),
        "points": [
            [round(point.x, 3), round(point.y, 3), round(point.z, 3)]
            for point in sampled
        ],
    }


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
    *,
    allow_escape_from_start: bool = False,
) -> CorridorAssessment:
    minimum: float | None = None
    nearest: Vec3 | None = None
    for point in points:
        if allow_escape_from_start:
            start_distance = math.sqrt(
                (point.x - start.x) ** 2
                + (point.y - start.y) ** 2
                + (point.z - start.z) ** 2
            )
            move_x = end.x - start.x
            move_y = end.y - start.y
            move_z = end.z - start.z
            moving_away = (
                (start.x - point.x) * move_x
                + (start.y - point.y) * move_y
                + (start.z - point.z) * move_z
            ) >= 0
            end_distance = math.sqrt(
                (point.x - end.x) ** 2
                + (point.y - end.y) ** 2
                + (point.z - end.z) ** 2
            )
            if (
                start_distance < required_clearance_m
                and moving_away
                and end_distance >= start_distance + 0.2
            ):
                # The vehicle is already inside this point's conservative
                # planning radius. A monotonic escape must not be rejected
                # solely because every candidate segment shares its start.
                continue
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
    recent_waypoints: Iterable[Vec3] = (),
    segment_revisit_cost: Callable[[Vec3, Vec3], float] | None = None,
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
    step_candidates = sorted(
        {max(2.5, step * 0.65), max(2.5, step * 0.82), step},
        reverse=True,
    )
    distance_to_goal = math.sqrt(
        dx * dx + dy * dy + (goal.z - start.z) * (goal.z - start.z)
    )
    # A 110-degree candidate looked like a recovery option but repeatedly sent
    # the vehicle back into the corridor it had just mapped. A local sidestep
    # may turn at most 90 degrees; a true deadlock fails closed and lets the
    # mission-level frontier planner choose another goal.
    angle_magnitudes = (40.0, 55.0, 70.0, 90.0)
    sides = (
        (preferred_side, -preferred_side)
        if preferred_side in (-1, 1)
        else (1, -1)
    )
    candidates: list[tuple[float, Detour]] = []
    start_clearance = min(
        (
            math.sqrt(
                (point.x - start.x) ** 2
                + (point.y - start.y) ** 2
                + (point.z - start.z) ** 2
            )
            for point in points
        ),
        default=None,
    )
    # A pure climb is the safest deadlock escape when all horizontal segments
    # share a close start-point return. It is still geofence/altitude checked.
    climb_candidates = (
        (min(2.0, step * 0.5), min(3.0, step * 0.75))
        if start_clearance is not None and start_clearance < required_clearance_m
        else ()
    )
    for climb in climb_candidates:
        waypoint = Vec3(x=start.x, y=start.y, z=start.z - climb)
        if not is_segment_allowed(start, waypoint):
            continue
        assessment = assess_corridor(
            start,
            waypoint,
            points,
            required_clearance_m,
            allow_escape_from_start=True,
        )
        if assessment.blocked:
            continue
        clearance_bonus = min(
            assessment.minimum_clearance_m or required_clearance_m * 2,
            required_clearance_m * 3,
        )
        revisit_ratio = min(
            1.0,
            max(0.0, segment_revisit_cost(start, waypoint)),
        ) if segment_revisit_cost else 0.0
        candidates.append(
            (
                distance_to_goal
                + climb * 0.6
                + revisit_ratio * 7.0
                - clearance_bonus * 0.03,
                Detour(
                    waypoint=waypoint,
                    side=preferred_side if preferred_side in (-1, 1) else 1,
                    angle_degrees=0.0,
                    minimum_clearance_m=assessment.minimum_clearance_m,
                    revisit_ratio=revisit_ratio,
                ),
            )
        )
    for side_rank, side in enumerate(sides):
        for angle_degrees in angle_magnitudes:
            angle = heading + side * math.radians(angle_degrees)
            for candidate_step in step_candidates:
                altitude_ratio = min(
                    1.0, candidate_step / max(distance_to_goal, 1e-9)
                )
                nominal_z = start.z + (goal.z - start.z) * altitude_ratio
                # EGO-style local replanning is three-dimensional. Horizontal
                # wall-following remains preferred; a short climb is a bounded
                # fallback when every same-altitude corridor is occupied.
                climb = min(2.0, candidate_step * 0.5)
                for vertical_offset in (0.0, -climb):
                    waypoint = Vec3(
                        x=start.x + math.cos(angle) * candidate_step,
                        y=start.y + math.sin(angle) * candidate_step,
                        z=nominal_z + vertical_offset,
                    )
                    if not is_segment_allowed(start, waypoint):
                        continue
                    assessment = assess_corridor(
                        start,
                        waypoint,
                        points,
                        required_clearance_m,
                        allow_escape_from_start=True,
                    )
                    if assessment.blocked:
                        continue
                    remaining = math.hypot(
                        goal.x - waypoint.x, goal.y - waypoint.y
                    )
                    clearance_bonus = min(
                        assessment.minimum_clearance_m
                        or required_clearance_m * 2,
                        required_clearance_m * 3,
                    )
                    heading_change = 0.0
                    if previous_heading_rad is not None:
                        heading_change = abs(
                            (angle - previous_heading_rad + math.pi)
                            % (2 * math.pi)
                            - math.pi
                        )
                        if heading_change > math.radians(100.0):
                            continue
                    if any(
                        math.hypot(
                            waypoint.x - recent.x, waypoint.y - recent.y
                        )
                        < candidate_step * 0.55
                        for recent in recent_waypoints
                    ):
                        continue
                    revisit_ratio = min(
                        1.0,
                        max(0.0, segment_revisit_cost(start, waypoint)),
                    ) if segment_revisit_cost else 0.0
                    score = (
                        remaining
                        + abs(angle_degrees) * 0.008
                        + side_rank * 0.35
                        + heading_change * 0.55
                        + abs(vertical_offset) * 0.4
                        + (step - candidate_step) * 0.08
                        + revisit_ratio * 7.0
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
                                revisit_ratio=revisit_ratio,
                            ),
                        )
                    )
    return min(candidates, key=lambda item: item[0])[1] if candidates else None
