from __future__ import annotations

import math

from .models import Polygon, SafetyEnvelope, SearchZone, Telemetry, Vec3


class SafetyViolation(RuntimeError):
    pass


def point_in_polygon(point: Vec3, polygon: Polygon) -> bool:
    x, y = point.x, point.y
    for index, (x1, y1) in enumerate(polygon.points):
        x2, y2 = polygon.points[(index + 1) % len(polygon.points)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-7 and min(x1, x2) - 1e-7 <= x <= max(x1, x2) + 1e-7 and min(
            y1, y2
        ) - 1e-7 <= y <= max(y1, y2) + 1e-7:
            return True
    inside = False
    j = len(polygon.points) - 1
    for i, (xi, yi) in enumerate(polygon.points):
        xj, yj = polygon.points[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def validate_position(position: Vec3, zone: SearchZone, envelope: SafetyEnvelope) -> None:
    altitude = -position.z
    if altitude < envelope.min_altitude_m or altitude > envelope.max_altitude_m:
        raise SafetyViolation(f"altitude {altitude:.2f} m is outside the safety envelope")
    if not point_in_polygon(position, zone.polygon):
        raise SafetyViolation("position is outside the configured search geofence")
    for no_fly in envelope.no_fly_zones:
        if point_in_polygon(position, no_fly):
            raise SafetyViolation("position intersects a configured no-fly zone")


def validate_telemetry(telemetry: Telemetry) -> None:
    if telemetry.collision:
        raise SafetyViolation("AirSim reported a collision")
    values = (
        telemetry.position.x,
        telemetry.position.y,
        telemetry.position.z,
        telemetry.velocity.x,
        telemetry.velocity.y,
        telemetry.velocity.z,
    )
    if not all(math.isfinite(value) for value in values):
        raise SafetyViolation("telemetry contains a non-finite value")


def approach_point(drone: Vec3, target: Vec3, altitude_m: float, standoff_m: float) -> Vec3:
    dx = drone.x - target.x
    dy = drone.y - target.y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        dx, dy, length = -1.0, 0.0, 1.0
    return Vec3(
        x=target.x + dx / length * standoff_m,
        y=target.y + dy / length * standoff_m,
        z=-abs(altitude_m),
    )
