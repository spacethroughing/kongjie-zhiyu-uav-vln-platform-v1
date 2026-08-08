from __future__ import annotations

from .models import MissionPlan, RoutePoint, SceneProfile, SearchMissionRequest, SearchZone, Vec3


def _point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def coverage_route(zone: SearchZone) -> list[RoutePoint]:
    """Create a deterministic boustrophedon route inside a configured polygon."""
    xs = [point[0] for point in zone.polygon.points]
    ys = [point[1] for point in zone.polygon.points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    spacing = zone.lane_spacing_m
    altitude_z = -abs(zone.search_altitude_m)
    route: list[RoutePoint] = []
    y = y_min
    reverse = False
    while y <= y_max + 1e-6:
        samples: list[tuple[float, float]] = []
        x = x_min
        sample_step = max(spacing / 2, 0.5)
        while x <= x_max + 1e-6:
            if _point_in_polygon(x, y, zone.polygon.points):
                samples.append((x, y))
            x += sample_step
        if samples:
            endpoints = [samples[0], samples[-1]] if len(samples) > 1 else [samples[0]]
            if reverse:
                endpoints.reverse()
            for px, py in endpoints:
                route.append(
                    RoutePoint(index=len(route), position=Vec3(x=px, y=py, z=altitude_z), observe=True)
                )
            reverse = not reverse
        y += spacing
    if not route:
        raise ValueError(f"search zone {zone.id!r} produced an empty route")
    return route


def build_plan(profile: SceneProfile, request: SearchMissionRequest) -> MissionPlan:
    try:
        zone = next(zone for zone in profile.zones if zone.id == request.zone_id)
    except StopIteration as error:
        raise ValueError(f"unknown search zone: {request.zone_id}") from error
    if zone.search_altitude_m < profile.safety.min_altitude_m:
        raise ValueError("zone search altitude is below the safety minimum")
    if zone.search_altitude_m > profile.safety.max_altitude_m:
        raise ValueError("zone search altitude exceeds the safety maximum")
    route = coverage_route(zone)
    return MissionPlan(
        request=request,
        route=route,
        observation_yaws_deg=zone.observation_yaws_deg,
        safety=profile.safety,
        safety_summary=[
            f"搜索高度 {zone.search_altitude_m:.1f} m",
            f"最大速度 {profile.safety.max_speed_mps:.1f} m/s",
            f"接近距离不少于 {profile.safety.min_standoff_m:.1f} m",
            f"任务时限 {profile.safety.max_mission_seconds} s",
            "至少两个不同视角一致后才允许接近",
        ],
    )

