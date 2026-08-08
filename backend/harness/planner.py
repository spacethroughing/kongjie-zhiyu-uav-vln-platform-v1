from __future__ import annotations

from .models import MissionPlan, RoutePoint, SafetyBounds, SceneProfile, SearchMissionRequest, SearchZone, Vec3


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
    coverage = zone.coverage_polygon or zone.polygon
    xs = [point[0] for point in coverage.points]
    ys = [point[1] for point in coverage.points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    spacing = zone.lane_spacing_m
    # A half-lane inset keeps the aircraft away from physical walls and props
    # at the geofence edge while the camera footprint still covers the border.
    inset = min(spacing / 2, (x_max - x_min) / 4, (y_max - y_min) / 4)
    x_min, x_max = x_min + inset, x_max - inset
    y_min, y_max = y_min + inset, y_max - inset
    altitude_z = -abs(zone.search_altitude_m)
    route: list[RoutePoint] = []
    y = y_min
    reverse = False
    while y <= y_max + 1e-6:
        samples: list[tuple[float, float]] = []
        x = x_min
        sample_step = max(spacing / 2, 0.5)
        while x <= x_max + 1e-6:
            if _point_in_polygon(x, y, coverage.points):
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


def _polygon_bounds(zone: SearchZone) -> SafetyBounds:
    xs = [point[0] for point in zone.polygon.points]
    ys = [point[1] for point in zone.polygon.points]
    return SafetyBounds(x_min=min(xs), x_max=max(xs), y_min=min(ys), y_max=max(ys))


def resolve_search_zone(profile: SceneProfile, request: SearchMissionRequest) -> SearchZone:
    try:
        zone = next(zone for zone in profile.zones if zone.id == request.zone_id)
    except StopIteration as error:
        raise ValueError(f"unknown search zone: {request.zone_id}") from error
    bounds = request.safety_bounds
    if bounds is None:
        return zone
    allowed = profile.manual_safety_bounds or _polygon_bounds(zone)
    epsilon = 1e-6
    if (
        bounds.x_min < allowed.x_min - epsilon
        or bounds.x_max > allowed.x_max + epsilon
        or bounds.y_min < allowed.y_min - epsilon
        or bounds.y_max > allowed.y_max + epsilon
    ):
        raise ValueError(
            "manual safety bounds exceed the scene limit "
            f"X[{allowed.x_min:.1f}, {allowed.x_max:.1f}] "
            f"Y[{allowed.y_min:.1f}, {allowed.y_max:.1f}]"
        )
    if not (bounds.x_min <= 0 <= bounds.x_max and bounds.y_min <= 0 <= bounds.y_max):
        raise ValueError("manual safety bounds must include the NED home point (0, 0)")
    polygon = bounds.polygon()
    return zone.model_copy(update={"polygon": polygon, "coverage_polygon": polygon})


def build_plan(profile: SceneProfile, request: SearchMissionRequest) -> MissionPlan:
    zone = resolve_search_zone(profile, request)
    if zone.search_altitude_m < profile.safety.min_altitude_m:
        raise ValueError("zone search altitude is below the safety minimum")
    if zone.search_altitude_m > profile.safety.max_altitude_m:
        raise ValueError("zone search altitude exceeds the safety maximum")
    route = coverage_route(zone)
    bounds = request.safety_bounds
    range_summary = (
        f"手动安全范围 X[{bounds.x_min:.1f}, {bounds.x_max:.1f}] m / "
        f"Y[{bounds.y_min:.1f}, {bounds.y_max:.1f}] m"
        if bounds
        else "使用场景预设安全范围"
    )
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
            range_summary,
            "目标居中并通过深度复核后才允许接近",
        ],
    )
