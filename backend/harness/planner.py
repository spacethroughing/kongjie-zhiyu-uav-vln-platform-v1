from __future__ import annotations

import math

from .models import (
    MissionPlan,
    MissionPlanParameters,
    MissionTask,
    RoutePoint,
    SafetyBounds,
    SceneProfile,
    SearchMissionRequest,
    SearchZone,
    Vec3,
)


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
    """Create a continuous, home-aware boustrophedon coverage route.

    Every generated swath is traversed exactly once and consecutive swaths are
    connected at the same side of the polygon.  Choosing among the two sweep
    directions and two first-swath orientations gives a short entry from home
    without inserting a centre ``spine`` that must later be retraced.
    """
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
    lanes: list[list[tuple[float, float]]] = []
    y = y_min
    while y <= y_max + 1e-6:
        samples: list[tuple[float, float]] = []
        x = x_min
        sample_step = max(spacing / 2, 0.5)
        while x <= x_max + 1e-6:
            if _point_in_polygon(x, y, coverage.points):
                samples.append((x, y))
            x += sample_step
        if samples:
            lanes.append(samples)
        y += spacing
    if not lanes:
        raise ValueError(f"search zone {zone.id!r} produced an empty route")

    variants: list[list[tuple[float, float]]] = []
    for reverse_lane_order in (False, True):
        ordered_lanes = list(reversed(lanes)) if reverse_lane_order else lanes
        for reverse_first_lane in (False, True):
            points: list[tuple[float, float]] = []
            reverse = reverse_first_lane
            for samples in ordered_lanes:
                endpoints = (
                    [samples[0], samples[-1]]
                    if len(samples) > 1
                    else list(samples)
                )
                if reverse:
                    endpoints.reverse()
                points.extend(endpoints)
                reverse = not reverse
            variants.append(points)

    def route_cost(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
        start_distance = math.hypot(points[0][0], points[0][1])
        path_distance = sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(points, points[1:])
        )
        # Total travel is the primary objective.  The remaining values make
        # ties deterministic and prefer a shorter entry from the takeoff point.
        return (
            start_distance + path_distance,
            start_distance,
            points[0][1],
            points[0][0],
        )

    selected = min(variants, key=route_cost)
    return [
        RoutePoint(
            index=index,
            position=Vec3(x=px, y=py, z=altitude_z),
            observe=True,
        )
        for index, (px, py) in enumerate(selected)
    ]


def route_metrics(route: list[RoutePoint]) -> dict[str, float | int | str]:
    """Summarize global-route continuity, including the entry leg from home."""
    if not route:
        return {
            "pattern": "continuous_boustrophedon",
            "waypoints": 0,
            "total_length_m": 0.0,
            "max_leg_m": 0.0,
        }
    positions = [Vec3(x=0, y=0, z=route[0].position.z)] + [
        point.position for point in route
    ]
    legs = [
        math.hypot(second.x - first.x, second.y - first.y)
        for first, second in zip(positions, positions[1:])
    ]
    return {
        "pattern": "continuous_boustrophedon",
        "waypoints": len(route),
        "total_length_m": round(sum(legs), 2),
        "max_leg_m": round(max(legs, default=0.0), 2),
    }


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


def default_plan_parameters(
    profile: SceneProfile,
    request: SearchMissionRequest,
) -> MissionPlanParameters:
    zone = resolve_search_zone(profile, request)
    return MissionPlanParameters(
        search_altitude_m=zone.search_altitude_m,
        lane_spacing_m=zone.lane_spacing_m,
        max_speed_mps=profile.safety.max_speed_mps,
        approach_speed_mps=profile.safety.approach_speed_mps,
        min_standoff_m=profile.safety.min_standoff_m,
        min_clearance_m=profile.safety.min_clearance_m,
        max_mission_seconds=profile.safety.max_mission_seconds,
        mapping_coverage_target=request.mapping_coverage_target,
    )


def resolve_plan_zone(
    profile: SceneProfile,
    request: SearchMissionRequest,
    parameters: MissionPlanParameters,
) -> SearchZone:
    zone = resolve_search_zone(profile, request)
    return zone.model_copy(
        update={
            "search_altitude_m": parameters.search_altitude_m,
            "lane_spacing_m": parameters.lane_spacing_m,
        }
    )


def _validate_plan_parameters(
    profile: SceneProfile,
    parameters: MissionPlanParameters,
) -> None:
    # Pydantic enforces the absolute simulator-editing envelope. The values are
    # deliberately mission-scoped: applying a review revision defines that
    # plan's safety envelope without modifying the scene defaults.
    del profile


def build_plan(
    profile: SceneProfile,
    request: SearchMissionRequest,
    *,
    parameters: MissionPlanParameters | None = None,
    version: int = 1,
) -> MissionPlan:
    parameters = parameters or default_plan_parameters(profile, request)
    _validate_plan_parameters(profile, parameters)
    request = request.model_copy(
        update={"mapping_coverage_target": parameters.mapping_coverage_target}
    )
    zone = resolve_plan_zone(profile, request, parameters)
    safety = profile.safety.model_copy(
        update={
            "min_altitude_m": min(
                profile.safety.min_altitude_m,
                parameters.search_altitude_m,
            ),
            "max_altitude_m": max(
                profile.safety.max_altitude_m,
                parameters.search_altitude_m,
            ),
            "max_speed_mps": parameters.max_speed_mps,
            "approach_speed_mps": parameters.approach_speed_mps,
            "min_standoff_m": parameters.min_standoff_m,
            "min_clearance_m": parameters.min_clearance_m,
            "max_mission_seconds": parameters.max_mission_seconds,
        }
    )
    route = coverage_route(zone)
    metrics = route_metrics(route)
    bounds = request.safety_bounds
    range_summary = (
        f"手动安全范围（相对任务起飞点）X[{bounds.x_min:.1f}, {bounds.x_max:.1f}] m / "
        f"Y[{bounds.y_min:.1f}, {bounds.y_max:.1f}] m"
        if bounds
        else "使用相对任务起飞点的场景预设安全范围"
    )
    if request.mission_mode == "semantic_mapping":
        tasks = [
            MissionTask(
                id="map-region",
                kind="semantic_mapping",
                label=f"建立 {zone.name} 占据与语义拓扑图",
                coverage_target=request.mapping_coverage_target,
            )
        ]
        mission_summary = [
            f"区域建图覆盖率目标 {request.mapping_coverage_target:.0%}",
            "按确定性覆盖航线采集 LiDAR、RGB 与深度，VLM 仅附加语义标签",
        ]
    else:
        tasks = [
            MissionTask(
                id=f"target-{index + 1}",
                kind="target_search",
                label=f"搜索并取证：{target}",
                target_text=target,
            )
            for index, target in enumerate(request.targets)
        ]
        mission_summary = [
            f"按顺序探索 {len(tasks)} 个开放词汇目标",
            "每个目标独立调用 VLM 识别、深度定位、安全接近并保存证据",
        ]
    return MissionPlan(
        version=version,
        request=request,
        parameters=parameters,
        tasks=tasks,
        route=route,
        observation_yaws_deg=zone.observation_yaws_deg,
        safety=safety,
        safety_summary=[
            *mission_summary,
            "本计划使用用户审核参数；物理量必须为正，绝对编辑上限 999",
            f"搜索高度 {zone.search_altitude_m:.1f} m",
            f"覆盖航线间距 {zone.lane_spacing_m:.1f} m",
            f"最大速度 {safety.max_speed_mps:.1f} m/s / 接近速度 {safety.approach_speed_mps:.1f} m/s",
            f"接近距离不少于 {safety.min_standoff_m:.1f} m",
            f"任务时限 {safety.max_mission_seconds} s",
            range_summary,
            (
                f"连续往复式覆盖航线 {metrics['waypoints']} 点 / "
                f"约 {metrics['total_length_m']:.1f} m / "
                f"最大航段 {metrics['max_leg_m']:.1f} m"
            ),
            f"LiDAR 航段避障，最小净空 {safety.min_clearance_m:.1f} m",
            "接近采用异步滚动 Action Chunk；VLM 后台连续更新，LiDAR 在段间实时重规划",
            "环视未发现目标时，VLM 仅在连续航线的局部前视窗口内选择探索目标，不得重排全图航点",
        ],
    )
