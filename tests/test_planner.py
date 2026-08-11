import math

import pytest

from harness.config import REPO_ROOT, load_scenes
from harness.models import MissionPlanParameters, Polygon, SafetyBounds, SceneProfile, SearchMissionRequest, SearchZone, Vec3
from harness.planner import build_plan, coverage_route, default_plan_parameters, resolve_search_zone, route_metrics
from harness.safety import point_in_polygon


def profile() -> SceneProfile:
    return SceneProfile(
        id="test",
        name="Test",
        mode="mock",
        zones=[
            SearchZone(
                id="main",
                name="Main",
                polygon=Polygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)]),
                search_altitude_m=5,
                lane_spacing_m=5,
            )
        ],
    )


def test_coverage_route_stays_inside_polygon():
    zone = profile().zones[0]
    route = coverage_route(zone)
    assert len(route) >= 4
    assert all(point_in_polygon(point.position, zone.polygon) for point in route)
    assert all(point.position.z == -5 for point in route)
    assert all(0 < point.position.x < 10 for point in route)
    assert all(0 < point.position.y < 10 for point in route)


def test_plan_contains_immutable_safety_summary():
    plan = build_plan(
        profile(), SearchMissionRequest(scene_id="test", zone_id="main", target_text="red cube")
    )
    assert plan.version == 1
    assert plan.route
    assert any("VLM 后台连续更新" in line for line in plan.safety_summary)


def test_review_parameters_rebuild_route_and_tighten_safety_envelope():
    configured = profile()
    request = SearchMissionRequest(
        scene_id="test",
        zone_id="main",
        target_text="region map",
        mission_mode="semantic_mapping",
    )
    defaults = default_plan_parameters(configured, request)
    custom = defaults.model_copy(
        update={
            "search_altitude_m": 7,
            "lane_spacing_m": 2.5,
            "max_speed_mps": 2,
            "approach_speed_mps": 0.8,
            "min_standoff_m": 4,
            "min_clearance_m": 2,
            "max_mission_seconds": 600,
            "mapping_coverage_target": 0.9,
        }
    )

    plan = build_plan(configured, request, parameters=custom, version=2)

    assert plan.version == 2
    assert plan.parameters == custom
    assert all(point.position.z == -7 for point in plan.route)
    assert plan.safety.max_speed_mps == 2
    assert plan.safety.min_clearance_m == 2
    assert plan.request.mapping_coverage_target == pytest.approx(0.9)
    assert plan.tasks[0].coverage_target == pytest.approx(0.9)
    assert any("覆盖航线间距 2.5 m" in line for line in plan.safety_summary)


def test_review_parameters_define_a_mission_scoped_safety_envelope():
    configured = profile()
    request = SearchMissionRequest(scene_id="test", zone_id="main", target_text="red cube")
    custom = MissionPlanParameters(
        **{
            **default_plan_parameters(configured, request).model_dump(),
            "search_altitude_m": 120,
            "lane_spacing_m": 80,
            "max_speed_mps": 120,
            "approach_speed_mps": 20,
            "min_standoff_m": 0.5,
            "min_clearance_m": 0.25,
            "max_mission_seconds": 999,
        }
    )

    plan = build_plan(configured, request, parameters=custom, version=2)

    assert plan.safety.max_altitude_m == 120
    assert plan.safety.max_speed_mps == 120
    assert plan.safety.approach_speed_mps == 20
    assert plan.safety.min_standoff_m == pytest.approx(0.5)
    assert plan.safety.min_clearance_m == pytest.approx(0.25)
    assert plan.safety.max_mission_seconds == 999


def test_coverage_polygon_can_be_smaller_than_target_geofence():
    zone = SearchZone(
        id="split-zone",
        name="Split",
        polygon=Polygon(points=[(-20, -20), (40, -20), (40, 40), (-20, 40)]),
        coverage_polygon=Polygon(points=[(-20, -20), (20, -20), (20, 20), (-20, 20)]),
        search_altitude_m=5,
        lane_spacing_m=8,
    )
    route = coverage_route(zone)
    assert max(point.position.x for point in route) <= 16
    assert max(point.position.y for point in route) <= 16
    assert point_in_polygon(Vec3(x=31, y=31, z=0), zone.polygon)


def test_asymmetric_coverage_route_is_a_continuous_monotonic_sweep():
    zone = SearchZone(
        id="asymmetric",
        name="Asymmetric",
        polygon=Polygon(points=[(-10, -70), (20, -70), (20, 10), (-10, 10)]),
        search_altitude_m=5,
        lane_spacing_m=8,
    )
    route = coverage_route(zone)
    first = route[0].position
    assert abs(first.y) <= 6
    assert math.hypot(first.x, first.y) < 10
    assert math.hypot(first.x, first.y) < math.hypot(first.x, -66)
    lane_ys = [route[index].position.y for index in range(0, len(route), 2)]
    assert lane_ys == sorted(lane_ys, reverse=True)
    assert all(
        route[index].position.y == route[index + 1].position.y
        for index in range(0, len(route), 2)
    )
    assert route_metrics(route)["max_leg_m"] <= 22.01


def test_manual_safety_bounds_are_frozen_into_the_plan_and_route():
    configured = profile().model_copy(
        update={
            "manual_safety_bounds": SafetyBounds(
                x_min=-20, x_max=20, y_min=-20, y_max=20
            )
        }
    )
    selected = SafetyBounds(x_min=-5, x_max=15, y_min=-5, y_max=15)
    request = SearchMissionRequest(
        scene_id="test", zone_id="main", target_text="red cube", safety_bounds=selected
    )
    plan = build_plan(configured, request)
    effective = resolve_search_zone(configured, plan.request)
    assert plan.request.safety_bounds == selected
    assert all(point_in_polygon(point.position, effective.polygon) for point in plan.route)
    assert any("X[-5.0, 15.0]" in line for line in plan.safety_summary)


@pytest.mark.parametrize(
    "bounds, message",
    [
        (SafetyBounds(x_min=-21, x_max=15, y_min=-5, y_max=15), "exceed"),
        (SafetyBounds(x_min=1, x_max=15, y_min=-5, y_max=15), "home point"),
    ],
)
def test_manual_safety_bounds_cannot_bypass_scene_limit_or_exclude_home(bounds, message):
    configured = profile().model_copy(
        update={
            "manual_safety_bounds": SafetyBounds(
                x_min=-20, x_max=20, y_min=-20, y_max=20
            )
        }
    )
    request = SearchMissionRequest(
        scene_id="test", zone_id="main", target_text="red cube", safety_bounds=bounds
    )
    with pytest.raises(ValueError, match=message):
        build_plan(configured, request)


def test_blocks_manual_bounds_can_authorize_the_native_cone_area():
    blocks = load_scenes(REPO_ROOT / "configs" / "scenes.json")["blocks"]
    request = SearchMissionRequest(
        scene_id="blocks",
        zone_id="blocks-main",
        target_text="圆锥体",
        safety_bounds=SafetyBounds(x_min=-10, x_max=40, y_min=-40, y_max=10),
    )
    plan = build_plan(blocks, request)
    effective = resolve_search_zone(blocks, plan.request)
    assert point_in_polygon(Vec3(x=35, y=-33, z=-2), effective.polygon)
    assert plan.route


def test_scene_profiles_allow_user_defined_ned_bounds_up_to_999_metres():
    blocks = load_scenes(REPO_ROOT / "configs" / "scenes.json")["blocks"]
    request = SearchMissionRequest(
        scene_id="blocks",
        zone_id="blocks-main",
        target_text="wide-area map",
        safety_bounds=SafetyBounds(x_min=-900, x_max=900, y_min=-850, y_max=850),
    )

    effective = resolve_search_zone(blocks, request)

    assert min(x for x, _ in effective.polygon.points) == -900
    assert max(x for x, _ in effective.polygon.points) == 900


def test_blocks_default_coverage_reaches_cone_from_the_sphere_landing_area():
    blocks = load_scenes(REPO_ROOT / "configs" / "scenes.json")["blocks"]
    plan = build_plan(
        blocks,
        SearchMissionRequest(
            scene_id="blocks", zone_id="blocks-main", target_text="圆锥体"
        ),
    )
    assert min(point.position.y for point in plan.route) <= -64
    effective = resolve_search_zone(blocks, plan.request)
    assert point_in_polygon(Vec3(x=3, y=-65, z=-3), effective.polygon)


def test_blocks_coverage_has_no_centre_spine_return_or_cross_map_jump():
    blocks = load_scenes(REPO_ROOT / "configs" / "scenes.json")["blocks"]
    plan = build_plan(
        blocks,
        SearchMissionRequest(
            scene_id="blocks", zone_id="blocks-main", target_text="圆锥体"
        ),
    )
    metrics = route_metrics(plan.route)

    # The previous centre-spine route was 662.6 m with a 72 m return jump.
    assert metrics["pattern"] == "continuous_boustrophedon"
    assert metrics["waypoints"] == 22
    assert metrics["total_length_m"] <= 455
    assert metrics["max_leg_m"] <= 32.01
    assert any("连续往复式覆盖航线" in line for line in plan.safety_summary)


def test_planner_builds_ordered_multi_target_and_semantic_mapping_tasks():
    multi = build_plan(
        profile(),
        SearchMissionRequest(
            scene_id="test",
            zone_id="main",
            target_text="圆锥体、橙色球体",
            targets=["圆锥体", "橙色球体"],
        ),
    )
    assert [task.target_text for task in multi.tasks] == ["圆锥体", "橙色球体"]
    assert all(task.kind == "target_search" for task in multi.tasks)

    mapping = build_plan(
        profile(),
        SearchMissionRequest(
            scene_id="test",
            zone_id="main",
            target_text="区域语义拓扑图",
            mission_mode="semantic_mapping",
            mapping_coverage_target=0.9,
        ),
    )
    assert len(mapping.tasks) == 1
    assert mapping.tasks[0].kind == "semantic_mapping"
    assert mapping.tasks[0].coverage_target == 0.9
