import pytest

from harness.config import REPO_ROOT, load_scenes
from harness.models import Polygon, SafetyBounds, SceneProfile, SearchMissionRequest, SearchZone, Vec3
from harness.planner import build_plan, coverage_route, resolve_search_zone
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
    assert any("目标居中" in line for line in plan.safety_summary)


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
