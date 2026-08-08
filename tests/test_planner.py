from harness.models import Polygon, SceneProfile, SearchMissionRequest, SearchZone, Vec3
from harness.planner import build_plan, coverage_route
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
    assert any("两个不同视角" in line for line in plan.safety_summary)


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
