import json
import math
from pathlib import Path

import pytest

from harness.avoidance import (
    assess_corridor,
    choose_local_detour,
    point_cloud_preview_payload,
    point_to_segment_distance,
)
from harness.bridge import MockVehicleAdapter
from harness.config import REPO_ROOT, Settings, load_scenes
from harness.events import EventBus
from harness.llm import MockProvider
from harness.mission import MissionService, RunControl
from harness.models import SearchMissionRequest, Vec3
from harness.planner import resolve_search_zone
from harness.simulator import SimulatorManager
from harness.store import Store


def obstacle_ring(center: Vec3, radius: float = 0.8) -> list[Vec3]:
    return [
        Vec3(
            x=center.x + math.cos(index * math.tau / 32) * radius,
            y=center.y + math.sin(index * math.tau / 32) * radius,
            z=center.z,
        )
        for index in range(32)
    ]


def test_point_cloud_preview_is_bounded_and_keeps_world_ned_coordinates():
    vehicle = Vec3(x=5, y=-3, z=-4)
    points = [
        Vec3(
            x=vehicle.x + 10 * math.cos(index * math.tau / 500),
            y=vehicle.y + 10 * math.sin(index * math.tau / 500),
            z=-4 + (index % 20) * 0.1,
        )
        for index in range(500)
    ]
    payload = point_cloud_preview_payload(points, vehicle, max_points=80)

    assert payload["data_frame"] == "VehicleInertialFrame"
    assert payload["point_count"] == 500
    assert payload["sampled_point_count"] <= 80
    assert payload["vehicle_position"] == vehicle.model_dump(mode="json")
    assert all(len(point) == 3 for point in payload["points"])


def test_corridor_detects_obstacle_and_selects_clear_detour():
    start = Vec3(x=0, y=0, z=-5)
    goal = Vec3(x=8, y=0, z=-5)
    points = obstacle_ring(Vec3(x=3, y=0, z=-5))
    direct = assess_corridor(start, Vec3(x=4, y=0, z=-5), points, 1.5)
    assert direct.blocked
    detour = choose_local_detour(
        start,
        goal,
        points,
        required_clearance_m=1.5,
        step_m=4,
        is_segment_allowed=lambda _start, _end: True,
    )
    assert detour is not None
    assert abs(detour.waypoint.y) > 2
    assert not assess_corridor(start, detour.waypoint, points, 1.5).blocked


def test_local_detour_uses_vertical_escape_when_horizontal_exit_is_occupied():
    start = Vec3(x=0, y=0, z=-5)
    points = obstacle_ring(start, radius=0.5)
    detour = choose_local_detour(
        start,
        Vec3(x=8, y=0, z=-5),
        points,
        required_clearance_m=1.5,
        step_m=4,
        is_segment_allowed=lambda _start, _end: True,
    )
    assert detour is not None
    assert detour.waypoint.x == start.x
    assert detour.waypoint.y == start.y
    assert detour.waypoint.z < start.z
    assert not assess_corridor(
        start,
        detour.waypoint,
        points,
        1.5,
        allow_escape_from_start=True,
    ).blocked


def test_local_detour_fails_closed_when_vertical_escape_is_disallowed():
    start = Vec3(x=0, y=0, z=-5)
    assert choose_local_detour(
        start,
        Vec3(x=8, y=0, z=-5),
        obstacle_ring(start, radius=0.5),
        required_clearance_m=1.5,
        step_m=4,
        is_segment_allowed=lambda _start, end: end.z >= start.z,
    ) is None


def test_local_detour_cost_preserves_heading_continuity_across_replans():
    detour = choose_local_detour(
        Vec3(x=0, y=0, z=-5),
        Vec3(x=8, y=0, z=-5),
        [],
        required_clearance_m=1.5,
        step_m=4,
        is_segment_allowed=lambda _start, _end: True,
        previous_heading_rad=math.radians(-40),
    )
    assert detour is not None
    assert detour.side == -1
    assert detour.angle_degrees == -40


def test_local_detour_prefers_an_unflown_corridor():
    detour = choose_local_detour(
        Vec3(x=0, y=0, z=-5),
        Vec3(x=8, y=0, z=-5),
        [],
        required_clearance_m=1.5,
        step_m=4,
        is_segment_allowed=lambda _start, _end: True,
        # The otherwise-preferred right candidate re-enters the old path.
        segment_revisit_cost=lambda _start, end: 1.0 if end.y > 0 else 0.0,
    )
    assert detour is not None
    assert detour.side == -1
    assert detour.revisit_ratio == 0.0


def test_local_detour_rejects_heading_reversal_and_recent_waypoint_loop():
    start = Vec3(x=0, y=0, z=-5)
    goal = Vec3(x=-8, y=0, z=-5)
    first = choose_local_detour(
        start,
        goal,
        [],
        required_clearance_m=1.5,
        step_m=4,
        is_segment_allowed=lambda _start, _end: True,
        previous_heading_rad=0.0,
    )
    assert first is not None
    first_heading = math.atan2(first.waypoint.y, first.waypoint.x)
    assert abs((first_heading + math.pi) % (2 * math.pi) - math.pi) <= math.radians(100)

    second = choose_local_detour(
        start,
        goal,
        [],
        required_clearance_m=1.5,
        step_m=4,
        is_segment_allowed=lambda _start, _end: True,
        previous_heading_rad=0.0,
        recent_waypoints=[first.waypoint],
    )
    assert second is not None
    assert math.hypot(
        second.waypoint.x - first.waypoint.x,
        second.waypoint.y - first.waypoint.y,
    ) >= 4 * 0.55


class StaticObstacleAdapter(MockVehicleAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.position = Vec3(x=0, y=0, z=-5)
        self.landed = False
        self.center = Vec3(x=3, y=0, z=-5)
        self.radius = 0.8
        self.moves: list[Vec3] = []

    async def request(self, operation: str, **arguments):
        if operation == "lidar_scan":
            points = obstacle_ring(self.center, self.radius)
            return {
                "point_cloud": [
                    value
                    for point in points
                    for value in (point.x, point.y, point.z)
                ],
                "point_count": len(points),
                "sampled_point_count": len(points),
                "data_frame": "VehicleInertialFrame",
            }
        if operation == "move_to":
            target = Vec3(
                x=float(arguments["x"]),
                y=float(arguments["y"]),
                z=float(arguments["z"]),
            )
            assert (
                point_to_segment_distance(self.center, self.position, target)
                > self.radius
            )
            self.moves.append(target)
        return await super().request(operation, **arguments)


@pytest.mark.asyncio
async def test_segmented_flight_replans_around_static_obstacle(tmp_path: Path):
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        scenes_file=REPO_ROOT / "configs" / "scenes.json",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        provider="mock",
        llm_base_url="",
        llm_model="",
        llm_api_key="",
        llm_timeout_seconds=10,
    )
    events = EventBus()
    store = Store(settings.data_dir / "test.sqlite3", settings.runs_dir)
    simulator = SimulatorManager(load_scenes(settings.scenes_file), events)
    missions = MissionService(settings, store, events, simulator, MockProvider())
    await simulator.start("mock")
    assert simulator.adapter
    await simulator.adapter.close()
    adapter = StaticObstacleAdapter()
    simulator.adapter = adapter
    plan = missions.create_plan(
        SearchMissionRequest(
            scene_id="mock", zone_id="mock-fixture", target_text="red cube"
        )
    )
    zone = resolve_search_zone(simulator.profiles["mock"], plan.request)
    run = store.create_run(plan.id)
    await missions._move_segmented(
        run,
        plan,
        zone,
        Vec3(x=8, y=0, z=-5),
        Vec3(x=0, y=0, z=0),
        RunControl(),
    )
    assert math.hypot(adapter.position.x - 8, adapter.position.y) <= 1.5
    assert any(abs(move.y) > 2 for move in adapter.moves)
    written = [
        json.loads(line)
        for line in (Path(run.artifact_dir) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(event["topic"] == "avoidance.detour" for event in written)
    await missions.close()
    await simulator.close()
    store.close()
