import json
import math
from pathlib import Path

import pytest

from harness.avoidance import (
    assess_corridor,
    choose_local_detour,
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


def test_local_detour_fails_closed_when_every_exit_is_occupied():
    start = Vec3(x=0, y=0, z=-5)
    points = obstacle_ring(start, radius=0.5)
    assert (
        choose_local_detour(
            start,
            Vec3(x=8, y=0, z=-5),
            points,
            required_clearance_m=1.5,
            step_m=4,
            is_segment_allowed=lambda _start, _end: True,
        )
        is None
    )


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
