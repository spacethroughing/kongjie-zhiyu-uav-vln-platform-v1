import asyncio
import json
import math
from pathlib import Path

import pytest

from harness.config import REPO_ROOT, Settings, load_scenes
from harness.events import EventBus
from harness.bridge import MockVehicleAdapter
from harness.llm import MockProvider, ModelProvider
from harness.mission import MissionService
from harness.models import (
    BoundingBox,
    DetectionAssessment,
    RoutePoint,
    RunState,
    SearchMissionRequest,
    TERMINAL_STATES,
    Vec3,
)
from harness.simulator import SimulatorManager
from harness.store import Store


def test_topology_connector_bridges_a_distant_vlm_goal_with_safe_hops():
    route = {
        f"route-{index}": (
            RoutePoint(index=index, position=Vec3(x=0, y=index * 8, z=-5)),
            Vec3(x=0, y=index * 8, z=-5),
        )
        for index in range(5)
    }

    connector = MissionService._topology_connector(
        Vec3(x=0, y=-1, z=-5), "route-4", route, set(), max_edge_m=10
    )

    assert connector == [
        "route-1",
        "route-2",
        "route-3",
        "route-4",
    ]


def test_dialog_exploration_heading_uses_ned_north_then_east_convention():
    origin = Vec3(x=10, y=-4, z=-5)
    north = MissionService._exploration_target(origin, 0, 8)
    east = MissionService._exploration_target(origin, 90, 8)

    assert north.x == pytest.approx(18)
    assert north.y == pytest.approx(-4)
    assert east.x == pytest.approx(10)
    assert east.y == pytest.approx(4)
    assert east.z == origin.z


@pytest.mark.asyncio
async def test_mock_scene_completes_open_vocabulary_search(tmp_path: Path):
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
    profiles = load_scenes(settings.scenes_file)
    events = EventBus()
    store = Store(settings.data_dir / "test.sqlite3", settings.runs_dir)
    simulator = SimulatorManager(profiles, events)
    provider = MockProvider()
    missions = MissionService(settings, store, events, simulator, provider)
    await simulator.start("mock")
    assert isinstance(simulator.adapter, MockVehicleAdapter)
    simulator.adapter.position = Vec3(x=12, y=-8, z=0)
    simulator.adapter.target = Vec3(x=22, y=2, z=-5)
    plan = missions.create_plan(
        SearchMissionRequest(
            scene_id="mock", zone_id="mock-fixture", target_text="red cube", end_policy="auto_rth"
        )
    )
    run = await missions.approve(plan.id)
    for _ in range(200):
        current = store.get_run(run.id)
        assert current
        if current.state in TERMINAL_STATES:
            break
        await asyncio.sleep(0.03)
    assert current.state == RunState.SUCCEEDED, current.error
    assert current.home_position is not None
    assert current.home_position.x == pytest.approx(12)
    assert current.home_position.y == pytest.approx(-8)
    assert current.target_position is not None
    assert (Path(current.artifact_dir) / "report.json").is_file()
    assert (Path(current.artifact_dir) / "telemetry.jsonl").is_file()
    topology = json.loads(
        (Path(current.artifact_dir) / "topology_map.json").read_text(encoding="utf-8")
    )
    assert topology["stats"]["semantic_objects"] >= 1
    assert "\"topic\":\"map.semantic\"" in (
        Path(current.artifact_dir) / "events.jsonl"
    ).read_text(encoding="utf-8")
    assert "\"topic\":\"lidar.points\"" in (
        Path(current.artifact_dir) / "events.jsonl"
    ).read_text(encoding="utf-8")
    assert "\"topic\":\"search.topology_vlm_plan\"" in (
        Path(current.artifact_dir) / "events.jsonl"
    ).read_text(encoding="utf-8")
    assert "\"topic\":\"run.home\"" in (
        Path(current.artifact_dir) / "events.jsonl"
    ).read_text(encoding="utf-8")
    await missions.close()
    await simulator.close()
    store.close()


async def _run_compound_request(tmp_path: Path, request: SearchMissionRequest):
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
    store = Store(settings.data_dir / "compound.sqlite3", settings.runs_dir)
    simulator = SimulatorManager(load_scenes(settings.scenes_file), events)
    missions = MissionService(settings, store, events, simulator, MockProvider())
    await simulator.start("mock")
    plan = missions.create_plan(request)
    run = await missions.approve(plan.id)
    current = run
    for _ in range(800):
        current = store.get_run(run.id)
        assert current
        if current.state in TERMINAL_STATES:
            break
        await asyncio.sleep(0.02)
    event_path = Path(current.artifact_dir) / "events.jsonl"
    event_text = event_path.read_text(encoding="utf-8") if event_path.exists() else ""
    await missions.close()
    await simulator.close()
    store.close()
    return current, event_text


@pytest.mark.asyncio
async def test_multi_target_mission_searches_each_target_in_one_run(tmp_path: Path):
    current, events = await _run_compound_request(
        tmp_path,
        SearchMissionRequest(
            scene_id="mock",
            zone_id="mock-fixture",
            target_text="圆锥体、橙色球体",
            targets=["圆锥体", "橙色球体"],
            end_policy="auto_rth",
        ),
    )
    assert current.state == RunState.SUCCEEDED, current.error
    assert [item.state for item in current.task_progress] == ["succeeded", "succeeded"]
    assert "圆锥体" in events
    assert "橙色球体" in events
    assert events.count('"topic":"mission.task_started"') == 2


@pytest.mark.asyncio
async def test_semantic_mapping_mission_completes_region_coverage(tmp_path: Path):
    current, events = await _run_compound_request(
        tmp_path,
        SearchMissionRequest(
            scene_id="mock",
            zone_id="mock-fixture",
            target_text="区域占据与语义拓扑图",
            mission_mode="semantic_mapping",
            mapping_coverage_target=0.85,
            end_policy="auto_rth",
        ),
    )
    assert current.state == RunState.SUCCEEDED, current.error
    assert current.mapping_coverage_ratio == pytest.approx(1.0)
    assert current.task_progress[0].state == "succeeded"
    assert '"topic":"mapping.completed"' in events
    topology = Path(current.artifact_dir) / "topology_map.json"
    assert topology.is_file()


@pytest.mark.asyncio
async def test_semantic_mapping_consumes_queued_altitude_action_chunk(tmp_path: Path):
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
    store = Store(settings.data_dir / "mapping-altitude.sqlite3", settings.runs_dir)
    simulator = SimulatorManager(load_scenes(settings.scenes_file), events)
    missions = MissionService(settings, store, events, simulator, MockProvider())
    await simulator.start("mock")
    assert simulator.adapter
    await simulator.adapter.close()
    adapter = SlowMoveAdapter()
    simulator.adapter = adapter
    plan = missions.create_plan(
        SearchMissionRequest(
            scene_id="mock",
            zone_id="mock-fixture",
            target_text="region occupancy and semantic topology map",
            mission_mode="semantic_mapping",
            mapping_coverage_target=0.85,
        )
    )
    run = await missions.approve(plan.id)
    for _ in range(300):
        current = store.get_run(run.id)
        assert current
        control = missions._controls[run.id]
        if current.state == RunState.SEARCHING and control.home_position:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("mapping mission did not enter SEARCHING")

    altitude = await missions.queue_altitude(run.id, target_altitude_m=7)
    assert altitude.target_altitude_m == pytest.approx(7)
    event_path = Path(current.artifact_dir) / "events.jsonl"
    for _ in range(500):
        if event_path.exists() and "vlm.altitude_completed" in event_path.read_text(
            encoding="utf-8"
        ):
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("mapping mission did not consume queued altitude control")

    assert any(
        operation["operation"] == "move_to_z"
        and float(operation["z"]) == pytest.approx(-7, abs=0.8)
        for operation in adapter.operations
    )
    assert missions._controls[run.id].altitude_override_m == pytest.approx(7)
    active = store.get_run(run.id)
    assert active
    if active.state not in TERMINAL_STATES:
        await missions.control(run.id, "abort")
    await missions.close()
    await simulator.close()
    store.close()


class FailingProvider(ModelProvider):
    name = "failing"

    async def inspect(self, frame, target_text, telemetry, observation_index):
        raise ValueError("malformed model output")


class InconsistentCenteredProvider(ModelProvider):
    name = "inconsistent-centered"

    def __init__(self):
        self.observations = []

    async def inspect(self, frame, target_text, telemetry, observation_index):
        self.observations.append(observation_index)
        boxes = {
            2: BoundingBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),
            3: BoundingBox(x_min=0.8, y_min=0.4, x_max=0.98, y_max=0.6),
        }
        box = boxes.get(observation_index)
        return DetectionAssessment(
            frame_id=frame.frame_id,
            is_match=box is not None,
            confidence=0.95 if box else 0.1,
            bbox_norm=box,
            evidence="synthetic inconsistent centered lock",
        )


class RecordingMockProvider(MockProvider):
    def __init__(self):
        self.observations = []

    async def inspect(self, frame, target_text, telemetry, observation_index):
        self.observations.append(observation_index)
        return await super().inspect(frame, target_text, telemetry, observation_index)


class OcclusionRecoveryProvider(ModelProvider):
    name = "occlusion-recovery"

    async def inspect(self, frame, target_text, telemetry, observation_index):
        # Model latency gives the deterministic planner time to mark the first
        # blocked corridor as an expected camera occlusion.
        await asyncio.sleep(0.01)
        visible = observation_index not in {1, 4, 5, 6}
        return DetectionAssessment(
            frame_id=frame.frame_id,
            is_match=visible,
            confidence=0.95 if visible else 0.0,
            bbox_norm=(
                BoundingBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6)
                if visible
                else None
            ),
            evidence="target temporarily hidden by planned detour",
        )


class ClippedThenVisibleProvider(ModelProvider):
    name = "clipped-then-visible"

    async def inspect(self, frame, target_text, telemetry, observation_index):
        box = (
            BoundingBox(x_min=0.93, y_min=0.7, x_max=1.0, y_max=0.8)
            if observation_index == 1
            else BoundingBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6)
        )
        return DetectionAssessment(
            frame_id=frame.frame_id,
            is_match=True,
            confidence=0.95,
            bbox_norm=box,
            evidence="first box is clipped, later target is fully visible",
        )


class PositionGatedSearchProvider(ModelProvider):
    """Miss the panorama, then lock only after the first search chunk moves."""

    name = "position-gated-search"

    async def inspect(self, frame, target_text, telemetry, observation_index):
        visible = math.hypot(telemetry.position.x, telemetry.position.y) > 0.5
        return DetectionAssessment(
            frame_id=frame.frame_id,
            is_match=visible,
            confidence=0.95 if visible else 0.1,
            bbox_norm=(
                BoundingBox(x_min=0.4, y_min=0.35, x_max=0.6, y_max=0.65)
                if visible
                else None
            ),
            evidence="target becomes visible from the first translated viewpoint",
        )


class NeverMatchProvider(ModelProvider):
    name = "never-match"

    async def inspect(self, frame, target_text, telemetry, observation_index):
        await asyncio.sleep(0.01)
        return DetectionAssessment(
            frame_id=frame.frame_id,
            is_match=False,
            confidence=0.1,
            bbox_norm=None,
            evidence="keep exploring",
        )


class NoDepthAdapter(MockVehicleAdapter):
    async def request(self, operation: str, **arguments):
        payload = await super().request(operation, **arguments)
        if operation == "capture":
            payload["depth_f32_zlib_b64"] = None
        return payload


class CollisionAdapter(MockVehicleAdapter):
    async def request(self, operation: str, **arguments):
        payload = await super().request(operation, **arguments)
        if operation == "state" and not self.landed:
            payload["collision"] = True
        return payload


class EnclosedLidarAdapter(MockVehicleAdapter):
    async def request(self, operation: str, **arguments):
        if operation == "lidar_scan":
            points = []
            for index in range(36):
                angle = index * math.tau / 36
                points.extend(
                    [
                        self.position.x + math.cos(angle) * 0.5,
                        self.position.y + math.sin(angle) * 0.5,
                        self.position.z,
                    ]
                )
            # The fixture represents a truly enclosed volume, including the
            # vertical escape direction introduced by the local planner.
            points.extend(
                [
                    self.position.x,
                    self.position.y,
                    self.position.z - 0.5,
                    self.position.x,
                    self.position.y,
                    self.position.z + 0.5,
                ]
            )
            return {
                "point_cloud": points,
                "point_count": 38,
                "sampled_point_count": 38,
                "data_frame": "VehicleInertialFrame",
            }
        return await super().request(operation, **arguments)


class GeofenceDriftAdapter(MockVehicleAdapter):
    async def request(self, operation: str, **arguments):
        payload = await super().request(operation, **arguments)
        if operation == "move_to" and float(arguments.get("speed", 0)) >= 3:
            self.position = self.position.model_copy(update={"x": 31.0})
        return payload


class LowTakeoffAdapter(MockVehicleAdapter):
    async def request(self, operation: str, **arguments):
        if operation == "takeoff":
            self.landed = False
            self.position = self.position.model_copy(update={"z": -0.8})
            return {"accepted": True}
        return await super().request(operation, **arguments)


class LooseCruiseAccuracyAdapter(MockVehicleAdapter):
    async def request(self, operation: str, **arguments):
        if operation == "move_to" and float(arguments.get("speed", 0)) >= 2.9:
            target = self.position.model_copy(
                update={
                    "x": float(arguments["x"]),
                    "y": float(arguments["y"]),
                    "z": float(arguments["z"]),
                }
            )
            dx = target.x - self.position.x
            dy = target.y - self.position.y
            dz = target.z - self.position.z
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length > 1.5:
                scale = (length - 1.2) / length
                self.position = self.position.model_copy(
                    update={
                        "x": self.position.x + dx * scale,
                        "y": self.position.y + dy * scale,
                        "z": self.position.z + dz * scale,
                    }
                )
                self.landed = False
                return {"accepted": True}
        return await super().request(operation, **arguments)


class GroundContactBeforeLandedAdapter(MockVehicleAdapter):
    def __init__(self):
        super().__init__()
        self.ground_contact = False

    async def request(self, operation: str, **arguments):
        if operation == "land":
            self.position = self.position.model_copy(update={"z": 0})
            self.landed = False
            self.ground_contact = True
            return {"accepted": True}
        payload = await super().request(operation, **arguments)
        if operation == "state" and self.ground_contact and abs(self.position.z) <= 0.1:
            payload["landed"] = False
            payload["collision"] = True
        return payload


class HeadingRecordingAdapter(MockVehicleAdapter):
    def __init__(self):
        super().__init__()
        self.target_heading_moves = []
        self.operations = []

    async def request(self, operation: str, **arguments):
        self.operations.append({"operation": operation, **dict(arguments)})
        if operation == "move_to" and "yaw_degrees" in arguments:
            self.target_heading_moves.append(
                {"origin": self.position.model_dump(), **dict(arguments)}
            )
        return await super().request(operation, **arguments)


class SlowMoveAdapter(HeadingRecordingAdapter):
    async def request(self, operation: str, **arguments):
        if operation == "move_to":
            await asyncio.sleep(0.05)
        return await super().request(operation, **arguments)


class OneDetourAdapter(SlowMoveAdapter):
    def __init__(self):
        super().__init__()
        self.lidar_scans = 0

    async def request(self, operation: str, **arguments):
        if operation == "lidar_scan":
            self.lidar_scans += 1
            if self.lidar_scans == 1:
                dx = self.target.x - self.position.x
                dy = self.target.y - self.position.y
                length = math.hypot(dx, dy)
                center_x = self.position.x + dx / length * 2.0
                center_y = self.position.y + dy / length * 2.0
                points = [center_x, center_y, self.position.z]
                return {
                    "point_cloud": points,
                    "point_count": 1,
                    "sampled_point_count": 1,
                    "data_frame": "VehicleInertialFrame",
                }
        return await super().request(operation, **arguments)


async def run_case(
    tmp_path: Path,
    provider: ModelProvider,
    adapter=None,
    panorama=False,
    end_policy="auto_rth",
):
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        scenes_file=REPO_ROOT / "configs" / "scenes.json",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        provider=provider.name,
        llm_base_url="",
        llm_model="",
        llm_api_key="",
        llm_timeout_seconds=10,
    )
    events = EventBus()
    store = Store(settings.data_dir / "test.sqlite3", settings.runs_dir)
    simulator = SimulatorManager(load_scenes(settings.scenes_file), events)
    if panorama:
        simulator.profiles["mock"].zones[0].initial_panorama_yaws_deg = [0, 45, 90]
    missions = MissionService(settings, store, events, simulator, provider)
    await simulator.start("mock")
    if adapter is not None:
        assert simulator.adapter
        await simulator.adapter.close()
        simulator.adapter = adapter
    plan = missions.create_plan(
        SearchMissionRequest(
            scene_id="mock",
            zone_id="mock-fixture",
            target_text="red cube",
            end_policy=end_policy,
        )
    )
    run = await missions.approve(plan.id)
    current = run
    for _ in range(400):
        current = store.get_run(run.id)
        assert current
        if current.state in TERMINAL_STATES:
            break
        await asyncio.sleep(0.02)
    event_path = Path(current.artifact_dir) / "events.jsonl"
    event_states = []
    if event_path.exists():
        event_states = [
            json.loads(line)["payload"].get("state")
            for line in event_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["topic"] == "run.state"
        ]
    await missions.close()
    await simulator.close()
    store.close()
    return current, event_states


@pytest.mark.asyncio
async def test_three_model_failures_enter_safe_hold(tmp_path: Path):
    current, states = await run_case(tmp_path, FailingProvider())
    assert current.state == RunState.FAILED
    assert "three consecutive model calls failed" in (current.error or "")
    assert RunState.SAFE_HOLD.value in states


@pytest.mark.asyncio
async def test_missing_depth_never_approaches_and_returns_not_found(tmp_path: Path):
    current, states = await run_case(tmp_path, MockProvider(), NoDepthAdapter())
    assert current.state == RunState.NOT_FOUND
    assert RunState.APPROACHING.value not in states


@pytest.mark.asyncio
async def test_collision_enters_safe_hold(tmp_path: Path):
    current, states = await run_case(tmp_path, MockProvider(), CollisionAdapter())
    assert current.state == RunState.FAILED
    assert "collision" in (current.error or "").lower()
    assert RunState.SAFE_HOLD.value in states


@pytest.mark.asyncio
async def test_occupied_search_waypoints_are_skipped_without_unsafe_motion(
    tmp_path: Path,
):
    current, states = await run_case(tmp_path, MockProvider(), EnclosedLidarAdapter())
    assert current.state == RunState.NOT_FOUND
    assert RunState.APPROACHING.value not in states
    assert RunState.SAFE_HOLD.value not in states
    events = (Path(current.artifact_dir) / "events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "search.waypoint_skipped" in events


@pytest.mark.asyncio
async def test_actual_position_drift_outside_geofence_enters_safe_hold(tmp_path: Path):
    current, states = await run_case(tmp_path, MockProvider(), GeofenceDriftAdapter())
    assert current.state == RunState.FAILED
    assert "outside the configured search geofence" in (current.error or "")
    assert RunState.SAFE_HOLD.value in states


@pytest.mark.asyncio
async def test_low_takeoff_climbs_vertically_before_search(tmp_path: Path):
    current, states = await run_case(tmp_path, MockProvider(), LowTakeoffAdapter())
    assert current.state == RunState.SUCCEEDED, current.error
    assert RunState.SEARCHING.value in states


@pytest.mark.asyncio
async def test_short_segments_accept_simpleflight_completion_radius(tmp_path: Path):
    current, states = await run_case(tmp_path, MockProvider(), LooseCruiseAccuracyAdapter())
    assert current.state == RunState.SUCCEEDED, current.error
    assert RunState.SEARCHING.value in states


@pytest.mark.asyncio
async def test_initial_panorama_locks_then_approaches_with_target_heading(tmp_path: Path):
    adapter = HeadingRecordingAdapter()
    current, states = await run_case(tmp_path, MockProvider(), adapter=adapter, panorama=True)
    assert current.state == RunState.SUCCEEDED, current.error
    assert RunState.VERIFYING.value not in states
    assert current.target_position is not None
    assert adapter.target_heading_moves
    approach_moves = [
        move for move in adapter.target_heading_moves if float(move["speed"]) == 4.0
    ]
    assert approach_moves
    for move in approach_moves:
        expected_yaw = math.degrees(
            math.atan2(
                current.target_position.y - move["origin"]["y"],
                current.target_position.x - move["origin"]["x"],
            )
        )
        angular_error = abs((move["yaw_degrees"] - expected_yaw + 180) % 360 - 180)
        assert angular_error < 1e-6
    final_move = approach_moves[-1]
    assert math.hypot(
        final_move["x"] - current.target_position.x,
        final_move["y"] - current.target_position.y,
    ) == pytest.approx(3.0)
    approach_operation_indexes = [
        index
        for index, operation in enumerate(adapter.operations)
        if operation["operation"] == "move_to"
        and float(operation["speed"]) == 4.0
    ]
    hover_indexes = [
        index
        for index, operation in enumerate(adapter.operations)
        if operation["operation"] == "hover"
    ]
    assert hover_indexes
    assert min(hover_indexes) > max(approach_operation_indexes)
    events = [
        json.loads(line)
        for line in (Path(current.artifact_dir) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not any(
        event["topic"] == "flight.phase"
        and event["payload"].get("phase") == "locked_baseline"
        for event in events
    )


@pytest.mark.asyncio
async def test_panorama_miss_uses_async_vlm_search_and_interrupts_the_path(
    tmp_path: Path,
):
    adapter = HeadingRecordingAdapter()
    current, states = await run_case(
        tmp_path,
        PositionGatedSearchProvider(),
        adapter=adapter,
        panorama=True,
    )
    assert current.state == RunState.SUCCEEDED, current.error
    assert RunState.APPROACHING.value in states
    events = [
        json.loads(line)
        for line in (Path(current.artifact_dir) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(event["topic"] == "search.path_started" for event in events)
    assert any(event["topic"] == "search.action_chunk" for event in events)
    acquired = next(
        event for event in events if event["topic"] == "search.target_acquired"
    )
    assert acquired["payload"]["action_chunks"] >= 1
    assert not any(
        event["topic"] == "search.path_completed"
        and event["payload"].get("result") == "not_found"
        for event in events
    )


@pytest.mark.asyncio
async def test_queued_dialog_exploration_executes_with_heading_then_replans(tmp_path: Path):
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        scenes_file=REPO_ROOT / "configs" / "scenes.json",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        provider="never-match",
        llm_base_url="",
        llm_model="",
        llm_api_key="",
        llm_timeout_seconds=10,
    )
    events = EventBus()
    store = Store(settings.data_dir / "test.sqlite3", settings.runs_dir)
    simulator = SimulatorManager(load_scenes(settings.scenes_file), events)
    missions = MissionService(
        settings, store, events, simulator, NeverMatchProvider()
    )
    await simulator.start("mock")
    assert simulator.adapter
    await simulator.adapter.close()
    adapter = SlowMoveAdapter()
    simulator.adapter = adapter
    plan = missions.create_plan(
        SearchMissionRequest(
            scene_id="mock",
            zone_id="mock-fixture",
            target_text="orange cone",
        )
    )
    run = await missions.approve(plan.id)
    for _ in range(200):
        current = store.get_run(run.id)
        assert current
        control = missions._controls[run.id]
        if current.state == RunState.SEARCHING and control.home_position:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("mission did not enter SEARCHING")

    directive = await missions.queue_exploration(run.id, 90, 4)
    assert directive.heading_degrees == 90
    event_path = Path(current.artifact_dir) / "events.jsonl"
    for _ in range(400):
        if event_path.exists() and "vlm.exploration_completed" in event_path.read_text(
            encoding="utf-8"
        ):
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("queued exploration was not completed")

    exploration_moves = [
        move
        for move in adapter.target_heading_moves
        if abs((float(move["yaw_degrees"]) - 90 + 180) % 360 - 180) < 1e-6
    ]
    assert exploration_moves
    active = store.get_run(run.id)
    assert active and active.state == RunState.SEARCHING
    altitude = await missions.queue_altitude(run.id, target_altitude_m=7)
    assert altitude.target_altitude_m == pytest.approx(7)
    for _ in range(400):
        if "vlm.altitude_completed" in event_path.read_text(encoding="utf-8"):
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("queued altitude control was not completed")
    altitude_moves = [
        move
        for move in adapter.operations
        if move["operation"] == "move_to_z"
        if float(move["z"]) == pytest.approx(-7, abs=0.8)
    ]
    assert altitude_moves
    assert missions._controls[run.id].altitude_override_m == pytest.approx(7)
    active = store.get_run(run.id)
    assert active
    if active.state not in TERMINAL_STATES:
        await missions.control(run.id, "abort")
    await missions.close()
    await simulator.close()
    store.close()


@pytest.mark.asyncio
async def test_land_at_target_skips_rth_and_lands_at_standoff(tmp_path: Path):
    adapter = HeadingRecordingAdapter()
    current, states = await run_case(
        tmp_path,
        MockProvider(),
        adapter=adapter,
        panorama=True,
        end_policy="land_at_target",
    )
    assert current.state == RunState.SUCCEEDED, current.error
    assert RunState.EVIDENCE.value in states
    assert RunState.LANDING.value in states
    assert RunState.RTH.value not in states
    assert adapter.landed is True
    assert math.hypot(adapter.position.x, adapter.position.y) > 5.0


@pytest.mark.asyncio
async def test_single_frame_lock_skips_centering_and_second_depth_check(tmp_path: Path):
    provider = RecordingMockProvider()
    current, states = await run_case(tmp_path, provider, panorama=True)
    assert current.state == RunState.SUCCEEDED, current.error
    assert RunState.APPROACHING.value in states
    assert len(provider.observations) >= 3
    assert provider.observations == list(range(1, len(provider.observations) + 1))
    events = (Path(current.artifact_dir) / "events.jsonl").read_text(encoding="utf-8")
    assert "locked_centering" not in events
    assert '"mode":"single_frame_depth"' in events
    assert "guidance.target_updated" in events


@pytest.mark.asyncio
async def test_continuous_guidance_fails_closed_after_three_lost_frames(tmp_path: Path):
    provider = InconsistentCenteredProvider()
    current, states = await run_case(
        tmp_path, provider, adapter=SlowMoveAdapter(), panorama=True
    )
    assert current.state == RunState.FAILED
    assert "VLM guidance lost the target" in (current.error or "")
    assert RunState.SAFE_HOLD.value in states
    assert provider.observations[:2] == [1, 2]
    assert len(provider.observations) >= 5
    events = [
        json.loads(line)
        for line in (Path(current.artifact_dir) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failed_sequence = next(
        event["sequence"]
        for event in events
        if event["topic"] == "guidance.stream_failed"
    )
    assert not any(
        event["topic"] == "action_chunk.completed"
        and event["sequence"] > failed_sequence
        for event in events
    )


@pytest.mark.asyncio
async def test_expected_obstacle_occlusion_retains_lock_and_recovers(tmp_path: Path):
    current, states = await run_case(
        tmp_path,
        OcclusionRecoveryProvider(),
        adapter=OneDetourAdapter(),
        panorama=True,
    )
    assert current.state == RunState.SUCCEEDED, current.error
    events = [
        json.loads(line)
        for line in (Path(current.artifact_dir) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(event["topic"] == "guidance.target_occluded" for event in events)
    assert not any(event["topic"] == "guidance.stream_failed" for event in events)


@pytest.mark.asyncio
async def test_initial_panorama_rejects_edge_clipped_box_without_centering_gate(
    tmp_path: Path,
):
    current, _ = await run_case(
        tmp_path,
        ClippedThenVisibleProvider(),
        panorama=True,
    )
    assert current.state == RunState.SUCCEEDED, current.error
    events = [
        json.loads(line)
        for line in (Path(current.artifact_dir) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    clipped = [
        event
        for event in events
        if event["topic"] == "vision.rejected"
        and "clipped" in event["payload"].get("reason", "")
    ]
    assert clipped
    locked = next(event for event in events if event["topic"] == "vision.locked")
    assert locked["payload"]["bbox_norm"] == {
        "x_min": 0.4,
        "y_min": 0.4,
        "x_max": 0.6,
        "y_max": 0.6,
    }


@pytest.mark.asyncio
async def test_ground_contact_during_landing_is_not_an_airborne_collision(tmp_path: Path):
    current, _ = await run_case(tmp_path, MockProvider(), GroundContactBeforeLandedAdapter())
    assert current.state == RunState.SUCCEEDED, current.error
