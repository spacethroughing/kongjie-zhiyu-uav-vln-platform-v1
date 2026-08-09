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
from harness.models import BoundingBox, DetectionAssessment, RunState, SearchMissionRequest, TERMINAL_STATES
from harness.simulator import SimulatorManager
from harness.store import Store


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
    assert current.target_position is not None
    assert (Path(current.artifact_dir) / "report.json").is_file()
    assert (Path(current.artifact_dir) / "telemetry.jsonl").is_file()
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
            return {
                "point_cloud": points,
                "point_count": 36,
                "sampled_point_count": 36,
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


async def run_case(tmp_path: Path, provider: ModelProvider, adapter=None, panorama=False):
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
            scene_id="mock", zone_id="mock-fixture", target_text="red cube", end_policy="auto_rth"
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
async def test_occupied_lidar_corridor_fails_closed_without_a_detour(tmp_path: Path):
    current, states = await run_case(tmp_path, MockProvider(), EnclosedLidarAdapter())
    assert current.state == RunState.FAILED
    assert "no safe local detour" in (current.error or "")
    assert RunState.SAFE_HOLD.value in states


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


@pytest.mark.asyncio
async def test_ground_contact_during_landing_is_not_an_airborne_collision(tmp_path: Path):
    current, _ = await run_case(tmp_path, MockProvider(), GroundContactBeforeLandedAdapter())
    assert current.state == RunState.SUCCEEDED, current.error
