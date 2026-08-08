from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .events import EventBus
from .geometry import DepthLocalizationError, distance, localize_bbox
from .llm import ModelProvider
from .models import (
    CameraFrame,
    DetectionAssessment,
    MissionPlan,
    RunRecord,
    RunState,
    SearchMissionRequest,
    SearchZone,
    Telemetry,
    TERMINAL_STATES,
    Vec3,
    utc_now,
)
from .planner import build_plan
from .safety import SafetyViolation, approach_point, point_in_polygon, validate_position, validate_telemetry
from .simulator import SimulatorManager
from .store import Store


class MissionError(RuntimeError):
    pass


class MissionAborted(MissionError):
    pass


class LandRequested(MissionError):
    pass


@dataclass
class RunControl:
    paused: asyncio.Event = field(default_factory=asyncio.Event)
    abort: asyncio.Event = field(default_factory=asyncio.Event)
    return_home: asyncio.Event = field(default_factory=asyncio.Event)
    land: asyncio.Event = field(default_factory=asyncio.Event)
    candidate_decision: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    home_position: Vec3 | None = None
    deadline_monotonic: float | None = None


class MissionService:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        events: EventBus,
        simulator: SimulatorManager,
        provider: ModelProvider,
    ) -> None:
        self.settings = settings
        self.store = store
        self.events = events
        self.simulator = simulator
        self.provider = provider
        self._tasks: dict[str, asyncio.Task] = {}
        self._controls: dict[str, RunControl] = {}

    def create_plan(self, request: SearchMissionRequest) -> MissionPlan:
        try:
            profile = self.simulator.profiles[request.scene_id]
        except KeyError as error:
            raise MissionError(f"unknown scene: {request.scene_id}") from error
        plan = build_plan(profile, request)
        self.store.save_plan(plan)
        return plan

    async def approve(self, plan_id: str) -> RunRecord:
        plan = self.store.get_plan(plan_id)
        if not plan:
            raise MissionError("mission plan does not exist")
        if self.simulator.state != "READY" or not self.simulator.active_profile:
            raise MissionError("simulator is not ready")
        if self.simulator.active_profile.id != plan.request.scene_id:
            raise MissionError("active simulator scene does not match the mission plan")
        if plan.approved_at:
            raise MissionError("mission plan is already approved")
        plan = plan.model_copy(update={"approved_at": utc_now()})
        self.store.save_plan(plan)
        run = self.store.create_run(plan.id)
        self._controls[run.id] = RunControl(
            deadline_monotonic=asyncio.get_running_loop().time() + plan.safety.max_mission_seconds
        )
        self.store.write_manifest(run, self._manifest(plan))
        self._tasks[run.id] = asyncio.create_task(self._execute(run, plan), name=f"mission-{run.id}")
        await self._event("run.created", {"run": run.model_dump(mode="json")}, run.id)
        return run

    def _manifest(self, plan: MissionPlan) -> dict:
        profile = self.simulator.profiles[plan.request.scene_id]
        manifest = {
            "harness_version": "0.1.0",
            "scene": profile.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "provider": {"name": self.provider.name, "model": self.settings.llm_model or "mock"},
            "environment": {"python": os.sys.version, "platform": os.name},
        }
        fingerprint_paths = [profile.project, profile.settings, *profile.checksum_paths]
        manifest["file_fingerprints"] = [
            self._fingerprint(Path(path)) for path in fingerprint_paths if path
        ]
        if profile.executable:
            ue_build = Path(profile.executable).parents[2] / "Build" / "Build.version"
            if ue_build.is_file():
                manifest["ue_build"] = json.loads(ue_build.read_text(encoding="utf-8"))
        if profile.project:
            plugin_file = Path(profile.project).parent / "Plugins" / "AirSim" / "AirSim.uplugin"
            if plugin_file.is_file():
                plugin = json.loads(plugin_file.read_text(encoding="utf-8"))
                manifest["airsim_plugin"] = {
                    "version": plugin.get("VersionName") or plugin.get("Version"),
                    "sha256": self._fingerprint(plugin_file)["sha256"],
                }
        airsim_repo = Path(r"E:\C\AirSim")
        if (airsim_repo / ".git").exists():
            try:
                manifest["airsim_commit"] = subprocess.run(
                    ["git", "-C", str(airsim_repo), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            except Exception:
                manifest["airsim_commit"] = "unavailable"
        return manifest

    @staticmethod
    def _fingerprint(path: Path) -> dict:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        stat = path.stat()
        return {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }

    async def _event(self, topic: str, payload: dict, run_id: str | None = None) -> None:
        event = await self.events.publish(topic, payload, run_id)
        self.store.append_event(event)

    async def _set_state(
        self,
        run: RunRecord,
        state: RunState,
        *,
        error: str | None = None,
        ended: bool = False,
        target_position: Vec3 | None = None,
    ) -> RunRecord:
        run = self.store.update_run(
            run, state, error=error, ended=ended, target_position=target_position
        )
        await self._event("run.state", {"state": state.value, "error": error}, run.id)
        return run

    def _zone(self, plan: MissionPlan) -> SearchZone:
        profile = self.simulator.profiles[plan.request.scene_id]
        return next(zone for zone in profile.zones if zone.id == plan.request.zone_id)

    async def _execute(self, run: RunRecord, plan: MissionPlan) -> None:
        adapter = self.simulator.adapter
        profile = self.simulator.active_profile
        if not adapter or not profile:
            return
        control = self._controls[run.id]
        zone = self._zone(plan)
        found = False
        candidate_positions: list[tuple[Vec3, Vec3]] = []
        observation_index = 0
        consecutive_model_failures = 0
        try:
            home_telemetry = await self._telemetry(run)
            home = home_telemetry.position
            control.home_position = home
            await adapter.request("api_control", vehicle_name=profile.vehicle_name, enabled=True)
            await adapter.request("arm", vehicle_name=profile.vehicle_name, armed=True)
            run = await self._set_state(run, RunState.TAKEOFF)
            for attempt in range(1, 4):
                await adapter.request("takeoff", vehicle_name=profile.vehicle_name, timeout=5)
                try:
                    await self._wait_for_altitude(
                        run, home.z, abs(plan.route[0].position.z), control, timeout=12
                    )
                    break
                except MissionError:
                    if attempt == 3:
                        raise
                    await adapter.request("cancel", vehicle_name=profile.vehicle_name, timeout=3)
                    await adapter.request(
                        "arm", vehicle_name=profile.vehicle_name, armed=False, timeout=3
                    )
                    await asyncio.sleep(0.5)
                    await adapter.request(
                        "arm", vehicle_name=profile.vehicle_name, armed=True, timeout=3
                    )
            await self._climb_to_search_altitude(run, plan, zone, home, control)
            run = await self._set_state(run, RunState.SEARCHING)

            confirmed, observation_index = await self._initial_panorama(
                run,
                plan,
                zone,
                home,
                control,
                candidate_positions,
                observation_index,
            )
            if confirmed:
                run = await self._set_state(run, RunState.VERIFYING, target_position=confirmed)
                accepted = await self._approach_and_review(
                    run, plan, zone, confirmed, home, control
                )
                if accepted:
                    found = True
                else:
                    candidate_positions.clear()
                    run = self.store.get_run(run.id) or run
                    run = await self._set_state(run, RunState.SEARCHING)

            for route_point in ([] if found else plan.route):
                directive = await self._handle_control(run, control)
                if directive == "return_home":
                    break
                world_point = self._from_home(route_point.position, home)
                await self._move_segmented(run, plan, zone, world_point, home, control)
                for yaw in plan.observation_yaws_deg:
                    directive = await self._handle_control(run, control)
                    if directive == "return_home":
                        break
                    await adapter.request(
                        "rotate_yaw", vehicle_name=profile.vehicle_name, yaw_degrees=yaw, timeout=10
                    )
                    await asyncio.sleep(0.05 if profile.mode == "mock" else 0.7)
                    observation_index += 1
                    try:
                        frame, telemetry, assessment = await self._capture_assessment(
                            run, plan, observation_index
                        )
                        consecutive_model_failures = 0
                    except Exception as error:
                        consecutive_model_failures += 1
                        await self._event(
                            "model.error",
                            {
                                "consecutive_failures": consecutive_model_failures,
                                "error": f"{type(error).__name__}: {error}",
                            },
                            run.id,
                        )
                        if consecutive_model_failures >= 3:
                            raise MissionError("three consecutive model calls failed") from error
                        continue
                    confirmed = await self._evaluate_candidate(
                        run, plan, zone, frame, telemetry, assessment, candidate_positions, home
                    )
                    if confirmed:
                        run = await self._set_state(
                            run, RunState.VERIFYING, target_position=confirmed
                        )
                        accepted = await self._approach_and_review(
                            run, plan, zone, confirmed, home, control
                        )
                        if accepted:
                            found = True
                            break
                        candidate_positions.clear()
                        run = self.store.get_run(run.id) or run
                        run = await self._set_state(run, RunState.SEARCHING)
                if found or control.return_home.is_set() or control.land.is_set():
                    break

            run = self.store.get_run(run.id) or run
            if control.land.is_set():
                raise LandRequested("operator requested immediate landing")
            run = await self._return_and_land(run, profile.vehicle_name, home)
            final_state = RunState.SUCCEEDED if found else RunState.NOT_FOUND
            run = await self._set_state(run, final_state, ended=True)
        except LandRequested:
            run = self.store.get_run(run.id) or run
            run = await self._set_state(run, RunState.LANDING)
            await adapter.request("cancel", timeout=3, vehicle_name=profile.vehicle_name)
            await adapter.request("land", timeout=3, vehicle_name=profile.vehicle_name)
            await self._wait_landed(
                run,
                control,
                timeout=45,
                ignore_controls=True,
                ground_z=control.home_position.z if control.home_position else None,
            )
            await adapter.request("arm", vehicle_name=profile.vehicle_name, armed=False, timeout=3)
            run = await self._set_state(run, RunState.ABORTED, error="operator requested landing", ended=True)
        except MissionAborted as error:
            run = self.store.get_run(run.id) or run
            run = await self._set_state(run, RunState.ABORTING, error=str(error))
            try:
                await adapter.request("cancel", timeout=3, vehicle_name=profile.vehicle_name)
                await adapter.request("hover", timeout=3, vehicle_name=profile.vehicle_name)
                await adapter.request("land", timeout=3, vehicle_name=profile.vehicle_name)
            finally:
                run = await self._set_state(run, RunState.ABORTED, error=str(error), ended=True)
        except Exception as error:
            run = self.store.get_run(run.id) or run
            run = await self._set_state(run, RunState.SAFE_HOLD, error=str(error))
            try:
                await adapter.request("cancel", timeout=2, vehicle_name=profile.vehicle_name)
                await adapter.request("hover", timeout=2, vehicle_name=profile.vehicle_name)
                await adapter.request("land", timeout=2, vehicle_name=profile.vehicle_name)
            except Exception:
                pass
            run = await self._set_state(run, RunState.FAILED, error=str(error), ended=True)
        finally:
            current = self.store.get_run(run.id) or run
            self.store.write_report(current, plan)
            self._tasks.pop(run.id, None)

    async def _capture_assessment(
        self, run: RunRecord, plan: MissionPlan, observation_index: int
    ) -> tuple[CameraFrame, Telemetry, DetectionAssessment]:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        frame = await adapter.capture(profile.vehicle_name)
        self._save_frame(run, frame)
        await self._event(
            "frame.preview",
            {
                "frame_id": frame.frame_id,
                "data_url": f"data:image/png;base64,{frame.scene_png_b64}",
                "width": frame.width,
                "height": frame.height,
            },
            run.id,
        )
        telemetry = await self._telemetry(run)
        assessment = await self.provider.inspect(
            frame, plan.request.target_text, telemetry, observation_index
        )
        self.store.append_jsonl(
            Path(run.artifact_dir) / "model_calls.jsonl",
            {
                "provider": self.provider.name,
                "model": self.settings.llm_model or "mock",
                "frame_id": frame.frame_id,
                "target_text": plan.request.target_text,
                "assessment": assessment.model_dump(mode="json"),
            },
        )
        await self._event("vision.assessment", assessment.model_dump(mode="json"), run.id)
        return frame, telemetry, assessment

    async def _initial_panorama(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
        candidates: list[tuple[Vec3, Vec3]],
        observation_index: int,
    ) -> tuple[Vec3 | None, int]:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        if not zone.initial_panorama_yaws_deg:
            return None, observation_index
        await self._event(
            "flight.phase",
            {
                "phase": "initial_panorama",
                "message": "原地环视 360°，VLM 发现有效目标框后立即锁定",
                "yaws_deg": zone.initial_panorama_yaws_deg,
            },
            run.id,
        )
        consecutive_failures = 0
        for yaw in zone.initial_panorama_yaws_deg:
            directive = await self._handle_control(run, control)
            if directive == "return_home":
                return None, observation_index
            await adapter.request(
                "rotate_yaw", vehicle_name=profile.vehicle_name, yaw_degrees=yaw, timeout=10
            )
            await asyncio.sleep(0.05 if profile.mode == "mock" else 0.7)
            observation_index += 1
            try:
                frame, telemetry, assessment = await self._capture_assessment(
                    run, plan, observation_index
                )
                consecutive_failures = 0
            except Exception as error:
                consecutive_failures += 1
                await self._event(
                    "model.error",
                    {
                        "consecutive_failures": consecutive_failures,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    run.id,
                )
                if consecutive_failures >= 3:
                    raise MissionError("three consecutive model calls failed") from error
                continue
            before = len(candidates)
            await self._evaluate_candidate(
                run, plan, zone, frame, telemetry, assessment, candidates, home
            )
            if len(candidates) == before:
                continue
            locked = candidates[-1][0]
            await self._event(
                "vision.locked",
                {
                    "target": locked.model_dump(),
                    "frame_id": frame.frame_id,
                    "bbox_norm": assessment.bbox_norm.model_dump()
                    if assessment.bbox_norm
                    else None,
                    "confidence": assessment.confidence,
                    "message": "VLM 目标框已锁定，执行短基线安全复核",
                },
                run.id,
            )
            return await self._verify_locked_candidate(
                run,
                plan,
                zone,
                home,
                control,
                candidates,
                locked,
                observation_index,
            )
        return None, observation_index

    async def _verify_locked_candidate(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
        candidates: list[tuple[Vec3, Vec3]],
        locked: Vec3,
        observation_index: int,
    ) -> tuple[Vec3 | None, int]:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        telemetry = await self._telemetry(run)
        center_yaw = math.degrees(
            math.atan2(
                locked.y - telemetry.position.y,
                locked.x - telemetry.position.x,
            )
        )
        await self._event(
            "flight.phase",
            {
                "phase": "locked_centering",
                "message": "目标已锁定，先转向使候选框居中并刷新深度坐标",
                "yaw_degrees": center_yaw,
            },
            run.id,
        )
        await adapter.request(
            "rotate_yaw",
            vehicle_name=profile.vehicle_name,
            yaw_degrees=center_yaw,
            timeout=10,
        )
        await asyncio.sleep(0.05 if profile.mode == "mock" else 0.7)
        observation_index += 1
        try:
            frame, telemetry, assessment = await self._capture_assessment(
                run, plan, observation_index
            )
            before = len(candidates)
            centered = await self._evaluate_candidate(
                run, plan, zone, frame, telemetry, assessment, candidates, home
            )
            if centered:
                locked = centered
            elif len(candidates) > before:
                locked = candidates[-1][0]
            await self._event(
                "vision.lock_refined",
                {
                    "target": locked.model_dump(),
                    "frame_id": frame.frame_id,
                    "bbox_norm": assessment.bbox_norm.model_dump()
                    if assessment.bbox_norm
                    else None,
                    "message": "目标已居中，使用刷新后的深度坐标执行短基线复核",
                },
                run.id,
            )
        except Exception as error:
            await self._event(
                "model.error",
                {"consecutive_failures": 1, "error": f"{type(error).__name__}: {error}"},
                run.id,
            )
            telemetry = await self._telemetry(run)
        dx = locked.x - telemetry.position.x
        dy = locked.y - telemetry.position.y
        horizontal = math.hypot(dx, dy)
        if horizontal < 1e-6:
            return None, observation_index
        baseline = zone.initial_verify_baseline_m
        offsets = [(-dy / horizontal * baseline, dx / horizontal * baseline)]
        offsets.append((-offsets[0][0], -offsets[0][1]))
        verify_point = None
        for offset_x, offset_y in offsets:
            candidate = Vec3(
                x=telemetry.position.x + offset_x,
                y=telemetry.position.y + offset_y,
                z=telemetry.position.z,
            )
            try:
                validate_position(self._to_home(candidate, home), zone, plan.safety)
            except SafetyViolation:
                continue
            verify_point = candidate
            break
        if verify_point is None:
            await self._event(
                "vision.rejected", {"reason": "no safe baseline verification point"}, run.id
            )
            return None, observation_index
        await self._event(
            "flight.phase",
            {
                "phase": "locked_baseline",
                "message": f"目标已锁定，横移 {baseline:.1f} m 获取第二视角",
            },
            run.id,
        )
        await self._move_segmented(
            run,
            plan,
            zone,
            verify_point,
            home,
            control,
            speed=min(2.0, plan.safety.max_speed_mps),
        )
        yaw = math.degrees(
            math.atan2(locked.y - verify_point.y, locked.x - verify_point.x)
        )
        await adapter.request(
            "rotate_yaw", vehicle_name=profile.vehicle_name, yaw_degrees=yaw, timeout=10
        )
        await asyncio.sleep(0.05 if profile.mode == "mock" else 0.7)
        observation_index += 1
        try:
            frame, telemetry, assessment = await self._capture_assessment(
                run, plan, observation_index
            )
        except Exception as error:
            await self._event(
                "model.error",
                {"consecutive_failures": 1, "error": f"{type(error).__name__}: {error}"},
                run.id,
            )
            return None, observation_index
        confirmed = await self._evaluate_candidate(
            run, plan, zone, frame, telemetry, assessment, candidates, home
        )
        if not confirmed:
            await self._event(
                "vision.rejected",
                {"reason": "locked target did not pass the second-view consistency check"},
                run.id,
            )
        return confirmed, observation_index

    async def _evaluate_candidate(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        frame: CameraFrame,
        telemetry: Telemetry,
        assessment: DetectionAssessment,
        candidates: list[tuple[Vec3, Vec3]],
        home: Vec3,
    ) -> Vec3 | None:
        if not assessment.is_match or assessment.confidence < 0.75 or not assessment.bbox_norm:
            return None
        try:
            target = localize_bbox(frame, assessment.bbox_norm)
        except DepthLocalizationError as error:
            await self._event("vision.rejected", {"reason": str(error)}, run.id)
            return None
        if not point_in_polygon(self._to_home(target, home), zone.polygon):
            await self._event("vision.rejected", {"reason": "target is outside the search zone"}, run.id)
            return None
        for previous_target, previous_camera in candidates:
            if distance(previous_camera, frame.camera_position) >= 0.5 and distance(previous_target, target) <= 3.0:
                average = Vec3(
                    x=(previous_target.x + target.x) / 2,
                    y=(previous_target.y + target.y) / 2,
                    z=(previous_target.z + target.z) / 2,
                )
                await self._event("vision.confirmed", {"target": average.model_dump()}, run.id)
                return average
        candidates.append((target, frame.camera_position))
        return None

    async def _approach_and_review(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        target: Vec3,
        home: Vec3,
        control: RunControl,
    ) -> bool:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        telemetry = await self._telemetry(run)
        approach = approach_point(
            telemetry.position, target, zone.search_altitude_m, plan.safety.min_standoff_m
        )
        approach = approach.model_copy(update={"z": home.z - abs(zone.search_altitude_m)})
        validate_position(self._to_home(approach, home), zone, plan.safety)
        lidar = await adapter.request(
            "lidar_min", timeout=4, vehicle_name=profile.vehicle_name, sensor_name="LidarSensor1"
        )
        minimum = lidar.get("minimum_m")
        if minimum is not None and minimum < plan.safety.min_clearance_m:
            raise SafetyViolation("lidar clearance is below the configured minimum")
        run = await self._set_state(run, RunState.APPROACHING, target_position=target)
        await self._move_segmented(
            run, plan, zone, approach, home, control, speed=plan.safety.approach_speed_mps
        )
        await adapter.request("hover", timeout=4, vehicle_name=profile.vehicle_name)
        evidence = await adapter.capture(profile.vehicle_name)
        self._save_frame(run, evidence, evidence=True)
        run = await self._set_state(run, RunState.EVIDENCE, target_position=target)
        if plan.request.end_policy == "auto_rth":
            return True
        await self._event(
            "candidate.review",
            {"timeout_seconds": 30, "default": "accept", "frame_id": evidence.frame_id},
            run.id,
        )
        try:
            decision = await asyncio.wait_for(control.candidate_decision.get(), timeout=30)
        except asyncio.TimeoutError:
            decision = "accept"
        return decision == "accept"

    async def _return_and_land(
        self, run: RunRecord, vehicle_name: str, home: Vec3
    ) -> RunRecord:
        adapter = self.simulator.adapter
        assert adapter
        telemetry = await self._telemetry(run)
        run = await self._set_state(run, RunState.RTH)
        await adapter.request(
            "move_to",
            vehicle_name=vehicle_name,
            x=home.x,
            y=home.y,
            z=telemetry.position.z,
            speed=2,
            timeout=60,
        )
        await self._wait_position(
            run,
            Vec3(x=home.x, y=home.y, z=telemetry.position.z),
            RunControl(),
            timeout=60,
        )
        run = await self._set_state(run, RunState.LANDING)
        await adapter.request("land", vehicle_name=vehicle_name, timeout=60)
        await self._wait_landed(
            run, RunControl(), timeout=60, ignore_controls=True, ground_z=home.z
        )
        await adapter.request("arm", vehicle_name=vehicle_name, armed=False, timeout=5)
        return run

    async def _move_segmented(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        target: Vec3,
        home: Vec3,
        control: RunControl,
        speed: float | None = None,
    ) -> None:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        telemetry = await self._telemetry(run)
        start = telemetry.position
        total = distance(start, target)
        # Segments must remain longer than SimpleFlight's automatic lookahead;
        # ~2 m commands at 3 m/s may resolve immediately without moving.
        # Four metres is still a short, bounded command while clearing that
        # dead zone reliably in UE4.27/AirSim 1.8.1.
        segments = max(1, math.ceil(total / 4.0))
        for index in range(1, segments + 1):
            await self._handle_control(run, control)
            ratio = index / segments
            point = Vec3(
                x=start.x + (target.x - start.x) * ratio,
                y=start.y + (target.y - start.y) * ratio,
                z=start.z + (target.z - start.z) * ratio,
            )
            validate_position(self._to_home(point, home), zone, plan.safety)
            await adapter.request(
                "move_to",
                vehicle_name=profile.vehicle_name,
                x=point.x,
                y=point.y,
                z=point.z,
                speed=min(speed or plan.safety.max_speed_mps, plan.safety.max_speed_mps),
                timeout=20,
            )
            # SimpleFlight can resolve a short moveToPosition future about
            # 1.2 m from its requested point at cruise speed. Intermediate
            # segments use a matching acceptance radius; every telemetry
            # sample is still checked against the full safety envelope.
            await self._wait_position(run, point, control, timeout=20, tolerance=1.5)

    async def _climb_to_search_altitude(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
    ) -> Telemetry:
        """Use a vertical takeoff corridor before enforcing the cruise altitude floor."""
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        target = Vec3(
            x=home.x,
            y=home.y,
            z=home.z - abs(zone.search_altitude_m),
        )
        validate_position(self._to_home(target, home), zone, plan.safety)
        await self._event(
            "flight.phase",
            {
                "phase": "initial_climb",
                "target_altitude_m": zone.search_altitude_m,
                "message": f"原地爬升到 {zone.search_altitude_m:.1f} m 搜索高度",
            },
            run.id,
        )
        await adapter.request(
            "move_to",
            vehicle_name=profile.vehicle_name,
            x=target.x,
            y=target.y,
            z=target.z,
            speed=min(1.5, plan.safety.max_speed_mps),
            timeout=30,
        )
        deadline = asyncio.get_running_loop().time() + 30
        while asyncio.get_running_loop().time() < deadline:
            await self._handle_control(run, control)
            telemetry = await self._telemetry(run)
            relative = self._to_home(telemetry.position, home)
            altitude = -relative.z
            if altitude > plan.safety.max_altitude_m + 0.5:
                raise SafetyViolation(
                    f"altitude {altitude:.2f} m exceeded the takeoff corridor ceiling"
                )
            if not point_in_polygon(relative, zone.polygon):
                raise SafetyViolation("vehicle left the search geofence during initial climb")
            if distance(telemetry.position, target) <= 0.75:
                await self._event(
                    "flight.phase",
                    {
                        "phase": "initial_climb_complete",
                        "altitude_m": altitude,
                        "message": f"已到达 {altitude:.1f} m，开始覆盖搜索",
                    },
                    run.id,
                )
                return telemetry
            await asyncio.sleep(0.2)
        raise MissionError("vehicle did not reach the search altitude before timeout")

    async def _wait_position(
        self,
        run: RunRecord,
        target: Vec3,
        control: RunControl,
        timeout: float,
        tolerance: float = 0.75,
    ) -> Telemetry:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            await self._handle_control(run, control)
            telemetry = await self._telemetry(run)
            if distance(telemetry.position, target) <= tolerance:
                return telemetry
            await asyncio.sleep(0.2)
        raise MissionError("vehicle did not reach the requested position before timeout")

    async def _wait_for_altitude(
        self,
        run: RunRecord,
        home_z: float,
        altitude_m: float,
        control: RunControl,
        timeout: float,
    ) -> Telemetry:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            await self._handle_control(run, control)
            telemetry = await self._telemetry(run)
            if not telemetry.landed and home_z - telemetry.position.z >= min(altitude_m, 0.5):
                return telemetry
            await asyncio.sleep(0.2)
        raise MissionError("takeoff did not complete before timeout")

    @staticmethod
    def _from_home(position: Vec3, home: Vec3) -> Vec3:
        return Vec3(x=home.x + position.x, y=home.y + position.y, z=home.z + position.z)

    @staticmethod
    def _to_home(position: Vec3, home: Vec3) -> Vec3:
        return Vec3(x=position.x - home.x, y=position.y - home.y, z=position.z - home.z)

    async def _wait_landed(
        self,
        run: RunRecord,
        control: RunControl,
        timeout: float,
        ignore_controls: bool = False,
        ground_z: float | None = None,
    ) -> Telemetry:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        if not profile or not adapter:
            raise MissionError("simulator disconnected")
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if not ignore_controls:
                await self._handle_control(run, control)
            telemetry = await adapter.telemetry(profile.vehicle_name)
            values = (
                telemetry.position.x,
                telemetry.position.y,
                telemetry.position.z,
                telemetry.velocity.x,
                telemetry.velocity.y,
                telemetry.velocity.z,
            )
            if not all(math.isfinite(value) for value in values):
                raise SafetyViolation("telemetry contains a non-finite value")
            age = (utc_now() - telemetry.timestamp).total_seconds()
            if age > profile.safety.telemetry_stale_seconds:
                raise SafetyViolation(f"telemetry is stale by {age:.2f} seconds")
            self.store.append_jsonl(
                Path(run.artifact_dir) / "telemetry.jsonl", telemetry.model_dump(mode="json")
            )
            await self._event("telemetry", telemetry.model_dump(mode="json"), run.id)
            speed = math.sqrt(
                telemetry.velocity.x**2
                + telemetry.velocity.y**2
                + telemetry.velocity.z**2
            )
            near_ground = (
                ground_z is not None
                and abs(telemetry.position.z - ground_z) <= 0.3
                and speed <= 0.5
            )
            if telemetry.collision and not near_ground:
                raise SafetyViolation("AirSim reported a collision")
            if telemetry.landed or near_ground:
                return telemetry
            await asyncio.sleep(0.2)
        raise MissionError("landing did not complete before timeout")

    async def _telemetry(self, run: RunRecord) -> Telemetry:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        if not profile or not adapter:
            raise MissionError("simulator disconnected")
        telemetry = await adapter.telemetry(profile.vehicle_name)
        validate_telemetry(telemetry)
        age = (utc_now() - telemetry.timestamp).total_seconds()
        if age > profile.safety.telemetry_stale_seconds:
            raise SafetyViolation(f"telemetry is stale by {age:.2f} seconds")
        self.store.append_jsonl(
            Path(run.artifact_dir) / "telemetry.jsonl", telemetry.model_dump(mode="json")
        )
        await self._event("telemetry", telemetry.model_dump(mode="json"), run.id)
        return telemetry

    async def _handle_control(self, run: RunRecord, control: RunControl) -> str | None:
        if (
            control.deadline_monotonic is not None
            and asyncio.get_running_loop().time() > control.deadline_monotonic
        ):
            raise MissionError("mission exceeded the configured time limit")
        if control.abort.is_set():
            raise MissionAborted("operator aborted the mission")
        if control.land.is_set():
            raise LandRequested("operator requested immediate landing")
        while control.paused.is_set():
            if control.abort.is_set():
                raise MissionAborted("operator aborted the paused mission")
            if control.land.is_set():
                raise LandRequested("operator requested landing")
            await asyncio.sleep(0.2)
        if control.return_home.is_set():
            return "return_home"
        return None

    def _save_frame(self, run: RunRecord, frame: CameraFrame, evidence: bool = False) -> None:
        root = Path(run.artifact_dir) / ("evidence" if evidence else "frames")
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{frame.frame_id}.png").write_bytes(base64.b64decode(frame.scene_png_b64))
        if frame.depth_f32_zlib_b64:
            (root / f"{frame.frame_id}.depth.zlib").write_bytes(
                base64.b64decode(frame.depth_f32_zlib_b64)
            )
        (root / f"{frame.frame_id}.json").write_text(
            frame.model_dump_json(indent=2, exclude={"scene_png_b64", "depth_f32_zlib_b64"}),
            encoding="utf-8",
        )

    async def control(self, run_id: str, action: str) -> RunRecord:
        run = self.store.get_run(run_id)
        control = self._controls.get(run_id)
        if not run or not control:
            raise MissionError("active run does not exist")
        if run.state in TERMINAL_STATES:
            raise MissionError("run has already finished")
        adapter = self.simulator.adapter
        profile = self.simulator.active_profile
        if action == "pause":
            control.paused.set()
            if adapter and profile:
                await adapter.request("hover", timeout=3, vehicle_name=profile.vehicle_name)
            run = await self._set_state(run, RunState.PAUSED)
        elif action == "resume":
            control.paused.clear()
            run = await self._set_state(run, RunState.SEARCHING)
        elif action == "return-home":
            control.return_home.set()
        elif action == "land":
            control.land.set()
            if adapter and profile:
                await adapter.request("cancel", timeout=3, vehicle_name=profile.vehicle_name)
                await adapter.request("land", timeout=3, vehicle_name=profile.vehicle_name)
        elif action == "abort":
            control.abort.set()
            if adapter and profile:
                await adapter.request("hover", timeout=3, vehicle_name=profile.vehicle_name)
        elif action == "hard-stop":
            control.abort.set()
            await self.simulator.stop(hard=True)
            run = await self._set_state(run, RunState.FAILED, error="simulator hard-stopped", ended=True)
        else:
            raise MissionError(f"unknown control action: {action}")
        await self._event("run.control", {"action": action}, run_id)
        return run

    async def candidate_decision(self, run_id: str, decision: str) -> None:
        control = self._controls.get(run_id)
        run = self.store.get_run(run_id)
        if not control or not run or run.state != RunState.EVIDENCE:
            raise MissionError("run is not awaiting a candidate decision")
        if decision not in {"accept", "continue"}:
            raise MissionError("invalid candidate decision")
        if control.candidate_decision.full():
            control.candidate_decision.get_nowait()
        control.candidate_decision.put_nowait(decision)
        await self._event("candidate.decision", {"decision": decision}, run_id)

    async def close(self) -> None:
        for control in self._controls.values():
            control.abort.set()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        await self.provider.close()
