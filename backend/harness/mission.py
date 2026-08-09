from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from .avoidance import assess_corridor, choose_local_detour, decode_point_cloud
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
from .planner import build_plan, resolve_search_zone
from .safety import (
    SafetyViolation,
    approach_point,
    point_in_polygon,
    validate_position,
    validate_telemetry,
)
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


@dataclass
class VlmGuidanceState:
    target: Vec3
    observation_index: int
    consecutive_failures: int = 0
    update_count: int = 0
    fatal_error: str | None = None


@dataclass
class LocalPlannerState:
    preferred_side: int | None = None
    last_heading_rad: float | None = None


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
        return resolve_search_zone(profile, plan.request)

    async def _execute(self, run: RunRecord, plan: MissionPlan) -> None:
        adapter = self.simulator.adapter
        profile = self.simulator.active_profile
        if not adapter or not profile:
            return
        control = self._controls[run.id]
        zone = self._zone(plan)
        found = False
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
                observation_index,
            )
            if confirmed:
                accepted, observation_index = await self._approach_and_review(
                    run, plan, zone, confirmed, home, control, observation_index
                )
                if accepted:
                    found = True
                elif not control.return_home.is_set():
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
                            run, plan, zone, home, observation_index
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
                        run, plan, zone, frame, telemetry, assessment, home
                    )
                    if confirmed:
                        accepted, observation_index = await self._approach_and_review(
                            run,
                            plan,
                            zone,
                            confirmed,
                            home,
                            control,
                            observation_index,
                        )
                        if accepted:
                            found = True
                            break
                        if not control.return_home.is_set():
                            run = self.store.get_run(run.id) or run
                            run = await self._set_state(run, RunState.SEARCHING)
                if found or control.return_home.is_set() or control.land.is_set():
                    break

            run = self.store.get_run(run.id) or run
            if control.land.is_set():
                raise LandRequested("operator requested immediate landing")
            control.return_home.clear()
            run = await self._return_and_land(
                run, plan, zone, profile.vehicle_name, home, control
            )
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
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        observation_index: int,
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
        telemetry = await self._telemetry(run, plan=plan, zone=zone, home=home)
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
                    run, plan, zone, home, observation_index
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
            locked = await self._evaluate_candidate(
                run, plan, zone, frame, telemetry, assessment, home
            )
            if locked is None:
                continue
            await self._event(
                "vision.locked",
                {
                    "target": locked.model_dump(),
                    "frame_id": frame.frame_id,
                    "bbox_norm": assessment.bbox_norm.model_dump()
                    if assessment.bbox_norm
                    else None,
                    "confidence": assessment.confidence,
                    "mode": "single_frame_depth",
                    "message": "VLM 单帧目标框已锁定，立即进入安全接近",
                },
                run.id,
            )
            return locked, observation_index
        return None, observation_index

    async def _evaluate_candidate(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        frame: CameraFrame,
        telemetry: Telemetry,
        assessment: DetectionAssessment,
        home: Vec3,
    ) -> Vec3 | None:
        if not assessment.is_match or assessment.confidence < 0.75 or not assessment.bbox_norm:
            return None
        try:
            target = localize_bbox(frame, assessment.bbox_norm)
        except DepthLocalizationError as error:
            await self._event("vision.rejected", {"reason": str(error)}, run.id)
            return None
        relative_target = self._to_home(target, home)
        if not point_in_polygon(relative_target, zone.polygon):
            await self._event(
                "vision.rejected",
                {
                    "reason": "target is outside the search zone",
                    "target_ned": relative_target.model_dump(),
                    "message": (
                        "候选目标位于安全范围外："
                        f"NED ({relative_target.x:.1f}, {relative_target.y:.1f}, "
                        f"{relative_target.z:.1f}) m"
                    ),
                },
                run.id,
            )
            return None
        await self._event(
            "vision.confirmed",
            {
                "target": target.model_dump(),
                "mode": "single_frame_depth",
                "confidence": assessment.confidence,
                "message": "单帧 VLM 匹配和初始深度定位有效，跳过居中与二次深度复核",
            },
            run.id,
        )
        return target

    async def _approach_and_review(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        target: Vec3,
        home: Vec3,
        control: RunControl,
        observation_index: int,
    ) -> tuple[bool, int]:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        run = await self._set_state(run, RunState.APPROACHING, target_position=target)
        guidance = VlmGuidanceState(target=target, observation_index=observation_index)
        stop_guidance = asyncio.Event()
        guidance_task = asyncio.create_task(
            self._stream_vlm_guidance(
                run, plan, zone, home, guidance, stop_guidance
            ),
            name=f"vlm-guidance-{run.id}",
        )
        planner_state = LocalPlannerState()
        reached_standoff = False
        max_action_chunks = 24
        try:
            # Let capture/model inference begin before the first motion chunk.
            await asyncio.sleep(0)
            for chunk_index in range(1, max_action_chunks + 1):
                directive = await self._handle_control(run, control)
                if directive == "return_home":
                    return False, guidance.observation_index
                if guidance.fatal_error:
                    raise SafetyViolation(guidance.fatal_error)
                telemetry = await self._telemetry(
                    run, plan=plan, zone=zone, home=home
                )
                # Snapshot the current ensemble. A committed action chunk is
                # never redirected mid-flight by a late/noisy VLM response.
                chunk_target = guidance.target
                approach = approach_point(
                    telemetry.position,
                    chunk_target,
                    zone.search_altitude_m,
                    plan.safety.min_standoff_m,
                ).model_copy(update={"z": home.z - abs(zone.search_altitude_m)})
                validate_position(self._to_home(approach, home), zone, plan.safety)
                remaining = distance(telemetry.position, approach)
                if remaining <= 1.5:
                    reached_standoff = True
                    break
                chunk_m = min(3.0, plan.safety.avoidance_segment_m)
                ratio = min(1.0, chunk_m / remaining)
                waypoint = Vec3(
                    x=telemetry.position.x + (approach.x - telemetry.position.x) * ratio,
                    y=telemetry.position.y + (approach.y - telemetry.position.y) * ratio,
                    z=telemetry.position.z + (approach.z - telemetry.position.z) * ratio,
                )
                await self._event(
                    "action_chunk.committed",
                    {
                        "chunk": chunk_index,
                        "target_generation": guidance.update_count,
                        "target": chunk_target.model_dump(),
                        "waypoint": waypoint.model_dump(),
                        "length_limit_m": chunk_m,
                        "remaining_to_standoff_m": remaining,
                        "message": "提交短时动作块；飞行中 VLM 与 LiDAR 继续更新下一段",
                    },
                    run.id,
                )
                await self._move_segmented(
                    run,
                    plan,
                    zone,
                    waypoint,
                    home,
                    control,
                    speed=plan.safety.approach_speed_mps,
                    look_at=chunk_target,
                    planner_state=planner_state,
                )
                await self._event(
                    "action_chunk.completed",
                    {
                        "chunk": chunk_index,
                        "latest_target_generation": guidance.update_count,
                        "message": "动作块完成，立即用最新目标和局部障碍重规划",
                    },
                    run.id,
                )
                # Fair scheduling for the mock provider; real VLM latency also
                # naturally yields while the vehicle keeps moving.
                await asyncio.sleep(0)
            if not reached_standoff:
                raise SafetyViolation(
                    "continuous VLM action chunks did not converge to the standoff point"
                )
        finally:
            stop_guidance.set()
            guidance_task.cancel()
            with suppress(asyncio.CancelledError):
                await guidance_task
        if guidance.fatal_error:
            raise SafetyViolation(guidance.fatal_error)
        observation_index = guidance.observation_index
        current_target = guidance.target
        # Hover is reserved for final evidence (and explicit safety/manual
        # controls), never inserted between guidance chunks.
        await adapter.request("hover", timeout=4, vehicle_name=profile.vehicle_name)
        evidence = await adapter.capture(profile.vehicle_name)
        self._save_frame(run, evidence, evidence=True)
        run = await self._set_state(
            run, RunState.EVIDENCE, target_position=current_target
        )
        if plan.request.end_policy == "auto_rth":
            return True, observation_index
        await self._event(
            "candidate.review",
            {"timeout_seconds": 30, "default": "accept", "frame_id": evidence.frame_id},
            run.id,
        )
        try:
            decision = await asyncio.wait_for(control.candidate_decision.get(), timeout=30)
        except asyncio.TimeoutError:
            decision = "accept"
        return decision == "accept", observation_index

    async def _stream_vlm_guidance(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        state: VlmGuidanceState,
        stop_event: asyncio.Event,
    ) -> None:
        profile = self.simulator.active_profile
        assert profile
        await self._event(
            "guidance.stream_started",
            {
                "mode": "continuous_in_flight",
                "message": "VLM 后台连续引导已启动，不为识别插入悬停",
            },
            run.id,
        )
        try:
            while not stop_event.is_set():
                state.observation_index += 1
                try:
                    frame, telemetry, assessment = await self._capture_assessment(
                        run, plan, zone, home, state.observation_index
                    )
                    observed_target = await self._evaluate_candidate(
                        run, plan, zone, frame, telemetry, assessment, home
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    observed_target = None
                    await self._event(
                        "guidance.model_error",
                        {
                            "observation_index": state.observation_index,
                            "error": f"{type(error).__name__}: {error}",
                        },
                        run.id,
                    )
                if observed_target is None:
                    state.consecutive_failures += 1
                    await self._event(
                        "guidance.target_lost",
                        {
                            "observation_index": state.observation_index,
                            "consecutive_misses": state.consecutive_failures,
                            "message": "当前帧未得到可靠目标；保持当前动作块并后台复查",
                        },
                        run.id,
                    )
                else:
                    update_distance = distance(state.target, observed_target)
                    distance_to_target = distance(telemetry.position, state.target)
                    allowed_update = max(
                        3.0, min(10.0, distance_to_target * 0.25)
                    )
                    if update_distance > allowed_update:
                        state.consecutive_failures += 1
                        await self._event(
                            "guidance.update_rejected",
                            {
                                "observation_index": state.observation_index,
                                "previous_target": state.target.model_dump(),
                                "observed_target": observed_target.model_dump(),
                                "update_distance_m": update_distance,
                                "allowed_update_m": allowed_update,
                                "consecutive_misses": state.consecutive_failures,
                                "message": "VLM 坐标跳变过大；不打断当前动作块",
                            },
                            run.id,
                        )
                    else:
                        # Temporal ensemble: keep most of the prior estimate so
                        # frame-to-frame box/depth noise cannot jerk the heading.
                        alpha = 0.35
                        state.target = Vec3(
                            x=state.target.x * (1 - alpha) + observed_target.x * alpha,
                            y=state.target.y * (1 - alpha) + observed_target.y * alpha,
                            z=state.target.z * (1 - alpha) + observed_target.z * alpha,
                        )
                        state.consecutive_failures = 0
                        state.update_count += 1
                        await self._event(
                            "guidance.target_updated",
                            {
                                "observation_index": state.observation_index,
                                "target_generation": state.update_count,
                                "target": state.target.model_dump(),
                                "raw_update_distance_m": update_distance,
                                "smoothing_alpha": alpha,
                                "message": "最新 VLM 结果已平滑写入下一动作块",
                            },
                            run.id,
                        )
                if state.consecutive_failures >= 3:
                    state.fatal_error = (
                        "VLM guidance lost the target or produced unsafe "
                        "coordinate jumps three times"
                    )
                    await self._event(
                        "guidance.stream_failed",
                        {"reason": state.fatal_error},
                        run.id,
                    )
                    return
                await asyncio.sleep(0)
        finally:
            await self._event(
                "guidance.stream_stopped",
                {"updates": state.update_count},
                run.id,
            )

    async def _return_and_land(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        vehicle_name: str,
        home: Vec3,
        control: RunControl,
    ) -> RunRecord:
        adapter = self.simulator.adapter
        assert adapter
        telemetry = await self._telemetry(run, plan=plan, zone=zone, home=home)
        run = await self._set_state(run, RunState.RTH)
        await self._move_segmented(
            run,
            plan,
            zone,
            Vec3(x=home.x, y=home.y, z=telemetry.position.z),
            home,
            control,
            speed=min(2.0, plan.safety.max_speed_mps),
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
        look_at: Vec3 | None = None,
        planner_state: LocalPlannerState | None = None,
    ) -> None:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        telemetry = await self._telemetry(run, plan=plan, zone=zone, home=home)
        total = distance(telemetry.position, target)
        segment_m = plan.safety.avoidance_segment_m
        max_commands = (
            max(1, math.ceil(total / segment_m))
            + plan.safety.avoidance_max_replans
            + 4
        )
        replans = 0
        planner_state = planner_state or LocalPlannerState()
        for _ in range(max_commands):
            directive = await self._handle_control(run, control)
            if directive == "return_home":
                return
            current = telemetry.position
            remaining = distance(current, target)
            if remaining <= 1.5:
                return
            ratio = min(1.0, segment_m / remaining)
            direct = Vec3(
                x=current.x + (target.x - current.x) * ratio,
                y=current.y + (target.y - current.y) * ratio,
                z=current.z + (target.z - current.z) * ratio,
            )
            direct_allowed = self._segment_is_allowed(
                current, direct, home, zone, plan
            )
            points = []
            assessment = None
            if plan.safety.obstacle_avoidance_enabled:
                points = await self._lidar_points(run, profile.vehicle_name)
                assessment = assess_corridor(
                    current,
                    direct,
                    points,
                    plan.safety.min_clearance_m,
                )
            blocked = not direct_allowed or bool(assessment and assessment.blocked)
            await self._event(
                "avoidance.scan",
                {
                    "from": current.model_dump(),
                    "to": direct.model_dump(),
                    "point_count": len(points),
                    "minimum_clearance_m": (
                        assessment.minimum_clearance_m if assessment else None
                    ),
                    "required_clearance_m": plan.safety.min_clearance_m,
                    "blocked": blocked,
                    "geofence_clear": direct_allowed,
                },
                run.id,
            )
            point = direct
            if blocked:
                replans += 1
                if replans > plan.safety.avoidance_max_replans:
                    raise SafetyViolation(
                        "local obstacle avoidance exceeded its replan limit"
                    )
                detour = choose_local_detour(
                    current,
                    target,
                    points,
                    plan.safety.min_clearance_m,
                    segment_m,
                    lambda start, end: self._segment_is_allowed(
                        start, end, home, zone, plan
                    ),
                    planner_state.preferred_side,
                    planner_state.last_heading_rad,
                )
                if detour is None:
                    raise SafetyViolation(
                        "LiDAR found an occupied flight corridor and no safe local detour"
                    )
                point = detour.waypoint
                planner_state.preferred_side = detour.side
                await self._event(
                    "avoidance.detour",
                    {
                        "blocked_target": direct.model_dump(),
                        "waypoint": point.model_dump(),
                        "side": "right" if detour.side > 0 else "left",
                        "angle_degrees": detour.angle_degrees,
                        "observed_clearance_m": detour.minimum_clearance_m,
                        "replan": replans,
                        "planner": "rolling_local_ego_inspired",
                        "previous_heading_degrees": (
                            math.degrees(planner_state.last_heading_rad)
                            if planner_state.last_heading_rad is not None
                            else None
                        ),
                        "message": "滚动局部规划发现阻塞；保持方向连续性绕行并在下一段重扫",
                    },
                    run.id,
                )
            validate_position(self._to_home(point, home), zone, plan.safety)
            arguments: dict[str, float | str] = {
                "vehicle_name": profile.vehicle_name,
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "speed": min(speed or plan.safety.max_speed_mps, plan.safety.max_speed_mps),
            }
            heading_target = look_at or point
            dx = heading_target.x - current.x
            dy = heading_target.y - current.y
            if math.hypot(dx, dy) > 1e-6:
                arguments["yaw_degrees"] = math.degrees(math.atan2(dy, dx))
            motion_dx = point.x - current.x
            motion_dy = point.y - current.y
            if math.hypot(motion_dx, motion_dy) > 1e-6:
                planner_state.last_heading_rad = math.atan2(motion_dy, motion_dx)
            await adapter.request(
                "move_to",
                timeout=20,
                **arguments,
            )
            # SimpleFlight can resolve a short moveToPosition future about
            # 1.2 m from its requested point at cruise speed. Intermediate
            # segments use a matching acceptance radius; every telemetry
            # sample is still checked against the full safety envelope.
            telemetry = await self._wait_position(
                run,
                point,
                control,
                timeout=20,
                tolerance=1.5,
                plan=plan,
                zone=zone,
                home=home,
            )
        raise SafetyViolation("local obstacle avoidance did not converge on the destination")

    async def _lidar_points(self, run: RunRecord, vehicle_name: str) -> list[Vec3]:
        adapter = self.simulator.adapter
        assert adapter
        for attempt in range(1, 4):
            scan = await adapter.request(
                "lidar_scan",
                timeout=4,
                vehicle_name=vehicle_name,
                sensor_name="LidarSensor1",
                max_points=6000,
            )
            if scan.get("data_frame") != "VehicleInertialFrame":
                raise SafetyViolation("LiDAR scan is not in the required world NED frame")
            points = decode_point_cloud(scan.get("point_cloud", []))
            if points:
                return points
            await self._event(
                "avoidance.sensor_wait",
                {"sensor": "LidarSensor1", "attempt": attempt},
                run.id,
            )
            await asyncio.sleep(0.2)
        raise SafetyViolation("LiDAR returned no usable obstacle points")

    def _segment_is_allowed(
        self,
        start: Vec3,
        end: Vec3,
        home: Vec3,
        zone: SearchZone,
        plan: MissionPlan,
    ) -> bool:
        segment_length = distance(start, end)
        samples = max(1, math.ceil(segment_length / 0.5))
        try:
            for index in range(samples + 1):
                ratio = index / samples
                point = Vec3(
                    x=start.x + (end.x - start.x) * ratio,
                    y=start.y + (end.y - start.y) * ratio,
                    z=start.z + (end.z - start.z) * ratio,
                )
                validate_position(self._to_home(point, home), zone, plan.safety)
        except SafetyViolation:
            return False
        return True

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
        plan: MissionPlan | None = None,
        zone: SearchZone | None = None,
        home: Vec3 | None = None,
    ) -> Telemetry:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            await self._handle_control(run, control)
            telemetry = await self._telemetry(run, plan=plan, zone=zone, home=home)
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

    async def _telemetry(
        self,
        run: RunRecord,
        *,
        plan: MissionPlan | None = None,
        zone: SearchZone | None = None,
        home: Vec3 | None = None,
    ) -> Telemetry:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        if not profile or not adapter:
            raise MissionError("simulator disconnected")
        telemetry = await adapter.telemetry(profile.vehicle_name)
        validate_telemetry(telemetry)
        age = (utc_now() - telemetry.timestamp).total_seconds()
        if age > profile.safety.telemetry_stale_seconds:
            raise SafetyViolation(f"telemetry is stale by {age:.2f} seconds")
        if plan is not None and zone is not None and home is not None:
            validate_position(self._to_home(telemetry.position, home), zone, plan.safety)
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
