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

from .avoidance import (
    assess_corridor,
    choose_local_detour,
    decode_point_cloud,
    point_cloud_preview_payload,
)
from .config import Settings
from .events import EventBus
from .geometry import (
    DepthLocalizationError,
    distance,
    frame_preview_payload,
    localize_bbox,
)
from .llm import ModelProvider
from .models import (
    CameraFrame,
    DetectionAssessment,
    MissionPlan,
    MissionPlanParameters,
    MissionTask,
    MissionTaskProgress,
    RoutePoint,
    RunRecord,
    RunState,
    SearchMissionRequest,
    SearchZone,
    Telemetry,
    TERMINAL_STATES,
    Vec3,
    utc_now,
)
from .planner import (
    build_plan,
    default_plan_parameters,
    resolve_plan_zone,
)
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


class SearchTargetAcquired(MissionError):
    """Interrupt a committed search chunk as soon as perception locks a target."""


class SearchPathBlocked(MissionError):
    """The current coverage waypoint is unreachable without violating safety."""


class AltitudeChangeRequested(MissionError):
    """Interrupt the current mapping segment to execute a queued altitude change."""


@dataclass(frozen=True)
class ExplorationDirective:
    heading_degrees: float
    distance_m: float


@dataclass(frozen=True)
class AltitudeDirective:
    target_altitude_m: float
    requested_delta_m: float | None = None


@dataclass
class RunControl:
    paused: asyncio.Event = field(default_factory=asyncio.Event)
    abort: asyncio.Event = field(default_factory=asyncio.Event)
    return_home: asyncio.Event = field(default_factory=asyncio.Event)
    land: asyncio.Event = field(default_factory=asyncio.Event)
    candidate_decision: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    exploration_directive: asyncio.Queue[ExplorationDirective] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1)
    )
    altitude_directive: asyncio.Queue[AltitudeDirective] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1)
    )
    altitude_override_m: float | None = None
    home_position: Vec3 | None = None
    deadline_monotonic: float | None = None


@dataclass
class VlmGuidanceState:
    target: Vec3
    observation_index: int
    consecutive_failures: int = 0
    consecutive_misses: int = 0
    consecutive_update_rejections: int = 0
    update_count: int = 0
    fatal_error: str | None = None
    occlusion_expected_until: float = 0.0
    relocalization_candidates: list[Vec3] = field(default_factory=list)


@dataclass
class SearchGuidanceState:
    observation_index: int
    target: Vec3 | None = None
    consecutive_model_errors: int = 0
    fatal_error: str | None = None


@dataclass
class LocalPlannerState:
    preferred_side: int | None = None
    last_heading_rad: float | None = None
    consecutive_blocked_replans: int = 0
    recovery_replans: int = 0
    recent_waypoints: list[Vec3] = field(default_factory=list)


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

    def revise_plan(
        self,
        plan_id: str,
        base_version: int,
        parameters: MissionPlanParameters,
    ) -> MissionPlan:
        plan = self.store.get_plan(plan_id)
        if not plan:
            raise MissionError("mission plan does not exist")
        if plan.approved_at:
            raise MissionError("an approved mission plan cannot be revised")
        if plan.version != base_version:
            raise MissionError(
                f"stale plan version: expected v{plan.version}, received v{base_version}"
            )
        profile = self.simulator.profiles.get(plan.request.scene_id)
        if not profile:
            raise MissionError(f"unknown scene: {plan.request.scene_id}")
        revised = build_plan(
            profile,
            plan.request,
            parameters=parameters,
            version=plan.version + 1,
        )
        self.store.save_plan(revised)
        return revised

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
        tasks = plan.tasks or [
            MissionTask(
                id="target-1",
                kind="target_search",
                label=f"搜索并取证：{plan.request.target_text}",
                target_text=plan.request.target_text,
            )
        ]
        run = run.model_copy(
            update={
                "task_progress": [
                    MissionTaskProgress(
                        task_id=task.id,
                        kind=task.kind,
                        label=task.label,
                    )
                    for task in tasks
                ],
                "mapping_coverage_ratio": (
                    0.0 if any(task.kind == "semantic_mapping" for task in tasks) else None
                ),
            }
        )
        self.store.save_run(run)
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

    async def _update_task_progress(
        self,
        run: RunRecord,
        task_index: int,
        state: str,
        *,
        message: str,
        target_position: Vec3 | None = None,
        coverage_ratio: float | None = None,
    ) -> RunRecord:
        current = self.store.get_run(run.id) or run
        progress = list(current.task_progress)
        if not (0 <= task_index < len(progress)):
            return current
        progress[task_index] = progress[task_index].model_copy(
            update={
                "state": state,
                "message": message,
                "target_position": target_position,
                "coverage_ratio": coverage_ratio,
            }
        )
        updates: dict = {
            "current_task_index": task_index,
            "task_progress": progress,
        }
        if coverage_ratio is not None:
            updates["mapping_coverage_ratio"] = coverage_ratio
        current = current.model_copy(update=updates)
        self.store.save_run(current)
        await self._event(
            "mission.progress",
            {
                "task_index": task_index,
                "task": progress[task_index].model_dump(mode="json"),
                "run": current.model_dump(mode="json"),
                "message": message,
            },
            current.id,
        )
        return current

    def _zone(self, plan: MissionPlan) -> SearchZone:
        profile = self.simulator.profiles[plan.request.scene_id]
        parameters = plan.parameters or default_plan_parameters(profile, plan.request)
        return resolve_plan_zone(profile, plan.request, parameters)

    async def _execute(self, run: RunRecord, plan: MissionPlan) -> None:
        adapter = self.simulator.adapter
        profile = self.simulator.active_profile
        if not adapter or not profile:
            return
        control = self._controls[run.id]
        zone = self._zone(plan)
        found = False
        observation_index = 0
        self.simulator.mission_active = True
        try:
            home_telemetry = await self._telemetry(run)
            home = home_telemetry.position
            control.home_position = home
            run = run.model_copy(update={"home_position": home})
            self.store.save_run(run)
            await self._event(
                "run.home",
                {
                    "home_position": home.model_dump(),
                    "coordinate_frame": "run-home-relative-ned",
                    "message": "任务安全范围已锚定到本次起飞点",
                },
                run.id,
            )
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
            tasks = plan.tasks or [
                MissionTask(
                    id="target-1",
                    kind="target_search",
                    label=f"搜索并取证：{plan.request.target_text}",
                    target_text=plan.request.target_text,
                )
            ]
            if tasks[0].kind == "semantic_mapping":
                run = await self._set_state(run, RunState.SEARCHING)
                run = await self._update_task_progress(
                    run, 0, "running", message="开始执行区域覆盖建图航线", coverage_ratio=0.0
                )
                found, observation_index, coverage_ratio = await self._execute_semantic_mapping(
                    run, plan, zone, home, control, observation_index
                )
                run = self.store.get_run(run.id) or run
                run = await self._update_task_progress(
                    run,
                    0,
                    "succeeded" if found else "not_found",
                    message=(
                        "给定区域占据与语义拓扑图已建立"
                        if found
                        else "区域建图航线未完整执行"
                    ),
                    coverage_ratio=coverage_ratio,
                )
            else:
                completed_targets = 0
                for task_index, task in enumerate(tasks):
                    if control.return_home.is_set() or control.land.is_set():
                        break
                    assert task.target_text is not None
                    # Multi-target plans use one approval for the whole sequence;
                    # intermediate evidence is accepted automatically and saved.
                    task_request = plan.request.model_copy(
                        update={
                            "target_text": task.target_text,
                            "targets": [task.target_text],
                            "end_policy": (
                                "auto_rth" if len(tasks) > 1 else plan.request.end_policy
                            ),
                        }
                    )
                    task_plan = plan.model_copy(update={"request": task_request})
                    run = self.store.get_run(run.id) or run
                    run = await self._set_state(run, RunState.SEARCHING)
                    run = await self._update_task_progress(
                        run,
                        task_index,
                        "running",
                        message=f"正在搜索目标 {task_index + 1}/{len(tasks)}：{task.target_text}",
                    )
                    await self._event(
                        "mission.task_started",
                        {
                            "task_index": task_index,
                            "task_count": len(tasks),
                            "target_text": task.target_text,
                            "message": f"开始独立视觉查询：{task.target_text}",
                        },
                        run.id,
                    )
                    task_found, observation_index = await self._execute_target_search(
                        run,
                        task_plan,
                        zone,
                        home,
                        control,
                        observation_index,
                    )
                    run = self.store.get_run(run.id) or run
                    if task_found:
                        completed_targets += 1
                        run = await self._update_task_progress(
                            run,
                            task_index,
                            "succeeded",
                            message=f"已找到并保存目标证据：{task.target_text}",
                            target_position=run.target_position,
                        )
                    else:
                        run = await self._update_task_progress(
                            run,
                            task_index,
                            "not_found",
                            message=f"覆盖搜索未发现：{task.target_text}",
                        )
                found = completed_targets == len(tasks)

            run = self.store.get_run(run.id) or run
            if control.land.is_set():
                raise LandRequested("operator requested immediate landing")
            if found and plan.request.end_policy == "land_at_target":
                run = await self._land_in_place(
                    run, profile.vehicle_name, ground_z=home.z
                )
            else:
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
            await self._emergency_land_and_release(
                run,
                control,
                profile.vehicle_name,
                ground_z=control.home_position.z if control.home_position else None,
            )
            run = await self._set_state(run, RunState.FAILED, error=str(error), ended=True)
        finally:
            self.simulator.mission_active = False
            current = self.store.get_run(run.id) or run
            self.store.write_topology_map(current, self.simulator.mapper.snapshot())
            self.store.write_report(current, plan)
            self._tasks.pop(run.id, None)

    async def _emergency_land_and_release(
        self,
        run: RunRecord,
        control: RunControl,
        vehicle_name: str,
        *,
        ground_z: float | None,
    ) -> None:
        """Land after a fault and release SimpleFlight control when grounded."""
        adapter = self.simulator.adapter
        if not adapter:
            return
        landed = False
        cleanup_error: str | None = None
        try:
            await adapter.request("cancel", timeout=3, vehicle_name=vehicle_name)
            await adapter.request("land", timeout=5, vehicle_name=vehicle_name)
            await self._wait_landed(
                run,
                control,
                timeout=45,
                ignore_controls=True,
                ignore_collision=True,
                ground_z=ground_z,
            )
            landed = True
        except Exception as error:
            cleanup_error = f"{type(error).__name__}: {error}"
        if landed:
            try:
                await adapter.request(
                    "arm", timeout=5, vehicle_name=vehicle_name, armed=False
                )
                await adapter.request(
                    "api_control",
                    timeout=5,
                    vehicle_name=vehicle_name,
                    enabled=False,
                )
            except Exception as error:
                cleanup_error = cleanup_error or f"{type(error).__name__}: {error}"
        await self._event(
            "flight.emergency_cleanup",
            {
                "landed": landed,
                "disarmed": landed and cleanup_error is None,
                "error": cleanup_error,
                "message": (
                    "Fault cleanup landed, disarmed and released API control"
                    if landed and cleanup_error is None
                    else "Fault cleanup could not fully confirm landing and disarm"
                ),
            },
            run.id,
        )

    async def _execute_target_search(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
        observation_index: int,
    ) -> tuple[bool, int]:
        found = False
        confirmed, observation_index = await self._initial_panorama(
            run, plan, zone, home, control, observation_index
        )
        if confirmed:
            accepted, observation_index = await self._approach_and_review(
                run, plan, zone, confirmed, home, control, observation_index
            )
            if accepted:
                found = True
            elif not control.return_home.is_set():
                run = self.store.get_run(run.id) or run
                await self._set_state(run, RunState.SEARCHING)

        while (
            not found
            and not control.return_home.is_set()
            and not control.land.is_set()
        ):
            confirmed, observation_index = await self._search_with_live_vlm(
                run, plan, zone, home, control, observation_index
            )
            if confirmed is None:
                break
            accepted, observation_index = await self._approach_and_review(
                run, plan, zone, confirmed, home, control, observation_index
            )
            if accepted:
                found = True
                break
            if not control.return_home.is_set():
                run = self.store.get_run(run.id) or run
                await self._set_state(run, RunState.SEARCHING)
        return found, observation_index

    async def _execute_semantic_mapping(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
        observation_index: int,
    ) -> tuple[bool, int, float]:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        planner_state = LocalPlannerState()
        total = max(1, len(plan.route))
        model_successes = 0
        coverage_ratio = 0.0
        world_polygon = [(home.x + x, home.y + y) for x, y in zone.polygon.points]
        await self._event(
            "mapping.started",
            {
                "route_points": len(plan.route),
                "coverage_target": plan.request.mapping_coverage_target,
                "message": "开始覆盖给定区域并同步建立 LiDAR 占据层、拓扑节点和 VLM 语义层",
            },
            run.id,
        )
        for route_index, route_point in enumerate(plan.route):
            while True:
                directive = await self._handle_control(run, control)
                if directive == "return_home":
                    return False, observation_index, coverage_ratio
                if not control.altitude_directive.empty():
                    await adapter.request(
                        "cancel", vehicle_name=profile.vehicle_name, timeout=3
                    )
                    altitude = control.altitude_directive.get_nowait()
                    await self._execute_altitude_directive(
                        run,
                        plan,
                        zone,
                        home,
                        control,
                        altitude,
                        planner_state,
                        None,
                    )
                    # A vertical action chunk changes both the next route target
                    # and the local planner's motion history. Replan from the
                    # newly reached pose instead of resuming a stale command.
                    planner_state = LocalPlannerState()
                    continue
                world_point = self._from_home(route_point.position, home)
                if control.altitude_override_m is not None:
                    world_point = world_point.model_copy(
                        update={"z": home.z - control.altitude_override_m}
                    )
                try:
                    await self._move_segmented(
                        run,
                        plan,
                        zone,
                        world_point,
                        home,
                        control,
                        planner_state=planner_state,
                        interrupt_on_altitude=True,
                    )
                except AltitudeChangeRequested:
                    # Stop the outstanding horizontal SimpleFlight command
                    # before executing the vertical action chunk.
                    await adapter.request(
                        "cancel", vehicle_name=profile.vehicle_name, timeout=3
                    )
                    continue
                break
            telemetry = await self._telemetry(run, plan=plan, zone=zone, home=home)
            await self._lidar_points(
                run,
                profile.vehicle_name,
                vehicle_position=telemetry.position,
            )
            yaw_options = zone.observation_yaws_deg or [0.0]
            yaw = yaw_options[route_index % len(yaw_options)]
            await adapter.request(
                "rotate_yaw",
                vehicle_name=profile.vehicle_name,
                yaw_degrees=yaw,
                timeout=10,
            )
            observation_index += 1
            try:
                await self._capture_assessment(
                    run, plan, zone, home, observation_index
                )
                model_successes += 1
            except Exception as error:
                await self._event(
                    "mapping.semantic_warning",
                    {
                        "route_index": route_index,
                        "error": f"{type(error).__name__}: {error}",
                        "message": "本观察点语义标注失败，继续执行确定性 LiDAR 建图航线",
                    },
                    run.id,
                )
            lidar_coverage = self.simulator.mapper.coverage_ratio_in_polygon(
                world_polygon
            )
            route_coverage = (route_index + 1) / total
            coverage_ratio = min(1.0, max(lidar_coverage, route_coverage))
            run = self.store.get_run(run.id) or run
            run = await self._update_task_progress(
                run,
                0,
                "running",
                message=f"区域建图 {route_index + 1}/{total}，覆盖率 {coverage_ratio:.0%}",
                coverage_ratio=coverage_ratio,
            )
            await self._event(
                "mapping.progress",
                {
                    "route_index": route_index,
                    "route_points": total,
                    "lidar_coverage_ratio": round(lidar_coverage, 4),
                    "coverage_ratio": round(coverage_ratio, 4),
                    "semantic_observations": model_successes,
                },
                run.id,
            )
        if model_successes == 0:
            raise MissionError("区域覆盖完成但没有成功的 VLM 语义观察")
        await self._event(
            "mapping.completed",
            {
                "coverage_ratio": coverage_ratio,
                "semantic_observations": model_successes,
                "map_stats": self.simulator.mapper.snapshot()["stats"],
                "message": "给定区域占据与语义拓扑图建立完成",
            },
            run.id,
        )
        return coverage_ratio >= plan.request.mapping_coverage_target, observation_index, coverage_ratio

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
        self.simulator.latest_frame = frame
        self._save_frame(run, frame)
        await self._event(
            "frame.preview",
            frame_preview_payload(frame, source="vlm_observation"),
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
        await self._update_semantic_map(run, plan, zone, home, frame, assessment)
        await self._event("vision.assessment", assessment.model_dump(mode="json"), run.id)
        return frame, telemetry, assessment

    async def _update_semantic_map(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        frame: CameraFrame,
        assessment: DetectionAssessment,
    ) -> None:
        observations = []
        if assessment.is_match and assessment.bbox_norm is not None:
            observations.append(
                (plan.request.target_text, assessment.confidence, assessment.bbox_norm)
            )
        observations.extend(
            (item.label, item.confidence, item.bbox_norm)
            for item in assessment.observed_objects
        )
        updated: list[dict] = []
        seen_boxes = []
        for label, confidence, bbox in observations:
            area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
            duplicate_box = False
            for previous in seen_boxes:
                intersection = max(
                    0.0, min(bbox.x_max, previous.x_max) - max(bbox.x_min, previous.x_min)
                ) * max(
                    0.0, min(bbox.y_max, previous.y_max) - max(bbox.y_min, previous.y_min)
                )
                previous_area = (previous.x_max - previous.x_min) * (
                    previous.y_max - previous.y_min
                )
                union = area + previous_area - intersection
                if union > 0 and intersection / union >= 0.72:
                    duplicate_box = True
                    break
            if duplicate_box:
                continue
            seen_boxes.append(bbox)
            try:
                position = localize_bbox(frame, bbox)
            except DepthLocalizationError:
                continue
            if distance(frame.camera_position, position) > 80.0:
                continue
            if not point_in_polygon(self._to_home(position, home), zone.polygon):
                continue
            landmark = self.simulator.mapper.integrate_semantic(
                label, position, confidence, frame.frame_id
            )
            if landmark:
                updated.append(landmark)
        if not updated:
            return
        await self._event(
            "map.semantic",
            {
                "frame_id": frame.frame_id,
                "objects": updated,
                "message": f"VLM 已向拓扑图更新 {len(updated)} 个语义物体",
            },
            run.id,
        )
        # Full map snapshots are intentionally live-only.  The compact semantic
        # event is persisted, and the final complete map is written as an artifact.
        await self.events.publish("map.update", self.simulator.mapper.snapshot(), run.id)

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
                run,
                plan,
                zone,
                frame,
                telemetry,
                assessment,
                home,
                require_fully_visible=True,
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

    async def _search_with_live_vlm(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
        observation_index: int,
    ) -> tuple[Vec3 | None, int]:
        """Fly deterministic short search chunks while VLM perception runs in parallel."""
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        state = SearchGuidanceState(observation_index=observation_index)
        stop_search = asyncio.Event()
        search_task: asyncio.Task | None = None
        planner_state = LocalPlannerState()
        action_chunk = 0
        return_requested = False
        explored_radius = max(2.5, min(5.0, zone.lane_spacing_m * 0.55))
        # Seed the radar coverage layer at the panorama location before asking
        # the VLM to rank frontiers. Subsequent short action chunks rescan and
        # dynamically skip any waypoint that has already become covered.
        mapping_telemetry = await self._telemetry(
            run, plan=plan, zone=zone, home=home
        )
        await self._lidar_points(
            run,
            profile.vehicle_name,
            vehicle_position=mapping_telemetry.position,
        )
        world_route = [
            (route_point, self._from_home(route_point.position, home))
            for route_point in plan.route
        ]
        unexplored_route = [
            item
            for item in world_route
            if not self.simulator.mapper.is_explored(item[1], explored_radius)
        ]
        explored_route = [item for item in world_route if item not in unexplored_route]
        # Keep the boustrophedon route as the safe fallback, but allow the
        # deterministic frontier scorer to promote a nearby, higher-gain
        # waypoint.  This is deliberately local: it cannot splice a distant
        # swath into the next command or bypass the geofence validator.
        ordered_route = world_route
        ordered_route = await self._vlm_prioritize_topology_route(
            run, plan, zone, home, ordered_route, explored_radius
        )
        initial_route_reordered = [item[0].index for item in ordered_route] != [
            item[0].index for item in world_route
        ]
        search_task = asyncio.create_task(
            self._stream_vlm_search(run, plan, zone, home, state, stop_search),
            name=f"vlm-search-{run.id}",
        )
        await self._event(
            "search.path_started",
            {
                "mode": "deterministic_action_chunks_with_async_vlm",
                "route_points": len(ordered_route),
                "topology_unexplored_first": len(unexplored_route),
                "topology_deferred_visited": len(explored_route),
                "route_order": "adaptive_frontier_gain_with_boustrophedon_fallback",
                "global_route_reordering": initial_route_reordered,
                "radar_explored_cells": self.simulator.mapper.snapshot()["stats"][
                    "explored_cells"
                ],
                "segment_limit_m": plan.safety.avoidance_segment_m,
                "message": (
                    "Panorama found no target; mapped points are skipped and each "
                    "completed or blocked leg triggers a bounded frontier replan"
                ),
            },
            run.id,
        )
        try:
            # Give perception a chance to inspect the last panorama attitude
            # before the first new path chunk is committed.
            await asyncio.sleep(0)
            pending_route = list(ordered_route)
            while pending_route:
                route_point, world_point = pending_route.pop(0)
                if control.altitude_override_m is not None:
                    world_point = world_point.model_copy(
                        update={"z": home.z - control.altitude_override_m}
                    )
                if (
                    unexplored_route
                    and self.simulator.mapper.is_explored(world_point, explored_radius)
                ):
                    await self._event(
                        "search.waypoint_skipped",
                        {
                            "waypoint_id": f"route-{route_point.index}",
                            "coverage_ratio": round(
                                self.simulator.mapper.exploration_coverage(
                                    world_point, explored_radius
                                ),
                                3,
                            ),
                            "message": "LiDAR 已覆盖该航点，跳过回访并继续选择未探索前沿",
                        },
                        run.id,
                    )
                    continue
                waypoint_blocked = False
                while True:
                    directive = await self._handle_control(run, control)
                    if directive == "return_home":
                        return_requested = True
                        break
                    if state.fatal_error:
                        raise MissionError(state.fatal_error)
                    if state.target is not None:
                        break
                    if not control.altitude_directive.empty():
                        altitude = control.altitude_directive.get_nowait()
                        await self._execute_altitude_directive(
                            run,
                            plan,
                            zone,
                            home,
                            control,
                            altitude,
                            planner_state,
                            state,
                        )
                        pending_route = await self._vlm_prioritize_topology_route(
                            run,
                            plan,
                            zone,
                            home,
                            pending_route,
                            explored_radius,
                        )
                        waypoint_blocked = True
                        break
                    if not control.exploration_directive.empty():
                        manual = control.exploration_directive.get_nowait()
                        await self._execute_exploration_directive(
                            run,
                            plan,
                            zone,
                            home,
                            control,
                            manual,
                            planner_state,
                            state,
                        )
                        # Do not turn back toward the interrupted waypoint.
                        # Re-rank only the remaining, still-unexplored route
                        # from the new operator-selected viewpoint.
                        pending_route = [
                            item
                            for item in pending_route
                            if not self.simulator.mapper.is_explored(
                                item[1], explored_radius
                            )
                        ]
                        pending_route = await self._vlm_prioritize_topology_route(
                            run,
                            plan,
                            zone,
                            home,
                            pending_route,
                            explored_radius,
                        )
                        waypoint_blocked = True
                        break
                    telemetry = await self._telemetry(
                        run, plan=plan, zone=zone, home=home
                    )
                    remaining = distance(telemetry.position, world_point)
                    if remaining <= 1.5:
                        break
                    action_chunk += 1
                    await self._event(
                        "search.action_chunk",
                        {
                            "chunk": action_chunk,
                            "route_index": route_point.index,
                            "route_target": world_point.model_dump(),
                            "remaining_m": remaining,
                            "message": (
                                "Committed one bounded search chunk; async VLM may cancel it "
                                "immediately after a reliable lock"
                            ),
                        },
                        run.id,
                    )
                    try:
                        await self._move_segmented(
                            run,
                            plan,
                            zone,
                            world_point,
                            home,
                            control,
                            planner_state=planner_state,
                            search_guidance=state,
                            rolling_step=True,
                        )
                    except SearchTargetAcquired:
                        break
                    except SearchPathBlocked as error:
                        waypoint_blocked = True
                        # Clear obstacle-side hysteresis after abandoning this
                        # goal, but preserve the last actual flight heading so
                        # the next frontier scorer does not immediately choose
                        # a 180-degree return into the blocked corridor.
                        planner_state = LocalPlannerState(
                            last_heading_rad=planner_state.last_heading_rad
                        )
                        await adapter.request(
                            "cancel", vehicle_name=profile.vehicle_name, timeout=3
                        )
                        await self._event(
                            "search.waypoint_skipped",
                            {
                                "route_index": route_point.index,
                                "route_target": world_point.model_dump(),
                                "reason": str(error),
                                "message": (
                                    "Coverage waypoint is unreachable; selecting the next "
                                    "bounded VLM search path"
                                ),
                            },
                            run.id,
                        )
                        break
                    # The real provider naturally yields for inference. This
                    # explicit yield keeps the same concurrency contract in tests.
                    await asyncio.sleep(0)
                if return_requested or state.target is not None:
                    break
                if pending_route:
                    replan_telemetry = await self._telemetry(
                        run, plan=plan, zone=zone, home=home
                    )
                    pending_route, replan = self._rank_exploration_route(
                        replan_telemetry.position,
                        pending_route,
                        plan,
                        zone,
                        home,
                        explored_radius,
                        planner_state.last_heading_rad,
                    )
                    await self._event(
                        "search.route_replanned",
                        {
                            **replan,
                            "trigger": "blocked" if waypoint_blocked else "waypoint_completed",
                            "remaining_waypoints": len(pending_route),
                            "message": (
                                "Re-ranked nearby frontiers using endpoint gain, segment "
                                "coverage and actual trajectory revisit cost"
                            ),
                        },
                        run.id,
                    )
        finally:
            stop_search.set()
            if search_task:
                search_task.cancel()
                with suppress(asyncio.CancelledError):
                    await search_task
        if state.fatal_error:
            raise MissionError(state.fatal_error)
        if state.target is None:
            await self._event(
                "search.path_completed",
                {
                    "result": "not_found" if not return_requested else "return_home",
                    "action_chunks": action_chunk,
                },
                run.id,
            )
            return None, state.observation_index
        await adapter.request(
            "cancel", vehicle_name=profile.vehicle_name, timeout=3
        )
        await self._event(
            "search.target_acquired",
            {
                "target": state.target.model_dump(),
                "observation_index": state.observation_index,
                "action_chunks": action_chunk,
                "message": "VLM locked the target; current search chunk was cancelled",
            },
            run.id,
        )
        return state.target, state.observation_index

    @staticmethod
    def _exploration_target(
        position: Vec3, heading_degrees: float, distance_m: float
    ) -> Vec3:
        heading = math.radians(heading_degrees % 360)
        return Vec3(
            x=position.x + math.cos(heading) * distance_m,
            y=position.y + math.sin(heading) * distance_m,
            z=position.z,
        )

    async def _execute_exploration_directive(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
        directive: ExplorationDirective,
        planner_state: LocalPlannerState,
        search_guidance: SearchGuidanceState | None,
    ) -> None:
        telemetry = await self._telemetry(
            run, plan=plan, zone=zone, home=home
        )
        target = self._exploration_target(
            telemetry.position,
            directive.heading_degrees,
            directive.distance_m,
        )
        try:
            validate_position(self._to_home(target, home), zone, plan.safety)
            if not self._segment_is_allowed(
                telemetry.position, target, home, zone, plan
            ):
                raise SafetyViolation(
                    "requested exploration segment leaves the allowed flight area"
                )
            await self._event(
                "vlm.exploration_started",
                {
                    "heading_degrees": directive.heading_degrees,
                    "distance_m": directive.distance_m,
                    "target": target.model_dump(),
                    "message": (
                        f"执行 VLM 对话探索指令：航向 {directive.heading_degrees:.0f}°，"
                        f"距离 {directive.distance_m:.1f} m"
                    ),
                },
                run.id,
            )
            await self._move_segmented(
                run,
                plan,
                zone,
                target,
                home,
                control,
                look_at=target,
                planner_state=planner_state,
                search_guidance=search_guidance,
                rolling_step=False,
            )
        except SearchTargetAcquired:
            raise
        except (SafetyViolation, SearchPathBlocked) as error:
            await self._event(
                "vlm.exploration_rejected",
                {
                    "heading_degrees": directive.heading_degrees,
                    "distance_m": directive.distance_m,
                    "reason": str(error),
                    "message": "对话探索指令被实时安全或避障校验拒绝",
                },
                run.id,
            )
            return
        await self._event(
            "vlm.exploration_completed",
            {
                "heading_degrees": directive.heading_degrees,
                "distance_m": directive.distance_m,
                "target": target.model_dump(),
                "message": "对话探索航段完成；从新视点重排未探索前沿",
            },
            run.id,
        )

    async def _execute_altitude_directive(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
        directive: AltitudeDirective,
        planner_state: LocalPlannerState,
        search_guidance: SearchGuidanceState | None,
    ) -> None:
        profile = self.simulator.active_profile
        adapter = self.simulator.adapter
        assert profile and adapter
        telemetry = await self._telemetry(run, plan=plan, zone=zone, home=home)
        target = Vec3(
            x=telemetry.position.x,
            y=telemetry.position.y,
            z=home.z - directive.target_altitude_m,
        )
        try:
            validate_position(self._to_home(target, home), zone, plan.safety)
            if not self._segment_is_allowed(
                telemetry.position, target, home, zone, plan
            ):
                raise SafetyViolation(
                    "requested altitude change leaves the allowed flight area"
                )
            if plan.safety.obstacle_avoidance_enabled:
                points = await self._lidar_points(
                    run,
                    profile.vehicle_name,
                    vehicle_position=telemetry.position,
                )
                assessment = assess_corridor(
                    telemetry.position,
                    target,
                    points,
                    plan.safety.min_clearance_m,
                )
                await self._event(
                    "avoidance.scan",
                    {
                        "from": telemetry.position.model_dump(),
                        "to": target.model_dump(),
                        "point_count": len(points),
                        "minimum_clearance_m": assessment.minimum_clearance_m,
                        "required_clearance_m": plan.safety.min_clearance_m,
                        "blocked": assessment.blocked,
                        "geofence_clear": True,
                        "motion": "vertical_altitude_change",
                    },
                    run.id,
                )
                if assessment.blocked:
                    raise SafetyViolation(
                        "LiDAR found an occupied vertical altitude corridor"
                    )
            await self._event(
                "vlm.altitude_started",
                {
                    "target_altitude_m": directive.target_altitude_m,
                    "requested_delta_m": directive.requested_delta_m,
                    "target": target.model_dump(),
                    "message": (
                        f"执行 VLM 高度指令：目标相对起飞点高度 "
                        f"{directive.target_altitude_m:.1f} m"
                    ),
                },
                run.id,
            )
            # AirSim's generic moveToPosition future can report completion for
            # a short vertical adjustment while horizontal cruise inertia still
            # places the vehicle outside its position-accuracy radius. Use the
            # dedicated Z controller and verify altitude from fresh telemetry.
            await adapter.request(
                "move_to_z",
                vehicle_name=profile.vehicle_name,
                z=target.z,
                speed=min(1.5, plan.safety.max_speed_mps),
                timeout=30,
            )
            reached = await self._wait_altitude_target(
                run,
                plan,
                zone,
                home,
                control,
                directive.target_altitude_m,
                timeout=30,
                search_guidance=search_guidance,
            )
            await adapter.request(
                "hover", vehicle_name=profile.vehicle_name, timeout=10
            )
            reached_altitude = home.z - reached.position.z
            if abs(reached_altitude - directive.target_altitude_m) > 0.8:
                raise SearchPathBlocked(
                    "vehicle did not reach the requested altitude"
                )
        except SearchTargetAcquired:
            raise
        except (SafetyViolation, SearchPathBlocked) as error:
            await self._event(
                "vlm.altitude_rejected",
                {
                    "target_altitude_m": directive.target_altitude_m,
                    "requested_delta_m": directive.requested_delta_m,
                    "reason": str(error),
                    "message": "高度指令被实时安全或避障校验拒绝",
                },
                run.id,
            )
            return
        control.altitude_override_m = directive.target_altitude_m
        await self._event(
            "vlm.altitude_completed",
            {
                "target_altitude_m": directive.target_altitude_m,
                "requested_delta_m": directive.requested_delta_m,
                "message": "高度调整完成；后续搜索航段保持该安全高度",
            },
            run.id,
        )

    async def _wait_altitude_target(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        control: RunControl,
        target_altitude_m: float,
        *,
        timeout: float,
        search_guidance: SearchGuidanceState | None = None,
        tolerance_m: float = 0.5,
    ) -> Telemetry:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if search_guidance and search_guidance.fatal_error:
                raise MissionError(search_guidance.fatal_error)
            if search_guidance and search_guidance.target is not None:
                raise SearchTargetAcquired()
            await self._handle_control(run, control)
            telemetry = await self._telemetry(run, plan=plan, zone=zone, home=home)
            altitude_m = home.z - telemetry.position.z
            if abs(altitude_m - target_altitude_m) <= tolerance_m:
                return telemetry
            await asyncio.sleep(0.2)
        raise SearchPathBlocked("vehicle did not reach the requested altitude before timeout")

    def _rank_exploration_route(
        self,
        current: Vec3,
        route: list[tuple[RoutePoint, Vec3]],
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        explored_radius: float,
        previous_heading_rad: float | None = None,
    ) -> tuple[list[tuple[RoutePoint, Vec3]], dict]:
        """Promote one nearby frontier using deterministic map-aware costs.

        Endpoint coverage alone cannot prevent an unexplored waypoint from
        being reached through a repeatedly flown corridor.  Candidate scores
        therefore include LiDAR coverage along the complete connector and a
        separate actual-trajectory revisit ratio.  Only a bounded local window
        is considered and every promoted connector must remain in the safety
        envelope.
        """
        if len(route) <= 1:
            selected_id = f"route-{route[0][0].index}" if route else None
            return route, {
                "selected_waypoint_id": selected_id,
                "route_reordered": False,
                "candidate_count": len(route),
            }

        mapper = self.simulator.mapper
        max_connector_m = max(12.0, min(28.0, zone.lane_spacing_m * 3.0))
        candidate_window = min(len(route), 10)
        candidates: list[dict] = []
        for pending_index, (route_point, point) in enumerate(route[:candidate_window]):
            connector_m = math.hypot(point.x - current.x, point.y - current.y)
            if pending_index > 0 and connector_m > max_connector_m:
                continue
            if not self._segment_is_allowed(current, point, home, zone, plan):
                continue
            endpoint_coverage = mapper.exploration_coverage(point, explored_radius)
            segment_coverage = mapper.segment_exploration_coverage(current, point)
            revisit_ratio = mapper.trajectory_revisit_ratio(current, point)
            heading_change = 0.0
            if previous_heading_rad is not None and connector_m > 1e-6:
                heading = math.atan2(point.y - current.y, point.x - current.x)
                heading_change = abs(
                    (heading - previous_heading_rad + math.pi) % (2 * math.pi) - math.pi
                )
            # Actual path re-entry is the strongest penalty. LiDAR coverage is
            # weaker because a visible free cell may never have been traversed.
            # A small order penalty preserves sweep continuity when candidates
            # have similar information gain.
            score = (
                endpoint_coverage * 16.0
                + segment_coverage * 7.0
                + revisit_ratio * 22.0
                + connector_m * 0.16
                + heading_change * 2.2
                + pending_index * 0.4
            )
            candidates.append(
                {
                    "pending_index": pending_index,
                    "waypoint_id": f"route-{route_point.index}",
                    "score": score,
                    "endpoint_coverage": endpoint_coverage,
                    "segment_coverage": segment_coverage,
                    "trajectory_revisit_ratio": revisit_ratio,
                    "connector_m": connector_m,
                    "heading_change_degrees": math.degrees(heading_change),
                }
            )
        if not candidates:
            return route, {
                "selected_waypoint_id": f"route-{route[0][0].index}",
                "route_reordered": False,
                "candidate_count": 0,
                "reason": "no_safe_local_frontier",
            }
        scored_candidate_count = len(candidates)
        forward_candidates = [
            item
            for item in candidates
            if previous_heading_rad is None
            or item["heading_change_degrees"] <= 100.0
        ]
        heading_reversal_deferred = bool(forward_candidates) and len(
            forward_candidates
        ) < len(candidates)
        if forward_candidates:
            candidates = forward_candidates
        selected = min(candidates, key=lambda item: (item["score"], item["pending_index"]))
        selected_index = int(selected["pending_index"])
        reordered = list(route)
        if selected_index > 0:
            selected_item = reordered.pop(selected_index)
            reordered.insert(0, selected_item)
        return reordered, {
            "selected_waypoint_id": selected["waypoint_id"],
            "route_reordered": selected_index > 0,
            "candidate_count": len(candidates),
            "scored_candidate_count": scored_candidate_count,
            "heading_reversal_deferred": heading_reversal_deferred,
            "score": round(float(selected["score"]), 3),
            "endpoint_coverage": round(float(selected["endpoint_coverage"]), 3),
            "segment_coverage": round(float(selected["segment_coverage"]), 3),
            "trajectory_revisit_ratio": round(
                float(selected["trajectory_revisit_ratio"]), 3
            ),
            "connector_m": round(float(selected["connector_m"]), 2),
            "heading_change_degrees": round(
                float(selected["heading_change_degrees"]), 1
            ),
        }

    async def _vlm_prioritize_topology_route(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        route: list[tuple[RoutePoint, Vec3]],
        explored_radius: float,
    ) -> list[tuple[RoutePoint, Vec3]]:
        """Combine deterministic frontier costs with one bounded VLM preference.

        The scorer performs the actual safe route promotion. The VLM sees only
        that bounded candidate set and may promote one of its members when the
        direct connector is short and remains inside the safety envelope.
        """
        if not route:
            return route
        # UI/report maps stay conservative (two observations). Exploration may
        # also use one-frame semantic hints, explicitly marked unconfirmed, so
        # a distant object seen during the first mission is not discarded.
        topology = self.simulator.mapper.snapshot(include_tentative_semantics=True)
        place_positions = [
            Vec3.model_validate(node["position"])
            for node in topology.get("nodes", [])
            if node.get("kind") == "place"
        ]
        # Descriptions commonly add exclusions in parentheses, e.g.
        # "cone (not the orange sphere)". Only the affirmative head phrase
        # defines the target category; otherwise the exclusion can win because
        # category matching is intentionally deterministic.
        target_head = plan.request.target_text.split("（", 1)[0].split("(", 1)[0]
        target_category = self.simulator.mapper._object_category(target_head)
        target_clue_positions = [
            Vec3.model_validate(node["position"])
            for node in topology.get("nodes", [])
            if node.get("kind") == "object"
            and target_category is not None
            and self.simulator.mapper._object_category(str(node.get("label", "")))
            == target_category
        ]
        current = await self._telemetry(run, plan=plan, zone=zone, home=home)
        route, deterministic_replan = self._rank_exploration_route(
            current.position,
            route,
            plan,
            zone,
            home,
            explored_radius,
        )
        max_lookahead_path_m = max(24.0, min(48.0, zone.lane_spacing_m * 5.0))
        local_route: list[tuple[RoutePoint, Vec3]] = []
        local_path_m = 0.0
        cursor = current.position
        for item in route:
            edge_m = math.hypot(item[1].x - cursor.x, item[1].y - cursor.y)
            if (
                local_route
                and local_path_m + edge_m > max_lookahead_path_m
                and len(local_route) >= 2
            ):
                break
            local_route.append(item)
            local_path_m += edge_m
            cursor = item[1]
            if len(local_route) >= 8:
                break

        candidates = []
        for local_index, (route_point, world_point) in enumerate(local_route):
            novelty = min(
                (
                    math.hypot(
                        world_point.x - place.x,
                        world_point.y - place.y,
                    )
                    for place in place_positions
                ),
                default=50.0,
            )
            # Adjacency follows route order, never geometric proximity across
            # a different swath. Nearby points on opposite lanes are not safe
            # substitutes for a forward connector.
            neighbors = []
            if local_index > 0:
                neighbors.append(f"route-{local_route[local_index - 1][0].index}")
            if local_index + 1 < len(local_route):
                neighbors.append(f"route-{local_route[local_index + 1][0].index}")
            candidate_payload = {
                "id": f"route-{route_point.index}",
                "position": {
                    "x": round(world_point.x, 2),
                    "y": round(world_point.y, 2),
                    "z": round(world_point.z, 2),
                },
                "explored": self.simulator.mapper.is_explored(
                    world_point, explored_radius
                ),
                "novelty_m": round(novelty, 1),
                "coverage_ratio": round(
                    self.simulator.mapper.exploration_coverage(
                        world_point, explored_radius
                    ),
                    3,
                ),
                "segment_coverage_ratio": round(
                    self.simulator.mapper.segment_exploration_coverage(
                        current.position, world_point
                    ),
                    3,
                ),
                "trajectory_revisit_ratio": round(
                    self.simulator.mapper.trajectory_revisit_ratio(
                        current.position, world_point
                    ),
                    3,
                ),
                "neighbors": neighbors,
            }
            candidate_payload["mapping_status"] = (
                "mapped"
                if candidate_payload["explored"]
                else (
                    "unmapped_frontier"
                    if candidate_payload["coverage_ratio"] < 0.1
                    else "partially_mapped_frontier"
                )
            )
            if target_clue_positions:
                candidate_payload["target_clue_distance_m"] = round(
                    min(
                        math.hypot(
                            world_point.x - clue.x,
                            world_point.y - clue.y,
                        )
                        for clue in target_clue_positions
                    ),
                    1,
                )
            candidates.append(candidate_payload)
        try:
            decision = await self.provider.plan_exploration(
                plan.request.target_text, topology, candidates
            )
            self.store.append_jsonl(
                Path(run.artifact_dir) / "model_calls.jsonl",
                {
                    "provider": self.provider.name,
                    "model": self.settings.llm_model or "mock",
                    "phase": "topology_exploration",
                    "target_text": plan.request.target_text,
                    "candidate_count": len(candidates),
                    "decision": decision.model_dump(mode="json"),
                },
            )
        except Exception as error:
            await self._event(
                "model.error",
                {
                    "phase": "topology_exploration",
                    "error": f"{type(error).__name__}: {error}",
                    "message": "拓扑 VLM 规划失败，继续确定性连续覆盖路线",
                },
                run.id,
            )
            return route

        candidate_ids = [candidate["id"] for candidate in candidates]
        unexplored_goal_ids = {
            candidate["id"] for candidate in candidates if not candidate["explored"]
        }
        current_clue_distance = min(
            (
                math.hypot(
                    current.position.x - clue.x,
                    current.position.y - clue.y,
                )
                for clue in target_clue_positions
            ),
            default=None,
        )
        accepted_goal_ids: list[str] = []
        rejected_ids: list[str] = []
        deterministic_seed_id = candidate_ids[0]
        for waypoint_id in decision.waypoint_ids:
            if waypoint_id not in candidate_ids or waypoint_id in accepted_goal_ids:
                rejected_ids.append(waypoint_id)
                continue
            if unexplored_goal_ids and waypoint_id not in unexplored_goal_ids:
                rejected_ids.append(waypoint_id)
                continue
            accepted_goal_ids.append(waypoint_id)
            if len(accepted_goal_ids) >= 1:
                break

        preferred_goal_id = (
            accepted_goal_ids[0]
            if accepted_goal_ids
            else next(
                (
                    candidate_id
                    for candidate_id in candidate_ids
                    if candidate_id in unexplored_goal_ids
                ),
                deterministic_seed_id,
            )
        )
        preferred_index = candidate_ids.index(preferred_goal_id)
        vlm_reordered = False
        vlm_rejection_reason = None
        reordered_route = list(route)
        if preferred_index > 0:
            preferred_point = local_route[preferred_index][1]
            connector_m = math.hypot(
                preferred_point.x - current.position.x,
                preferred_point.y - current.position.y,
            )
            max_direct_connector_m = max(
                12.0, min(28.0, zone.lane_spacing_m * 3.0)
            )
            seed_revisit = float(candidates[0]["trajectory_revisit_ratio"])
            preferred_revisit = float(
                candidates[preferred_index]["trajectory_revisit_ratio"]
            )
            if connector_m > max_direct_connector_m:
                vlm_rejection_reason = "connector_too_long"
            elif preferred_revisit > seed_revisit + 0.15:
                vlm_rejection_reason = "connector_revisits_more_flown_space"
            elif not self._segment_is_allowed(
                current.position, preferred_point, home, zone, plan
            ):
                vlm_rejection_reason = "connector_outside_safety_envelope"
            else:
                preferred_route_index = next(
                    index
                    for index, (route_point, _point) in enumerate(reordered_route)
                    if f"route-{route_point.index}" == preferred_goal_id
                )
                preferred_item = reordered_route.pop(preferred_route_index)
                reordered_route.insert(0, preferred_item)
                vlm_reordered = True
        planned_prefix_ids = [
            f"route-{route_point.index}"
            for route_point, _point in reordered_route[: max(1, preferred_index + 1)]
        ]
        await self._event(
            "search.topology_vlm_plan",
            {
                "selected_waypoint_ids": accepted_goal_ids,
                "preferred_local_goal_id": preferred_goal_id,
                "connector_waypoint_ids": (
                    [] if vlm_reordered else planned_prefix_ids[:-1]
                ),
                "planned_waypoint_ids": planned_prefix_ids,
                "rejected_waypoint_ids": rejected_ids,
                "candidate_count": len(candidates),
                "unmapped_frontier_count": len(unexplored_goal_ids),
                "target_clue_count": len(target_clue_positions),
                "deterministic_seed_id": deterministic_seed_id,
                "initial_target_clue_distance_m": current_clue_distance,
                "local_lookahead_path_m": round(local_path_m, 2),
                "max_local_lookahead_path_m": max_lookahead_path_m,
                "deterministic_replan": deterministic_replan,
                "vlm_preference_applied": vlm_reordered,
                "vlm_preference_rejection_reason": vlm_rejection_reason,
                "global_route_reordered": bool(
                    deterministic_replan.get("route_reordered") or vlm_reordered
                ),
                "rationale": decision.rationale,
                "fallback_appended": len(route) - len(planned_prefix_ids),
                "message": (
                    "确定性前沿评分负责选择安全局部航路；只有不增加重访代价且不违反"
                    "安全包线时，VLM 的局部偏好才会应用到下一航点"
                ),
            },
            run.id,
        )
        return reordered_route

    @staticmethod
    def _topology_connector(
        cursor: Vec3,
        goal_id: str,
        by_id: dict[str, tuple[RoutePoint, Vec3]],
        excluded_ids: set[str],
        max_edge_m: float,
    ) -> list[str] | None:
        """Connect a VLM-selected goal through the deterministic safe route graph."""
        available = [waypoint_id for waypoint_id in by_id if waypoint_id not in excluded_ids]
        parents: dict[str, str | None] = {}
        queue: list[str] = []
        for waypoint_id in available:
            point = by_id[waypoint_id][1]
            if math.hypot(point.x - cursor.x, point.y - cursor.y) <= max_edge_m:
                parents[waypoint_id] = None
                queue.append(waypoint_id)
        offset = 0
        while offset < len(queue):
            waypoint_id = queue[offset]
            offset += 1
            if waypoint_id == goal_id:
                path: list[str] = []
                cursor_id: str | None = waypoint_id
                while cursor_id is not None:
                    path.append(cursor_id)
                    cursor_id = parents[cursor_id]
                return list(reversed(path))
            point = by_id[waypoint_id][1]
            for neighbor_id in available:
                if neighbor_id in parents:
                    continue
                neighbor = by_id[neighbor_id][1]
                if math.hypot(neighbor.x - point.x, neighbor.y - point.y) <= max_edge_m:
                    parents[neighbor_id] = waypoint_id
                    queue.append(neighbor_id)
        return None

    async def _stream_vlm_search(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        home: Vec3,
        state: SearchGuidanceState,
        stop_event: asyncio.Event,
    ) -> None:
        profile = self.simulator.active_profile
        assert profile
        await self._event(
            "search.stream_started",
            {
                "mode": "continuous_in_flight",
                "message": "VLM search stream started without inserting hover commands",
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
                    state.consecutive_model_errors = 0
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    state.consecutive_model_errors += 1
                    await self._event(
                        "model.error",
                        {
                            "phase": "search",
                            "consecutive_failures": state.consecutive_model_errors,
                            "error": f"{type(error).__name__}: {error}",
                        },
                        run.id,
                    )
                    if state.consecutive_model_errors >= 3:
                        state.fatal_error = "three consecutive model calls failed"
                        return
                    await asyncio.sleep(0 if profile.mode == "mock" else 0.25)
                    continue
                locked = await self._evaluate_candidate(
                    run,
                    plan,
                    zone,
                    frame,
                    telemetry,
                    assessment,
                    home,
                    require_fully_visible=True,
                )
                if locked is not None:
                    state.target = locked
                    await self._event(
                        "vision.locked",
                        {
                            "target": locked.model_dump(),
                            "frame_id": frame.frame_id,
                            "bbox_norm": assessment.bbox_norm.model_dump()
                            if assessment.bbox_norm
                            else None,
                            "confidence": assessment.confidence,
                            "mode": "asynchronous_search_stream",
                            "message": "Target locked during the bounded search path",
                        },
                        run.id,
                    )
                    return
                # Avoid provider request bursts while still sampling continuously.
                await asyncio.sleep(0 if profile.mode == "mock" else 0.25)
        finally:
            await self._event(
                "search.stream_stopped",
                {
                    "observation_index": state.observation_index,
                    "target_acquired": state.target is not None,
                },
                run.id,
            )

    async def _evaluate_candidate(
        self,
        run: RunRecord,
        plan: MissionPlan,
        zone: SearchZone,
        frame: CameraFrame,
        telemetry: Telemetry,
        assessment: DetectionAssessment,
        home: Vec3,
        require_fully_visible: bool = False,
    ) -> Vec3 | None:
        if not assessment.is_match or assessment.confidence < 0.75 or not assessment.bbox_norm:
            return None
        box = assessment.bbox_norm
        edge_margin = 0.01
        if require_fully_visible and (
            box.x_min <= edge_margin
            or box.y_min <= edge_margin
            or box.x_max >= 1.0 - edge_margin
            or box.y_max >= 1.0 - edge_margin
        ):
            await self._event(
                "vision.rejected",
                {
                    "reason": "initial target bounding box is clipped by the image edge",
                    "bbox_norm": box.model_dump(),
                    "message": "初始候选触及画面边缘，深度射线不可靠；无需居中，但必须完整可见",
                },
                run.id,
            )
            return None
        try:
            target = localize_bbox(frame, box)
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
        # A 3 m command can complete about 1.2-1.5 m before its requested
        # point in SimpleFlight.  A fixed 24-chunk ceiling therefore rejected
        # valid long approaches (the Blocks orange sphere was one chunk away
        # from standoff).  The mission deadline and the no-progress watchdog
        # remain the hard bounds.
        max_action_chunks = 64
        best_remaining = math.inf
        stalled_chunks = 0
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
                if remaining < best_remaining - 0.5:
                    best_remaining = remaining
                    stalled_chunks = 0
                else:
                    stalled_chunks += 1
                    if stalled_chunks >= 12:
                        raise SafetyViolation(
                            "continuous VLM action chunks made no approach progress"
                        )
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
                    guidance=guidance,
                    rolling_step=True,
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
        self.simulator.latest_frame = evidence
        self._save_frame(run, evidence, evidence=True)
        run = await self._set_state(
            run, RunState.EVIDENCE, target_position=current_target
        )
        if plan.request.end_policy in {"auto_rth", "land_at_target"}:
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
                    state.consecutive_misses += 1
                    now = asyncio.get_running_loop().time()
                    if now <= state.occlusion_expected_until:
                        # LiDAR has just diverted the vehicle around an
                        # obstacle.  Keep the last geometrically stable lock:
                        # a camera pointed at the target cannot see through a
                        # wall, and that expected occlusion is not evidence of
                        # a bad target coordinate.
                        state.consecutive_failures = 0
                        await self._event(
                            "guidance.target_occluded",
                            {
                                "observation_index": state.observation_index,
                                "occluded_frames": state.consecutive_misses,
                                "lock_retained": True,
                                "message": "绕障期间目标被遮挡；保留最后稳定锁定并继续滚动规划",
                            },
                            run.id,
                        )
                    else:
                        state.consecutive_update_rejections = 0
                        state.consecutive_failures += 1
                        await self._event(
                            "guidance.target_lost",
                            {
                                "observation_index": state.observation_index,
                                "consecutive_misses": state.consecutive_failures,
                                "message": "无遮挡预期时未得到可靠目标；保持当前动作块并后台复查",
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
                        state.consecutive_misses = 0
                        cluster = state.relocalization_candidates
                        if cluster and distance(cluster[-1], observed_target) > 3.0:
                            cluster.clear()
                        cluster.append(observed_target)
                        cluster[:] = cluster[-3:]
                        if len(cluster) >= 2 and all(
                            distance(cluster[0], candidate) <= 3.0
                            for candidate in cluster[1:]
                        ):
                            previous_target = state.target
                            count = len(cluster)
                            state.target = Vec3(
                                x=sum(candidate.x for candidate in cluster) / count,
                                y=sum(candidate.y for candidate in cluster) / count,
                                z=sum(candidate.z for candidate in cluster) / count,
                            )
                            state.consecutive_failures = 0
                            state.consecutive_update_rejections = 0
                            state.update_count += 1
                            cluster.clear()
                            await self._event(
                                "guidance.target_relocalized",
                                {
                                    "observation_index": state.observation_index,
                                    "target_generation": state.update_count,
                                    "previous_target": previous_target.model_dump(),
                                    "target": state.target.model_dump(),
                                    "consistent_jump_observations": count,
                                    "message": "连续跳变观测彼此一致，替换错误或过时的目标基准",
                                },
                                run.id,
                            )
                        else:
                            state.consecutive_update_rejections += 1
                            state.consecutive_failures = (
                                state.consecutive_update_rejections
                            )
                            await self._event(
                                "guidance.update_rejected",
                                {
                                    "observation_index": state.observation_index,
                                    "previous_target": state.target.model_dump(),
                                    "observed_target": observed_target.model_dump(),
                                    "update_distance_m": update_distance,
                                    "allowed_update_m": allowed_update,
                                    "consecutive_misses": state.consecutive_failures,
                                    "message": "单次 VLM 坐标跳变过大；等待一致观测，不打断当前动作块",
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
                        state.consecutive_misses = 0
                        state.consecutive_update_rejections = 0
                        state.relocalization_candidates.clear()
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
                    if state.consecutive_update_rejections >= 3:
                        state.fatal_error = (
                            "VLM guidance produced unsafe coordinate jumps three times"
                        )
                    else:
                        state.fatal_error = (
                            "VLM guidance lost the target for three unoccluded frames"
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
        await self._event(
            "flight.disarmed",
            {"location": "home", "message": "Landing confirmed and vehicle disarmed"},
            run.id,
        )
        return run

    async def _land_in_place(
        self, run: RunRecord, vehicle_name: str, ground_z: float
    ) -> RunRecord:
        adapter = self.simulator.adapter
        assert adapter
        run = await self._set_state(run, RunState.LANDING)
        await self._event(
            "flight.phase",
            {
                "phase": "land_at_target",
                "message": "Evidence complete; landing at the safe target standoff point",
            },
            run.id,
        )
        await adapter.request("cancel", vehicle_name=vehicle_name, timeout=3)
        await adapter.request("land", vehicle_name=vehicle_name, timeout=60)
        await self._wait_landed(
            run,
            RunControl(),
            timeout=60,
            ignore_controls=True,
            ground_z=ground_z,
        )
        await adapter.request("arm", vehicle_name=vehicle_name, armed=False, timeout=5)
        await self._event(
            "flight.disarmed",
            {
                "location": "target_standoff",
                "message": "In-place landing confirmed and vehicle disarmed",
            },
            run.id,
        )
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
        guidance: VlmGuidanceState | None = None,
        search_guidance: SearchGuidanceState | None = None,
        rolling_step: bool = False,
        arrival_tolerance_m: float = 1.5,
        interrupt_on_altitude: bool = False,
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
            if guidance and guidance.fatal_error:
                raise SafetyViolation(guidance.fatal_error)
            if search_guidance and search_guidance.fatal_error:
                raise MissionError(search_guidance.fatal_error)
            if search_guidance and search_guidance.target is not None:
                raise SearchTargetAcquired()
            if interrupt_on_altitude and not control.altitude_directive.empty():
                raise AltitudeChangeRequested()
            directive = await self._handle_control(run, control)
            if directive == "return_home":
                return
            current = telemetry.position
            remaining = distance(current, target)
            if remaining <= arrival_tolerance_m:
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
            planning_clearance_m = plan.safety.min_clearance_m + 0.35
            if plan.safety.obstacle_avoidance_enabled:
                points = await self._lidar_points(
                    run, profile.vehicle_name, vehicle_position=current
                )
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
                    "planned_clearance_m": planning_clearance_m,
                    "blocked": blocked,
                    "geofence_clear": direct_allowed,
                },
                run.id,
            )
            point = direct
            if blocked:
                replans += 1
                planner_state.consecutive_blocked_replans += 1
                if (
                    replans > plan.safety.avoidance_max_replans
                    or planner_state.consecutive_blocked_replans
                    > plan.safety.avoidance_max_replans
                ):
                    message = "local obstacle avoidance exceeded its replan limit"
                    if search_guidance is not None:
                        raise SearchPathBlocked(message)
                    raise SafetyViolation(message)
                if guidance:
                    guidance.occlusion_expected_until = max(
                        guidance.occlusion_expected_until,
                        asyncio.get_running_loop().time() + 8.0,
                    )
                detour_clearance_m = planning_clearance_m
                detour = choose_local_detour(
                    current,
                    look_at or target,
                    points,
                    detour_clearance_m,
                    segment_m,
                    lambda start, end: self._segment_is_allowed(
                        start, end, home, zone, plan
                    ),
                    preferred_side=planner_state.preferred_side,
                    previous_heading_rad=planner_state.last_heading_rad,
                    recent_waypoints=planner_state.recent_waypoints,
                    segment_revisit_cost=(
                        self.simulator.mapper.trajectory_revisit_ratio
                        if search_guidance is not None
                        else None
                    ),
                )
                recovery_replan = False
                if (
                    detour is None
                    and planner_state.recovery_replans < 3
                ):
                    # A wall-following detour can end next to the geofence.
                    # Once the vehicle has decelerated, allow a bounded escape
                    # that relaxes only heading hysteresis; clearance, LiDAR,
                    # geofence and recent-waypoint loop checks remain active.
                    recovery_replan = True
                    detour = choose_local_detour(
                        current,
                        look_at or target,
                        points,
                        detour_clearance_m,
                        segment_m,
                        lambda start, end: self._segment_is_allowed(
                            start, end, home, zone, plan
                        ),
                        preferred_side=(
                            -planner_state.preferred_side
                            if planner_state.preferred_side in (-1, 1)
                            else None
                        ),
                        previous_heading_rad=None,
                        recent_waypoints=planner_state.recent_waypoints,
                        segment_revisit_cost=(
                            self.simulator.mapper.trajectory_revisit_ratio
                            if search_guidance is not None
                            else None
                        ),
                    )
                if detour is None:
                    message = (
                        "LiDAR found an occupied flight corridor and no safe local detour"
                    )
                    if search_guidance is not None:
                        raise SearchPathBlocked(message)
                    raise SafetyViolation(message)
                if recovery_replan:
                    planner_state.recovery_replans += 1
                point = detour.waypoint
                planner_state.preferred_side = detour.side
                planner_state.recent_waypoints.append(point)
                planner_state.recent_waypoints = planner_state.recent_waypoints[
                    -16 if search_guidance is not None else -8:
                ]
                await self._event(
                    "avoidance.recovery" if recovery_replan else "avoidance.detour",
                    {
                        "blocked_target": direct.model_dump(),
                        "waypoint": point.model_dump(),
                        "side": "right" if detour.side > 0 else "left",
                        "angle_degrees": detour.angle_degrees,
                        "observed_clearance_m": detour.minimum_clearance_m,
                        "planned_clearance_m": detour_clearance_m,
                        "trajectory_revisit_ratio": round(detour.revisit_ratio, 3),
                        "replan": planner_state.consecutive_blocked_replans,
                        "recovery_replans": planner_state.recovery_replans,
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
            else:
                planner_state.consecutive_blocked_replans = 0
                planner_state.recovery_replans = 0
                planner_state.recent_waypoints = planner_state.recent_waypoints[
                    -12 if search_guidance is not None else -2:
                ]
            validate_position(self._to_home(point, home), zone, plan.safety)
            command_speed = min(
                speed or plan.safety.max_speed_mps,
                plan.safety.max_speed_mps,
            )
            near_obstacle = bool(
                assessment
                and assessment.minimum_clearance_m is not None
                and assessment.minimum_clearance_m
                < plan.safety.min_clearance_m + 1.0
            )
            if blocked or near_obstacle:
                # A local detour is deliberately slower and must decelerate
                # before the next LiDAR replan. This prevents SimpleFlight
                # inertia from carrying a just-clear path into the obstacle.
                command_speed = min(
                    command_speed, plan.safety.approach_speed_mps
                )
            arguments: dict[str, float | str] = {
                "vehicle_name": profile.vehicle_name,
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "speed": command_speed,
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
                guidance=guidance,
                search_guidance=search_guidance,
                settle_speed_mps=0.6 if blocked or near_obstacle else None,
                interrupt_on_altitude=interrupt_on_altitude,
            )
            if rolling_step:
                return
        message = "local obstacle avoidance did not converge on the destination"
        if search_guidance is not None:
            raise SearchPathBlocked(message)
        raise SafetyViolation(message)

    async def _lidar_points(
        self, run: RunRecord, vehicle_name: str, *, vehicle_position: Vec3
    ) -> list[Vec3]:
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
                self.simulator.mapper.integrate_lidar(points, vehicle_position)
                self.simulator.latest_lidar = point_cloud_preview_payload(
                    points, vehicle_position
                )
                await self._event(
                    "lidar.points", self.simulator.latest_lidar, run.id
                )
                await self.events.publish(
                    "map.update", self.simulator.mapper.snapshot(), run.id
                )
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
        guidance: VlmGuidanceState | None = None,
        search_guidance: SearchGuidanceState | None = None,
        settle_speed_mps: float | None = None,
        interrupt_on_altitude: bool = False,
    ) -> Telemetry:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if guidance and guidance.fatal_error:
                raise SafetyViolation(guidance.fatal_error)
            if search_guidance and search_guidance.fatal_error:
                raise MissionError(search_guidance.fatal_error)
            if search_guidance and search_guidance.target is not None:
                raise SearchTargetAcquired()
            if interrupt_on_altitude and not control.altitude_directive.empty():
                raise AltitudeChangeRequested()
            await self._handle_control(run, control)
            telemetry = await self._telemetry(run, plan=plan, zone=zone, home=home)
            speed = math.sqrt(
                telemetry.velocity.x**2
                + telemetry.velocity.y**2
                + telemetry.velocity.z**2
            )
            if distance(telemetry.position, target) <= tolerance and (
                settle_speed_mps is None or speed <= settle_speed_mps
            ):
                return telemetry
            await asyncio.sleep(0.2)
        # A rolling search chunk is only one disposable local-planner proposal.
        # SimpleFlight may fail to converge on a short point (most commonly after
        # a sharp heading reversal) while the vehicle, telemetry and VLM stream
        # are all still healthy.  Treating that as a mission-wide fault used to
        # abort compound searches halfway through their second target.  Let the
        # search loop cancel the stale command, mark this waypoint unreachable
        # and re-rank the remaining unexplored frontiers instead.  Approach, RTH
        # and operator commands retain the stricter mission-failure behaviour.
        if search_guidance is not None:
            raise SearchPathBlocked(
                "rolling search chunk did not converge before timeout"
            )
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
        ignore_collision: bool = False,
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
            self.simulator.mapper.integrate_pose(telemetry.position)
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
                # AirSim reports the multirotor body origin, not its lowest
                # contact point. In Blocks the stationary body origin is about
                # 0.68 m below the NED takeoff origin at ground contact.
                and abs(telemetry.position.z - ground_z) <= 1.0
                and speed <= 0.5
            )
            if telemetry.collision and not near_ground and not ignore_collision:
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
        self.simulator.latest_telemetry = telemetry
        validate_telemetry(telemetry)
        age = (utc_now() - telemetry.timestamp).total_seconds()
        if age > profile.safety.telemetry_stale_seconds:
            raise SafetyViolation(f"telemetry is stale by {age:.2f} seconds")
        if plan is not None and zone is not None and home is not None:
            validate_position(self._to_home(telemetry.position, home), zone, plan.safety)
        self.simulator.mapper.integrate_pose(telemetry.position)
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

    async def queue_exploration(
        self, run_id: str, heading_degrees: float, distance_m: float
    ) -> ExplorationDirective:
        run = self.store.get_run(run_id)
        control = self._controls.get(run_id)
        if not run or not control:
            raise MissionError("active run does not exist")
        if run.state != RunState.SEARCHING:
            raise MissionError("directional exploration is only allowed while SEARCHING")
        if not math.isfinite(heading_degrees) or not math.isfinite(distance_m):
            raise MissionError("exploration heading and distance must be finite")
        if distance_m < 1 or distance_m > 20:
            raise MissionError("exploration distance must be between 1 and 20 m")
        plan = self.store.get_plan(run.plan_id)
        telemetry = self.simulator.latest_telemetry
        if not plan or not telemetry or not control.home_position:
            raise MissionError("current mission pose is not ready for directional exploration")
        zone = self._zone(plan)
        directive = ExplorationDirective(
            heading_degrees=heading_degrees % 360,
            distance_m=distance_m,
        )
        target = self._exploration_target(
            telemetry.position, directive.heading_degrees, directive.distance_m
        )
        try:
            validate_position(
                self._to_home(target, control.home_position), zone, plan.safety
            )
        except SafetyViolation as error:
            raise MissionError(str(error)) from error
        if not self._segment_is_allowed(
            telemetry.position, target, control.home_position, zone, plan
        ):
            raise MissionError("requested exploration segment leaves the allowed flight area")
        if control.exploration_directive.full():
            control.exploration_directive.get_nowait()
        control.exploration_directive.put_nowait(directive)
        await self._event(
            "vlm.exploration_queued",
            {
                "heading_degrees": directive.heading_degrees,
                "distance_m": directive.distance_m,
                "target": target.model_dump(),
                "message": "VLM 对话探索指令已进入安全 Action Chunk 队列",
            },
            run_id,
        )
        return directive

    async def queue_altitude(
        self,
        run_id: str,
        *,
        altitude_delta_m: float | None = None,
        target_altitude_m: float | None = None,
    ) -> AltitudeDirective:
        run = self.store.get_run(run_id)
        control = self._controls.get(run_id)
        if not run or not control:
            raise MissionError("active run does not exist")
        if run.state != RunState.SEARCHING:
            raise MissionError("altitude changes are only allowed while SEARCHING")
        if (altitude_delta_m is None) == (target_altitude_m is None):
            raise MissionError("provide exactly one altitude target or delta")
        values = [
            value
            for value in (altitude_delta_m, target_altitude_m)
            if value is not None
        ]
        if not all(math.isfinite(value) for value in values):
            raise MissionError("altitude request must be finite")
        if altitude_delta_m is not None and not -10 <= altitude_delta_m <= 10:
            raise MissionError("altitude delta must be between -10 and 10 m")
        plan = self.store.get_plan(run.plan_id)
        telemetry = self.simulator.latest_telemetry
        home = control.home_position
        if not plan or not telemetry or not home:
            raise MissionError("current mission pose is not ready for altitude control")
        current_altitude_m = home.z - telemetry.position.z
        resolved_altitude_m = (
            target_altitude_m
            if target_altitude_m is not None
            else current_altitude_m + float(altitude_delta_m)
        )
        if abs(resolved_altitude_m - current_altitude_m) < 0.1:
            raise MissionError("requested altitude is already reached")
        zone = self._zone(plan)
        target = Vec3(
            x=telemetry.position.x,
            y=telemetry.position.y,
            z=home.z - resolved_altitude_m,
        )
        try:
            validate_position(self._to_home(target, home), zone, plan.safety)
        except SafetyViolation as error:
            raise MissionError(str(error)) from error
        if not self._segment_is_allowed(telemetry.position, target, home, zone, plan):
            raise MissionError("requested altitude change leaves the allowed flight area")
        directive = AltitudeDirective(
            target_altitude_m=resolved_altitude_m,
            requested_delta_m=altitude_delta_m,
        )
        if control.altitude_directive.full():
            control.altitude_directive.get_nowait()
        control.altitude_directive.put_nowait(directive)
        await self._event(
            "vlm.altitude_queued",
            {
                "current_altitude_m": current_altitude_m,
                "target_altitude_m": directive.target_altitude_m,
                "requested_delta_m": directive.requested_delta_m,
                "message": "VLM 高度指令已进入安全 Action Chunk 队列",
            },
            run_id,
        )
        return directive

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
