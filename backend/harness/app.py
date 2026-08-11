from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import REPO_ROOT, Settings, load_scenes
from .events import EventBus
from .geometry import quaternion_yaw_degrees
from .llm import create_provider
from .mission import MissionError, MissionService
from .models import (
    ApiMessage,
    ArtifactsResponse,
    CandidateDecision,
    HealthResponse,
    MissionPlan,
    MissionPlanRevisionRequest,
    ProviderConfigRequest,
    ProviderConfigResponse,
    ProviderModelOption,
    RunRecord,
    RunState,
    SearchMissionRequest,
    SimulatorStartRequest,
    TERMINAL_STATES,
    VlmChatRequest,
    VlmChatResponse,
)
from .simulator import SimulatorError, SimulatorManager
from .store import Store


class ApplicationServices:
    def __init__(self) -> None:
        self.settings = Settings.from_env()
        self.profiles = load_scenes(self.settings.scenes_file)
        self.events = EventBus()
        self.store = Store(self.settings.data_dir / "harness.sqlite3", self.settings.runs_dir)
        for run in self.store.list_runs():
            if run.state not in TERMINAL_STATES:
                self.store.update_run(
                    run,
                    RunState.FAILED,
                    error="control plane restarted before the run reached a terminal state",
                    ended=True,
                )
        self.simulator = SimulatorManager(self.profiles, self.events)
        self.provider = create_provider(self.settings)
        self.provider_config_lock = asyncio.Lock()
        self.vlm_chat_lock = asyncio.Lock()
        self.missions = MissionService(
            self.settings, self.store, self.events, self.simulator, self.provider
        )

    async def close(self) -> None:
        await self.missions.close()
        await self.simulator.close()
        self.store.close()

    def active_runs(self) -> list[RunRecord]:
        return [run for run in self.store.list_runs() if run.state not in TERMINAL_STATES]

    def provider_config(self) -> ProviderConfigResponse:
        return ProviderConfigResponse(
            provider=self.provider.name,
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model or "mock",
            api_key_configured=bool(self.settings.llm_api_key),
            runtime_only=True,
            models=ZHIPU_VISION_MODELS,
        )

    async def configure_provider(
        self, request: ProviderConfigRequest
    ) -> ProviderConfigResponse:
        async with self.provider_config_lock:
            if self.active_runs():
                raise MissionError("cannot change the VLM provider during an active mission")
            supplied_key = (
                request.api_key.get_secret_value().strip() if request.api_key else ""
            )
            api_key = supplied_key or self.settings.llm_api_key
            if not api_key:
                raise ValueError("API Key is required because no backend key is configured")
            next_settings = replace(
                self.settings,
                provider="openai-compatible",
                llm_base_url=ZHIPU_BASE_URL,
                llm_model=request.model,
                llm_api_key=api_key,
            )
            candidate = create_provider(next_settings)
            try:
                await candidate.probe()
            except Exception:
                await candidate.close()
                raise
            if self.active_runs():
                await candidate.close()
                raise MissionError("a mission started while the VLM was being verified")
            previous = self.provider
            self.settings = next_settings
            self.provider = candidate
            self.missions.settings = next_settings
            self.missions.provider = candidate
            await previous.close()
            await self.events.publish(
                "provider.configured",
                {
                    "provider": candidate.name,
                    "model": request.model,
                    "api_key_configured": True,
                    "message": f"VLM 已切换为 {request.model}；API Key 仅保存在后端内存",
                },
            )
            return self.provider_config()


ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_VISION_MODELS = [
    ProviderModelOption(
        id="glm-4.6v-flashx",
        name="GLM-4.6V-FlashX",
        description="轻量高速视觉模型",
        billing="paid",
    ),
    ProviderModelOption(
        id="glm-4.6v-flash",
        name="GLM-4.6V-Flash",
        description="免费视觉模型，高峰期可能限流",
        billing="free",
    ),
    ProviderModelOption(
        id="glm-4.6v",
        name="GLM-4.6V",
        description="高性能视觉推理模型",
        billing="paid",
    ),
    ProviderModelOption(
        id="glm-5v-turbo",
        name="GLM-5V-Turbo",
        description="高性能多模态 Agent 模型",
        billing="paid",
    ),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = ApplicationServices()
    app.state.services = services
    yield
    await services.close()


app = FastAPI(
    title="空界智语—无人机视觉语言导航平台",
    version="0.1.0",
    lifespan=lifespan,
)


def services() -> ApplicationServices:
    return app.state.services


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    service = services()
    return HealthResponse(
        ok=True,
        simulator_state=service.simulator.state,
        active_scene_id=service.simulator.active_profile.id if service.simulator.active_profile else None,
        provider=service.provider.name,
    )


@app.get("/api/scenes")
async def list_scenes():
    return [profile.model_dump(mode="json") for profile in services().profiles.values()]


@app.get("/api/simulator/diagnostics")
async def simulator_diagnostics():
    simulator = services().simulator
    adapter = simulator.adapter
    return {
        "state": simulator.state,
        "scene_id": simulator.active_profile.id if simulator.active_profile else None,
        "bridge_pid": adapter.process.pid if getattr(adapter, "process", None) else None,
        "bridge_stderr_tail": list(getattr(adapter, "stderr_tail", [])),
    }


@app.post("/api/simulator/start")
async def start_simulator(request: SimulatorStartRequest):
    try:
        if request.scene_id not in services().profiles:
            raise SimulatorError(f"scene is not allowlisted: {request.scene_id}")
        await services().provider.probe()
        profile = await services().simulator.start(request.scene_id)
        return {"state": services().simulator.state, "scene": profile.model_dump(mode="json")}
    except SimulatorError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=424, detail=f"model capability probe failed: {error}") from error


@app.post("/api/provider/probe")
async def probe_provider():
    try:
        return await services().provider.probe()
    except Exception as error:
        raise HTTPException(status_code=424, detail=f"model capability probe failed: {error}") from error


@app.get("/api/provider/config", response_model=ProviderConfigResponse)
async def get_provider_config() -> ProviderConfigResponse:
    return services().provider_config()


@app.put("/api/provider/config", response_model=ProviderConfigResponse)
async def configure_provider(request: ProviderConfigRequest) -> ProviderConfigResponse:
    try:
        return await services().configure_provider(request)
    except MissionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=424,
            detail=f"new model capability probe failed; previous provider was kept: {error}",
        ) from error


@app.post("/api/simulator/stop", response_model=ApiMessage)
async def stop_simulator() -> ApiMessage:
    active = services().active_runs()
    if active:
        raise HTTPException(status_code=409, detail="an active mission must be aborted before stopping the simulator")
    await services().simulator.stop(hard=False)
    return ApiMessage(message="simulator stopped")


@app.post("/api/simulator/smoke")
async def smoke_simulator():
    active = [run for run in services().store.list_runs() if run.state not in TERMINAL_STATES]
    if active:
        raise HTTPException(status_code=409, detail="an active mission prevents a vehicle smoke test")
    try:
        return await services().simulator.smoke()
    except SimulatorError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/missions/plan", response_model=MissionPlan)
async def plan_mission(request: SearchMissionRequest) -> MissionPlan:
    try:
        return services().missions.create_plan(request)
    except (MissionError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.patch("/api/missions/{plan_id}", response_model=MissionPlan)
async def revise_mission_plan(
    plan_id: str,
    request: MissionPlanRevisionRequest,
) -> MissionPlan:
    try:
        return services().missions.revise_plan(
            plan_id,
            request.base_version,
            request.parameters,
        )
    except MissionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/missions/{plan_id}/approve", response_model=RunRecord)
async def approve_mission(plan_id: str) -> RunRecord:
    try:
        return await services().missions.approve(plan_id)
    except MissionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/runs", response_model=list[RunRecord])
async def list_runs() -> list[RunRecord]:
    return services().store.list_runs()


@app.get("/api/runs/{run_id}", response_model=RunRecord)
async def get_run(run_id: str) -> RunRecord:
    run = services().store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/api/runs/{run_id}/artifacts", response_model=ArtifactsResponse)
async def list_artifacts(run_id: str) -> ArtifactsResponse:
    if not services().store.get_run(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return ArtifactsResponse(run_id=run_id, files=services().store.list_artifacts(run_id))


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
async def get_artifact(run_id: str, artifact_path: str):
    run = services().store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    root = Path(run.artifact_dir).resolve()
    requested = (root / artifact_path).resolve()
    if root not in requested.parents or not requested.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(requested)


@app.post("/api/runs/{run_id}/candidate")
async def candidate_decision(run_id: str, request: CandidateDecision) -> ApiMessage:
    try:
        await services().missions.candidate_decision(run_id, request.decision)
        return ApiMessage(message=f"candidate {request.decision} recorded")
    except MissionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/runs/{run_id}/{action}", response_model=RunRecord)
async def run_control(run_id: str, action: str) -> RunRecord:
    allowed = {"pause", "resume", "return-home", "land", "abort", "hard-stop"}
    if action not in allowed:
        raise HTTPException(status_code=404, detail="unknown run action")
    try:
        return await services().missions.control(run_id, action)
    except MissionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/vlm/chat", response_model=VlmChatResponse)
async def vlm_chat(request: VlmChatRequest) -> VlmChatResponse:
    service = services()
    requested_run = service.store.get_run(request.run_id) if request.run_id else None
    if request.run_id and requested_run is None:
        raise HTTPException(status_code=404, detail="run not found")
    active_runs = service.active_runs()
    active_run = next(
        (run for run in active_runs if run.id == request.run_id),
        active_runs[0] if active_runs and request.run_id is None else None,
    )
    context_run = requested_run or active_run
    context_plan = (
        service.store.get_plan(context_run.plan_id) if context_run else None
    )
    context_target_text = (
        context_plan.request.target_text
        if active_run is not None and context_plan is not None
        else request.target_text
    )
    map_snapshot = service.simulator.mapper.snapshot(include_tentative_semantics=True)
    semantic_objects = [
        node
        for node in map_snapshot.get("nodes", [])
        if node.get("kind") == "object"
    ][-20:]
    telemetry = service.simulator.latest_telemetry
    frame = service.simulator.latest_frame if request.include_frame else None
    camera_yaw_degrees = (
        quaternion_yaw_degrees(frame.camera_orientation) if frame else None
    )
    context = {
        "simulator_state": service.simulator.state,
        "scene_id": (
            service.simulator.active_profile.id
            if service.simulator.active_profile
            else None
        ),
        "run_id": context_run.id if context_run else None,
        "run_state": context_run.state.value if context_run else None,
        "target_text": context_target_text,
        "target_text_source": "approved_mission" if active_run else "web_input",
        "camera_yaw_degrees": camera_yaw_degrees,
        "target_position": (
            context_run.target_position.model_dump(mode="json")
            if context_run and context_run.target_position
            else None
        ),
        "telemetry": telemetry.model_dump(mode="json") if telemetry else None,
        "map": {
            "stats": map_snapshot["stats"],
            "semantic_objects": semantic_objects,
        },
    }
    async with service.vlm_chat_lock:
        try:
            decision = await service.provider.chat(
                request.message, context, request.history, frame
            )
        except Exception as error:
            raise HTTPException(
                status_code=424,
                detail=f"VLM chat failed: {type(error).__name__}: {error}",
            ) from error

    executed_action = None
    command_error = None
    command_status = "none"
    mission_plan = None
    if decision.mission and request.execute_command:
        if active_run is not None:
            command_error = "活动任务执行中不能替换不可变任务计划，请先完成或终止当前任务"
            command_status = "rejected"
        else:
            scene_id = request.scene_id or context.get("scene_id")
            profile = service.simulator.profiles.get(scene_id or "")
            zone_id = request.zone_id or (profile.zones[0].id if profile and profile.zones else None)
            if not scene_id or not profile:
                command_error = "自然语言任务缺少有效仿真场景"
                command_status = "rejected"
            elif not zone_id:
                command_error = "自然语言任务缺少有效搜索区域"
                command_status = "rejected"
            else:
                try:
                    mission = decision.mission
                    target_text = (
                        "、".join(mission.targets)
                        if mission.kind == "target_search"
                        else "给定区域占据与语义拓扑图"
                    )
                    mission_plan = service.missions.create_plan(
                        SearchMissionRequest(
                            scene_id=scene_id,
                            zone_id=zone_id,
                            target_text=target_text,
                            mission_mode=mission.kind,
                            targets=mission.targets,
                            mapping_coverage_target=mission.coverage_target,
                            end_policy=(
                                request.end_policy
                                if mission.kind == "target_search"
                                else "auto_rth"
                            ),
                            safety_bounds=request.safety_bounds,
                        )
                    )
                    context_target_text = mission_plan.request.target_text
                    command_status = "planned"
                except (MissionError, ValueError) as error:
                    command_error = str(error)
                    command_status = "rejected"
    if decision.action and request.execute_command:
        if active_run is None:
            command_error = "没有可执行该指令的活动任务"
            command_status = "rejected"
        else:
            try:
                if decision.action == "explore":
                    assert decision.heading_degrees is not None
                    assert decision.distance_m is not None
                    await service.missions.queue_exploration(
                        active_run.id,
                        decision.heading_degrees,
                        decision.distance_m,
                    )
                    command_status = "queued"
                elif decision.action == "change-altitude":
                    await service.missions.queue_altitude(
                        active_run.id,
                        altitude_delta_m=decision.altitude_delta_m,
                        target_altitude_m=decision.target_altitude_m,
                    )
                    command_status = "queued"
                else:
                    await service.missions.control(active_run.id, decision.action)
                    command_status = "executed"
                executed_action = decision.action
            except MissionError as error:
                command_error = str(error)
                command_status = "rejected"
    await service.events.publish(
        "vlm.chat",
        {
            "message": "VLM 实时对话已响应",
            "requested_action": decision.action,
            "executed_action": executed_action,
            "command_error": command_error,
            "frame_used": frame is not None,
            "heading_degrees": decision.heading_degrees,
            "distance_m": decision.distance_m,
            "altitude_delta_m": decision.altitude_delta_m,
            "target_altitude_m": decision.target_altitude_m,
            "context_target_text": context_target_text,
            "command_status": command_status,
            "mission_kind": decision.mission.kind if decision.mission else None,
            "mission_plan_id": mission_plan.id if mission_plan else None,
            "task_count": len(mission_plan.tasks) if mission_plan else 0,
        },
        active_run.id if active_run else None,
    )
    return VlmChatResponse(
        reply=(
            f"{decision.reply}\n已生成 {len(mission_plan.tasks)} 步确定性任务计划，请在计划审核区确认后执行。"
            if mission_plan
            else decision.reply
        ),
        requested_action=decision.action,
        executed_action=executed_action,
        command_error=command_error,
        run_id=active_run.id if active_run else (context_run.id if context_run else None),
        frame_used=frame is not None,
        heading_degrees=decision.heading_degrees,
        distance_m=decision.distance_m,
        altitude_delta_m=decision.altitude_delta_m,
        target_altitude_m=decision.target_altitude_m,
        context_target_text=context_target_text,
        mission_intent=decision.mission,
        mission_plan=mission_plan,
        task_breakdown=mission_plan.tasks if mission_plan else [],
        command_status=command_status,
    )


@app.websocket("/api/ws")
async def websocket_events(websocket: WebSocket) -> None:
    await websocket.accept()
    service = services()
    await websocket.send_json(
        {
            "topic": "snapshot",
            "run_id": None,
            "sequence": 0,
            "timestamp": None,
            "payload": {
                "simulator_state": service.simulator.state,
                "active_scene_id": service.simulator.active_profile.id
                if service.simulator.active_profile
                else None,
                "runs": [run.model_dump(mode="json") for run in service.store.list_runs(10)],
                "map": service.simulator.mapper.snapshot(),
                "lidar": service.simulator.latest_lidar,
            },
        }
    )
    try:
        async for event in service.events.subscribe():
            await websocket.send_json(event.model_dump(mode="json"))
    except (WebSocketDisconnect, asyncio.CancelledError):
        return


FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{full_path:path}")
async def frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="not found")
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse(
        "<h1>空界智语—无人机视觉语言导航平台</h1><p>Frontend 尚未构建。运行 <code>npm --prefix frontend run build</code>。</p>"
    )
