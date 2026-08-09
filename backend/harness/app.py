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
from .llm import create_provider
from .mission import MissionError, MissionService
from .models import (
    ApiMessage,
    ArtifactsResponse,
    CandidateDecision,
    HealthResponse,
    MissionPlan,
    ProviderConfigRequest,
    ProviderConfigResponse,
    ProviderModelOption,
    RunRecord,
    RunState,
    SearchMissionRequest,
    SimulatorStartRequest,
    TERMINAL_STATES,
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


app = FastAPI(title="AirSim LLM Harness", version="0.1.0", lifespan=lifespan)


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
        "<h1>AirSim LLM Harness</h1><p>Frontend 尚未构建。运行 <code>npm --prefix frontend run build</code>。</p>"
    )
