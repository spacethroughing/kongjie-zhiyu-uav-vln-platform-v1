from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
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
        self.missions = MissionService(
            self.settings, self.store, self.events, self.simulator, self.provider
        )

    async def close(self) -> None:
        await self.missions.close()
        await self.simulator.close()
        self.store.close()


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


@app.post("/api/simulator/stop", response_model=ApiMessage)
async def stop_simulator() -> ApiMessage:
    active = [run for run in services().store.list_runs() if run.state not in TERMINAL_STATES]
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
