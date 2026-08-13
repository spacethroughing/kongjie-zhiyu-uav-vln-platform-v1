import asyncio

import pytest

from harness.bridge import MockVehicleAdapter
from harness.config import REPO_ROOT, load_scenes
from harness.events import EventBus
from harness.simulator import SimulatorManager


class ExitProcess:
    def __init__(self) -> None:
        self.returncode = None
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def exit(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._exited.set()


class CountingAdapter(MockVehicleAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.connect_calls = 0

    async def request(self, operation: str, **arguments):
        if operation == "connect":
            self.connect_calls += 1
        return await super().request(operation, **arguments)


@pytest.mark.asyncio
async def test_real_simulator_readiness_requires_stable_rpc_and_vehicle(monkeypatch):
    manager = SimulatorManager(
        load_scenes(REPO_ROOT / "configs" / "scenes.json"), EventBus()
    )
    manager.process = ExitProcess()  # type: ignore[assignment]
    adapter = CountingAdapter()
    manager.adapter = adapter

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    await manager._wait_ready(manager.profiles["blocks"], timeout=1)

    assert adapter.connect_calls == 2


@pytest.mark.asyncio
async def test_unreal_exit_clears_false_ready_state_and_live_preview():
    events = EventBus()
    manager = SimulatorManager(
        load_scenes(REPO_ROOT / "configs" / "scenes.json"), events
    )
    profile = manager.profiles["blocks"]
    process = ExitProcess()
    adapter = MockVehicleAdapter()
    manager.active_profile = profile
    manager.process = process  # type: ignore[assignment]
    manager.adapter = adapter
    manager.state = "READY"
    manager._preview_task = asyncio.create_task(asyncio.sleep(60))

    watcher = asyncio.create_task(manager._watch_process(profile, process))
    process.exit(0)
    await asyncio.wait_for(watcher, timeout=1)

    assert manager.state == "STOPPED"
    assert manager.active_profile is None
    assert manager.process is None
    assert manager.adapter is None
    assert adapter.connected is False
    assert manager._preview_task is None
