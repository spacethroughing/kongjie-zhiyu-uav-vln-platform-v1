from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import shutil
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

import numpy as np

from .models import CameraFrame, Quaternion, Telemetry, Vec3


class BridgeError(RuntimeError):
    pass


class VehicleAdapter(ABC):
    @abstractmethod
    async def request(self, operation: str, **arguments):
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    async def telemetry(self, vehicle_name: str) -> Telemetry:
        payload = await self.request("state", vehicle_name=vehicle_name)
        return Telemetry.model_validate(payload)

    async def capture(self, vehicle_name: str) -> CameraFrame:
        payload = await self.request("capture", vehicle_name=vehicle_name)
        return CameraFrame.model_validate(payload)


class JsonLineBridge(VehicleAdapter):
    def __init__(self, repo_root: Path, conda_env: str) -> None:
        self.repo_root = repo_root
        self.conda_env = conda_env
        self.process: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self.stderr_tail: list[str] = []

    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return
        script = self.repo_root / "bridge" / "airsim_bridge.py"
        command = self._python_command(script)
        environment = os.environ.copy()
        interpreter = Path(command[0])
        if interpreter.is_absolute() and interpreter.is_file():
            environment_root = interpreter.parent
            environment["PATH"] = os.pathsep.join(
                [
                    str(environment_root),
                    str(environment_root / "Library" / "bin"),
                    str(environment_root / "Scripts"),
                    environment.get("PATH", ""),
                ]
            )
        self.process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.repo_root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
            # A 640x360 PNG plus float depth data is much larger than asyncio's
            # default 64 KiB line limit; JSONL keeps each response on one line.
            limit=16 * 1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    def _python_command(self, script: Path) -> list[str]:
        """Prefer the environment interpreter so shutdown cannot orphan a conda child."""
        executable = "python.exe" if os.name == "nt" else "python"
        candidates: list[Path] = []
        conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if conda_executable:
            conda_root = Path(conda_executable).resolve().parent
            candidates.append(conda_root / "envs" / self.conda_env / executable)
        candidates.extend(
            [
                Path.home() / "anaconda3" / "envs" / self.conda_env / executable,
                Path.home() / "miniconda3" / "envs" / self.conda_env / executable,
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return [str(candidate), "-u", str(script)]
        return ["conda", "run", "--no-capture-output", "-n", self.conda_env, "python", "-u", str(script)]

    async def _read_stdout(self) -> None:
        assert self.process and self.process.stdout
        while line := await self.process.stdout.readline():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            future = self._pending.pop(str(message.get("id")), None)
            if future and not future.done():
                if message.get("ok"):
                    future.set_result(message.get("result"))
                else:
                    future.set_exception(BridgeError(message.get("error", "bridge request failed")))
        error = BridgeError("AirSim bridge exited")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def _read_stderr(self) -> None:
        assert self.process and self.process.stderr
        while line := await self.process.stderr.readline():
            self.stderr_tail.append(line.decode(errors="replace").rstrip())
            self.stderr_tail = self.stderr_tail[-40:]

    async def request(self, operation: str, timeout: float = 20, **arguments):
        if not self.process or self.process.returncode is not None:
            await self.start()
        assert self.process and self.process.stdin
        request_id = str(uuid4())
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = json.dumps({"id": request_id, "op": operation, "args": arguments}, separators=(",", ":"))
        self.process.stdin.write((payload + "\n").encode())
        await self.process.stdin.drain()
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        process, self.process = self.process, None
        if process and process.returncode is None:
            if process.stdin:
                process.stdin.close()
                try:
                    await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=4)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        for task in (self._reader_task, self._stderr_task):
            if task:
                task.cancel()
        self._reader_task = None
        self._stderr_task = None


class MockVehicleAdapter(VehicleAdapter):
    _PNG_1PX = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    def __init__(self) -> None:
        self.position = Vec3(x=0, y=0, z=0)
        self.armed = False
        self.landed = True
        self.connected = True
        self.target = Vec3(x=10, y=10, z=-5)

    async def request(self, operation: str, **arguments):
        if operation in {"connect", "ping"}:
            return {"connected": self.connected, "vehicles": [arguments.get("vehicle_name", "Drone1")]}
        if operation == "api_control":
            return {"enabled": bool(arguments.get("enabled", True))}
        if operation == "arm":
            self.armed = bool(arguments.get("armed", True))
            return {"armed": self.armed}
        if operation == "takeoff":
            self.landed = False
            self.position = Vec3(x=self.position.x, y=self.position.y, z=-5)
            return {"accepted": True}
        if operation == "land":
            self.position = Vec3(x=self.position.x, y=self.position.y, z=0)
            self.landed = True
            self.armed = False
            return {"accepted": True}
        if operation == "move_to":
            self.position = Vec3(x=arguments["x"], y=arguments["y"], z=arguments["z"])
            return {"accepted": True}
        if operation in {"hover", "cancel", "rotate_yaw"}:
            return {"accepted": True}
        if operation == "state":
            return {
                "position": self.position.model_dump(),
                "velocity": {"x": 0, "y": 0, "z": 0},
                "armed": self.armed,
                "landed": self.landed,
                "collision": False,
            }
        if operation == "capture":
            width, height = 64, 48
            dx = self.target.x - self.position.x
            dy = self.target.y - self.position.y
            dz = self.target.z - self.position.z
            yaw = math.atan2(dy, dx)
            horizontal = math.hypot(dx, dy)
            pitch = math.atan2(-dz, horizontal or 1e-9)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            orientation = Quaternion(w=cy * cp, x=-sy * sp, y=cy * sp, z=sy * cp)
            depth = np.full((height, width), math.sqrt(dx * dx + dy * dy + dz * dz), dtype="<f4")
            return {
                "frame_id": str(uuid4()),
                "width": width,
                "height": height,
                "scene_png_b64": self._PNG_1PX,
                "depth_f32_zlib_b64": base64.b64encode(zlib.compress(depth.tobytes())).decode(),
                "camera_position": self.position.model_dump(),
                "camera_orientation": orientation.model_dump(),
                "fov_degrees": 90,
            }
        if operation == "lidar_min":
            return {"minimum_m": 20.0, "point_count": 10}
        if operation == "lidar_scan":
            return {
                "point_cloud": [100.0, 100.0, -5.0],
                "point_count": 1,
                "sampled_point_count": 1,
                "data_frame": "VehicleInertialFrame",
            }
        if operation == "reset":
            self.position = Vec3(x=0, y=0, z=0)
            self.armed = False
            self.landed = True
            return {"accepted": True}
        raise BridgeError(f"unsupported mock operation: {operation}")

    async def close(self) -> None:
        self.connected = False
