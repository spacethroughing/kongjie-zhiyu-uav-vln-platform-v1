from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

from .avoidance import decode_point_cloud, point_cloud_preview_payload
from .bridge import JsonLineBridge, MockVehicleAdapter, VehicleAdapter
from .config import REPO_ROOT
from .events import EventBus
from .geometry import frame_preview_payload
from .mapping import TopologicalMapper
from .models import CameraFrame, SceneProfile, Telemetry


class SimulatorError(RuntimeError):
    pass


class SimulatorManager:
    def __init__(self, profiles: dict[str, SceneProfile], events: EventBus) -> None:
        self.profiles = profiles
        self.events = events
        self.active_profile: SceneProfile | None = None
        self.adapter: VehicleAdapter | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.state = "STOPPED"
        self.mapper = TopologicalMapper()
        self.latest_frame: CameraFrame | None = None
        self.latest_telemetry: Telemetry | None = None
        self.latest_lidar: dict | None = None
        self.mission_active = False
        self._lock = asyncio.Lock()
        self._preview_task: asyncio.Task | None = None

    async def start(self, scene_id: str) -> SceneProfile:
        async with self._lock:
            if self.active_profile:
                if self.active_profile.id == scene_id and self.state == "READY":
                    return self.active_profile
                raise SimulatorError("another simulator scene is already active")
            try:
                profile = self.profiles[scene_id]
            except KeyError as error:
                raise SimulatorError(f"scene is not allowlisted: {scene_id}") from error
            self.state = "STARTING"
            self.active_profile = profile
            self.mapper.reset()
            self.latest_frame = None
            self.latest_telemetry = None
            self.latest_lidar = None
            await self.events.publish("simulator.state", {"state": self.state, "scene_id": scene_id})
            try:
                if profile.mode == "mock":
                    self.adapter = MockVehicleAdapter()
                else:
                    self._validate_paths(profile)
                    args = [profile.executable]
                    if profile.mode == "editor":
                        args.append(profile.project)
                        if profile.map:
                            args.append(profile.map)
                        args.append("-game")
                    if profile.settings:
                        args.append(f"-settings={profile.settings}")
                    args.extend(profile.launch_args)
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                    self.process = await asyncio.create_subprocess_exec(
                        *args,
                        cwd=str(Path(profile.project or profile.executable).parent),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        creationflags=creationflags,
                    )
                    self.adapter = JsonLineBridge(REPO_ROOT, profile.bridge_conda_env)
                    await self.adapter.start()
                await self._wait_ready(profile)
                self.state = "READY"
                await self.events.publish("simulator.state", {"state": self.state, "scene_id": scene_id})
                self._preview_task = asyncio.create_task(
                    self._stream_preview(profile), name=f"preview-{scene_id}"
                )
                return profile
            except Exception:
                await self._stop_unlocked(hard=True)
                raise

    @staticmethod
    def _validate_paths(profile: SceneProfile) -> None:
        executable = Path(profile.executable or "")
        if not executable.is_absolute() or not executable.is_file():
            raise SimulatorError(f"configured executable does not exist: {executable}")
        if profile.mode == "editor":
            project = Path(profile.project or "")
            if not project.is_absolute() or not project.is_file() or project.suffix.lower() != ".uproject":
                raise SimulatorError(f"configured Unreal project is invalid: {project}")
        if profile.settings:
            settings = Path(profile.settings)
            if not settings.is_absolute() or not settings.is_file():
                raise SimulatorError(f"configured AirSim settings do not exist: {settings}")
            json.loads(settings.read_text(encoding="utf-8"))
        for checksum_path in profile.checksum_paths:
            artifact = Path(checksum_path)
            if not artifact.is_absolute() or not artifact.is_file():
                raise SimulatorError(f"configured scene fingerprint path does not exist: {artifact}")

    async def _wait_ready(self, profile: SceneProfile, timeout: float = 120) -> None:
        assert self.adapter
        deadline = asyncio.get_running_loop().time() + timeout
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if self.process and self.process.returncode is not None:
                raise SimulatorError(f"Unreal process exited with code {self.process.returncode}")
            try:
                result = await self.adapter.request("connect", timeout=3, vehicle_name=profile.vehicle_name)
                if result.get("connected"):
                    return
            except Exception as error:
                last_error = error
            await asyncio.sleep(2)
        raise SimulatorError(f"AirSim RPC did not become ready: {last_error}")

    async def stop(self, hard: bool = False) -> None:
        async with self._lock:
            await self._stop_unlocked(hard=hard)

    async def smoke(self) -> dict:
        """Run a deterministic takeoff/hover/capture/land check without invoking an LLM."""
        async with self._lock:
            if self.state != "READY" or not self.active_profile or not self.adapter:
                raise SimulatorError("simulator must be READY before a vehicle smoke test")
            profile = self.active_profile
            vehicle_name = profile.vehicle_name
            phase = "initialization"
            await self.events.publish("simulator.smoke", {"phase": "starting", "scene_id": profile.id})
            try:
                initial = await self._wait_ground_stable(vehicle_name)
                ground_z = initial.position.z
                await self.events.publish("telemetry", initial.model_dump(mode="json"))
                phase = "API control"
                await self.adapter.request("api_control", timeout=5, enabled=True, vehicle_name=vehicle_name)
                phase = "arming"
                await self.adapter.request("arm", timeout=5, armed=True, vehicle_name=vehicle_name)
                for attempt in range(1, 4):
                    phase = f"takeoff attempt {attempt}"
                    await self.adapter.request("takeoff", timeout=5, vehicle_name=vehicle_name)
                    try:
                        await self._wait_vehicle(
                            # NED origins differ by map; evaluate climb relative
                            # to the vehicle's initial ground position.
                            lambda telemetry: not telemetry.landed
                            and telemetry.position.z <= ground_z - 0.5,
                            vehicle_name,
                            "takeoff",
                            timeout=12,
                        )
                        break
                    except SimulatorError:
                        if attempt == 3:
                            raise
                        await self.adapter.request("cancel", timeout=3, vehicle_name=vehicle_name)
                        await self.adapter.request("arm", timeout=3, armed=False, vehicle_name=vehicle_name)
                        await asyncio.sleep(0.5)
                        await self.adapter.request("arm", timeout=3, armed=True, vehicle_name=vehicle_name)
                test_altitude_m = min(
                    profile.safety.max_altitude_m,
                    max(
                        profile.safety.min_altitude_m,
                        profile.zones[0].search_altitude_m,
                    ),
                )
                phase = "climb to smoke altitude"
                await self.adapter.request(
                    "move_to",
                    timeout=25,
                    vehicle_name=vehicle_name,
                    x=initial.position.x,
                    y=initial.position.y,
                    z=ground_z - test_altitude_m,
                    speed=min(2.0, profile.safety.max_speed_mps),
                )
                airborne = await self._wait_vehicle(
                    lambda telemetry: not telemetry.landed
                    and telemetry.position.z <= ground_z - test_altitude_m + 1.2,
                    vehicle_name,
                    "smoke altitude",
                    timeout=30,
                )
                phase = "hover"
                await self.adapter.request("hover", timeout=5, vehicle_name=vehicle_name)
                phase = "image capture"
                frame = await self.adapter.capture(vehicle_name)
                preview_payload = frame_preview_payload(
                    frame, source="smoke_airborne"
                )
                await self.events.publish(
                    "frame.preview",
                    preview_payload,
                )
                phase = "LiDAR scan"
                lidar = await self.adapter.request(
                    "lidar_scan",
                    timeout=5,
                    vehicle_name=vehicle_name,
                    sensor_name="LidarSensor1",
                    max_points=1000,
                )
                if lidar.get("data_frame") != "VehicleInertialFrame":
                    raise SimulatorError("LiDAR is not using the required world NED frame")
                if int(lidar.get("sampled_point_count", 0)) <= 0:
                    raise SimulatorError("LiDAR returned no obstacle points")
                smoke_lidar_points = decode_point_cloud(lidar.get("point_cloud", []))
                self.latest_lidar = point_cloud_preview_payload(
                    smoke_lidar_points, airborne.position
                )
                await self.events.publish("lidar.points", self.latest_lidar)
                if profile.smoke_cleanup == "reset":
                    # CityPark's stock spawn can be below its water surface;
                    # SimpleFlight then cannot confirm a normal landing. A
                    # simulator reset is the deterministic diagnostic cleanup.
                    phase = "reset cleanup"
                    await self.adapter.request(
                        "reset", timeout=8, vehicle_name=vehicle_name
                    )
                    await self.adapter.request(
                        "arm",
                        timeout=5,
                        armed=False,
                        vehicle_name=vehicle_name,
                    )
                    await self.adapter.request(
                        "api_control",
                        timeout=5,
                        enabled=False,
                        vehicle_name=vehicle_name,
                    )
                    final = await self._wait_ground_stable(vehicle_name)
                    landed = final
                else:
                    phase = "landing"
                    await self.adapter.request("land", timeout=5, vehicle_name=vehicle_name)
                    landed = await self._wait_vehicle(
                        lambda telemetry: telemetry.landed
                        or (
                            abs(telemetry.position.z - ground_z) <= 0.25
                            and abs(telemetry.velocity.z) <= 0.15
                        ),
                        vehicle_name,
                        "landing",
                        timeout=65,
                    )
                    await self.adapter.request("arm", timeout=5, armed=False, vehicle_name=vehicle_name)
                    await asyncio.sleep(0.5)
                    final = await self.adapter.telemetry(vehicle_name)
                await self.events.publish("telemetry", final.model_dump(mode="json"))
                result = {
                    "ok": True,
                    "scene_id": profile.id,
                    "airborne_position": airborne.position.model_dump(mode="json"),
                    "airborne_altitude_m": ground_z - airborne.position.z,
                    "frame": {
                        "width": frame.width,
                        "height": frame.height,
                        "frame_id": frame.frame_id,
                        "depth_source": preview_payload.get("depth_source"),
                        "depth_metric_valid": preview_payload.get(
                            "depth_metric_valid"
                        ),
                        "depth_min_m": preview_payload.get("depth_min_m"),
                        "depth_max_m": preview_payload.get("depth_max_m"),
                        "depth_warning": preview_payload.get("depth_warning"),
                    },
                    "lidar": {
                        "data_frame": lidar["data_frame"],
                        "point_count": int(lidar.get("point_count", 0)),
                        "sampled_point_count": int(lidar.get("sampled_point_count", 0)),
                    },
                    "ground_contact": landed.landed
                    or abs(landed.position.z - ground_z) <= 0.25,
                    "cleanup": profile.smoke_cleanup,
                    "telemetry": final.model_dump(mode="json"),
                }
                await self.events.publish("simulator.smoke", {"phase": "completed", **result})
                return result
            except Exception as error:
                bridge_tail = list(getattr(self.adapter, "stderr_tail", [])[-8:])
                try:
                    await self.adapter.request("cancel", timeout=2, vehicle_name=vehicle_name)
                    await self.adapter.request("land", timeout=2, vehicle_name=vehicle_name)
                except Exception:
                    pass
                await self.events.publish(
                    "simulator.smoke",
                    {
                        "phase": "failed",
                        "failed_phase": phase,
                        "scene_id": profile.id,
                        "error": f"{type(error).__name__}: {error}",
                        "bridge_stderr": bridge_tail,
                    },
                )
                raise SimulatorError(
                    f"vehicle smoke test failed during {phase}: {type(error).__name__}: {error}; "
                    f"bridge stderr tail: {bridge_tail}"
                ) from error

    async def _wait_vehicle(self, predicate, vehicle_name: str, phase: str, timeout: float):
        assert self.adapter
        deadline = asyncio.get_running_loop().time() + timeout
        last = None
        while asyncio.get_running_loop().time() < deadline:
            last = await self.adapter.telemetry(vehicle_name)
            await self.events.publish("telemetry", last.model_dump(mode="json"))
            if predicate(last):
                return last
            await asyncio.sleep(0.5)
        raise SimulatorError(f"timed out waiting for {phase}; last telemetry: {last}")

    async def _wait_ground_stable(self, vehicle_name: str, timeout: float = 6):
        """Let a newly spawned AirSim pawn settle before recording its home z."""
        assert self.adapter
        deadline = asyncio.get_running_loop().time() + timeout
        samples = []
        last = await self.adapter.telemetry(vehicle_name)
        while asyncio.get_running_loop().time() < deadline:
            last = await self.adapter.telemetry(vehicle_name)
            if last.landed and abs(last.velocity.z) <= 0.05:
                samples.append(last.position.z)
                samples = samples[-3:]
                if len(samples) == 3 and max(samples) - min(samples) <= 0.03:
                    return last
            else:
                samples.clear()
            await asyncio.sleep(0.25)
        return last

    async def _stop_unlocked(self, hard: bool) -> None:
        profile = self.active_profile
        preview_task, self._preview_task = self._preview_task, None
        if preview_task:
            preview_task.cancel()
            await asyncio.gather(preview_task, return_exceptions=True)
        adapter, self.adapter = self.adapter, None
        if adapter:
            if not hard and profile:
                try:
                    await adapter.request("cancel", timeout=2, vehicle_name=profile.vehicle_name)
                    await adapter.request("land", timeout=2, vehicle_name=profile.vehicle_name)
                    deadline = asyncio.get_running_loop().time() + 12
                    while asyncio.get_running_loop().time() < deadline:
                        telemetry = await adapter.telemetry(profile.vehicle_name)
                        if telemetry.landed:
                            await adapter.request(
                                "arm", timeout=2, armed=False, vehicle_name=profile.vehicle_name
                            )
                            break
                        await asyncio.sleep(0.5)
                except Exception:
                    pass
            await adapter.close()
        process, self.process = self.process, None
        if process and process.returncode is None:
            process.kill() if hard else process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=8)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        self.active_profile = None
        self.state = "STOPPED"
        await self.events.publish(
            "simulator.state", {"state": self.state, "scene_id": profile.id if profile else None}
        )

    async def _stream_preview(self, profile: SceneProfile) -> None:
        """Publish RGB/depth preview and a bounded LiDAR topology map."""
        await self.events.publish(
            "system.log",
            {"level": "info", "message": f"{profile.name} 实时预览已启动", "source": "preview"},
        )
        consecutive_failures = 0
        mapping_failures = 0
        try:
            while (
                self.state == "READY"
                and self.active_profile is profile
                and self.adapter is not None
            ):
                try:
                    telemetry = await self.adapter.telemetry(profile.vehicle_name)
                    self.latest_telemetry = telemetry
                    self.mapper.integrate_pose(telemetry.position)
                    await self.events.publish("telemetry", telemetry.model_dump(mode="json"))
                    frame = await self.adapter.capture(profile.vehicle_name)
                    self.latest_frame = frame
                    await self.events.publish(
                        "frame.preview",
                        frame_preview_payload(frame, source="live_preview"),
                    )
                    if not self.mission_active and telemetry.armed:
                        try:
                            scan = await self.adapter.request(
                                "lidar_scan",
                                timeout=4,
                                vehicle_name=profile.vehicle_name,
                                max_points=1800,
                            )
                            if scan.get("data_frame") == "VehicleInertialFrame":
                                lidar_points = decode_point_cloud(
                                    scan.get("point_cloud", [])
                                )
                                self.mapper.integrate_lidar(
                                    lidar_points, telemetry.position
                                )
                                self.latest_lidar = point_cloud_preview_payload(
                                    lidar_points, telemetry.position
                                )
                                await self.events.publish(
                                    "lidar.points", self.latest_lidar
                                )
                                mapping_failures = 0
                            else:
                                raise ValueError("LiDAR map frame is not VehicleInertialFrame")
                        except asyncio.CancelledError:
                            raise
                        except Exception as mapping_error:
                            mapping_failures += 1
                            if mapping_failures in {1, 3} or mapping_failures % 10 == 0:
                                await self.events.publish(
                                    "mapping.sensor_warning",
                                    {
                                        "level": "warning",
                                        "message": (
                                            "在线建图暂未收到有效 LiDAR："
                                            f"{type(mapping_error).__name__}: {mapping_error}"
                                        ),
                                        "consecutive_failures": mapping_failures,
                                    },
                                )
                    await self.events.publish("map.update", self.mapper.snapshot())
                    consecutive_failures = 0
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    consecutive_failures += 1
                    await self.events.publish(
                        "system.log",
                        {
                            "level": "error" if consecutive_failures >= 3 else "warning",
                            "message": f"实时预览取帧失败：{type(error).__name__}: {error}",
                            "source": "preview",
                            "consecutive_failures": consecutive_failures,
                        },
                    )
                await asyncio.sleep(0.75 if profile.mode == "mock" else 1.0)
        finally:
            await self.events.publish(
                "system.log",
                {"level": "info", "message": f"{profile.name} 实时预览已停止", "source": "preview"},
            )

    async def close(self) -> None:
        await self.stop(hard=False)
