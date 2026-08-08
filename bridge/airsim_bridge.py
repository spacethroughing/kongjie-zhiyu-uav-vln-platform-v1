"""Python 3.7-compatible AirSim JSONL bridge.

Stdout is protocol-only. Diagnostics go to stderr so the modern control plane can
use a long-lived subprocess without importing AirSim's legacy RPC dependencies.
"""
from __future__ import print_function

import base64
import json
import math
import multiprocessing
import sys
import time
import traceback
import uuid
import zlib

import airsim
import numpy as np


def execute_command(task, issued):
    """Run one AirSim future on its own msgpack client/event loop."""
    client = airsim.MultirotorClient()
    try:
        # Establish the socket synchronously so call_async writes the request
        # before the worker is allowed to process a newer command. Without
        # this handshake, a slow-to-connect old yaw thread can be issued after
        # a newer move command and silently supersede it.
        client.ping()
        operation = task["operation"]
        vehicle = task["vehicle_name"]
        if operation == "move_to":
            print(
                "command issue move_to target=%.2f,%.2f,%.2f"
                % (task["x"], task["y"], task["z"]),
                file=sys.stderr,
            )
        else:
            print("command issue %s" % operation, file=sys.stderr)
        if operation == "takeoff":
            future = client.takeoffAsync(task["timeout"], vehicle)
        elif operation == "land":
            future = client.landAsync(task["timeout"], vehicle)
        elif operation == "hover":
            future = client.hoverAsync(vehicle)
        elif operation == "rotate_yaw":
            future = client.rotateToYawAsync(
                task["yaw_degrees"], task["margin"], task["timeout"], vehicle
            )
        elif operation == "move_to":
            future = client.moveToPositionAsync(
                task["x"],
                task["y"],
                task["z"],
                task["speed"],
                task["timeout"],
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, task["yaw_degrees"]),
                -1,
                1,
                vehicle,
            )
        else:
            raise ValueError("unknown command operation: %s" % operation)
        issued.set()
        # get() waits like join() but also raises an RPC-side error instead of
        # making a rejected command look successfully completed.
        future.get()
        state = client.getMultirotorState(vehicle)
        position = state.kinematics_estimated.position
        print(
            "command complete %s position=%.2f,%.2f,%.2f"
            % (operation, position.x_val, position.y_val, position.z_val),
            file=sys.stderr,
        )
    except Exception:
        issued.set()
        traceback.print_exc(file=sys.stderr)


class Bridge(object):
    def __init__(self):
        self.client = None
        self.vehicle_name = "Drone1"
        self.camera_name = "front_center"
        self.armed = {}
        self.command_process = None

    def close(self):
        if self.command_process and self.command_process.is_alive():
            self.command_process.terminate()
            self.command_process.join(2)

    def _start_command(self, task):
        """Replace the previous flight future and wait only for RPC acceptance."""
        if self.command_process and self.command_process.is_alive():
            # Mission-side arrival tolerance can be reached just before the
            # AirSim future resolves. Prefer natural completion; cancelling
            # every short segment can race with and cancel the next RPC too.
            self.command_process.join(1)
            if self.command_process.is_alive():
                print("command supersede %s" % task["operation"], file=sys.stderr)
                self._ensure_client().cancelLastTask(task["vehicle_name"])
                time.sleep(0.2)
                self.command_process.terminate()
                self.command_process.join(2)
        issued = multiprocessing.Event()
        process = multiprocessing.Process(target=execute_command, args=(task, issued))
        process.daemon = True
        process.start()
        if not issued.wait(8):
            process.terminate()
            process.join(2)
            raise RuntimeError("AirSim command process did not issue the RPC")
        print("command accepted %s" % task["operation"], file=sys.stderr)
        self.command_process = process

    def _ensure_client(self):
        if self.client is None:
            self.client = airsim.MultirotorClient()
        return self.client

    @staticmethod
    def _vec(value):
        return {"x": value.x_val, "y": value.y_val, "z": value.z_val}

    @staticmethod
    def _quat(value):
        return {"w": value.w_val, "x": value.x_val, "y": value.y_val, "z": value.z_val}

    def dispatch(self, operation, arguments):
        client = self._ensure_client()
        vehicle = arguments.get("vehicle_name", self.vehicle_name)
        if operation in ("connect", "ping"):
            try:
                connected = bool(client.ping())
                vehicles = client.listVehicles() if connected else []
                return {"connected": connected, "vehicles": vehicles}
            except Exception:
                # msgpack-rpc clients can become unusable after a refused
                # connection; a fresh client is required for the next retry.
                self.client = None
                raise
        if operation == "api_control":
            enabled = bool(arguments.get("enabled", True))
            client.enableApiControl(enabled, vehicle)
            return {"enabled": bool(client.isApiControlEnabled(vehicle))}
        if operation == "arm":
            armed = bool(arguments.get("armed", True))
            client.armDisarm(armed, vehicle)
            self.armed[vehicle] = armed
            return {"armed": armed}
        if operation == "takeoff":
            timeout = float(arguments.get("timeout", 20))
            self._start_command(
                {"operation": operation, "vehicle_name": vehicle, "timeout": timeout}
            )
            return {"accepted": True}
        if operation == "land":
            timeout = float(arguments.get("timeout", 60))
            self._start_command(
                {"operation": operation, "vehicle_name": vehicle, "timeout": timeout}
            )
            return {"accepted": True}
        if operation == "hover":
            self._start_command({"operation": operation, "vehicle_name": vehicle})
            return {"accepted": True}
        if operation == "cancel":
            client.cancelLastTask(vehicle)
            if self.command_process and self.command_process.is_alive():
                time.sleep(0.2)
                self.command_process.terminate()
                self.command_process.join(2)
            return {"accepted": True}
        if operation == "rotate_yaw":
            yaw = float(arguments["yaw_degrees"])
            margin = float(arguments.get("margin", 5))
            timeout = float(arguments.get("timeout", 10))
            self._start_command(
                {
                    "operation": operation,
                    "vehicle_name": vehicle,
                    "yaw_degrees": yaw,
                    "margin": margin,
                    "timeout": timeout,
                }
            )
            return {"accepted": True}
        if operation == "move_to":
            x = float(arguments["x"])
            y = float(arguments["y"])
            z = float(arguments["z"])
            speed = float(arguments["speed"])
            timeout = float(arguments.get("timeout", 30))
            yaw = float(arguments.get("yaw_degrees", 0))
            self._start_command(
                {
                    "operation": operation,
                    "vehicle_name": vehicle,
                    "x": x,
                    "y": y,
                    "z": z,
                    "speed": speed,
                    "timeout": timeout,
                    "yaw_degrees": yaw,
                }
            )
            return {"accepted": True}
        if operation == "state":
            state = client.getMultirotorState(vehicle)
            collision = client.simGetCollisionInfo(vehicle)
            landed = state.landed_state == airsim.LandedState.Landed
            return {
                "position": self._vec(state.kinematics_estimated.position),
                "velocity": self._vec(state.kinematics_estimated.linear_velocity),
                "armed": bool(self.armed.get(vehicle, False)),
                "landed": bool(landed),
                "collision": bool(collision.has_collided),
                "timestamp_ns": int(state.timestamp),
            }
        if operation == "capture":
            camera_name = arguments.get("camera_name", self.camera_name)
            responses = client.simGetImages(
                [
                    airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, True),
                    airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False),
                ],
                vehicle,
            )
            if len(responses) != 2:
                raise RuntimeError("AirSim did not return scene and depth images")
            scene, depth = responses
            scene_bytes = bytes(scene.image_data_uint8)
            depth_array = np.asarray(depth.image_data_float, dtype="<f4")
            if depth_array.size != int(depth.width) * int(depth.height):
                raise RuntimeError("AirSim depth image has an unexpected shape")
            info = client.simGetCameraInfo(camera_name, vehicle)
            return {
                "frame_id": str(uuid.uuid4()),
                "width": int(depth.width),
                "height": int(depth.height),
                "scene_png_b64": base64.b64encode(scene_bytes).decode("ascii"),
                "depth_f32_zlib_b64": base64.b64encode(zlib.compress(depth_array.tobytes(), 3)).decode("ascii"),
                "camera_position": self._vec(info.pose.position),
                "camera_orientation": self._quat(info.pose.orientation),
                "fov_degrees": float(info.fov),
            }
        if operation == "lidar_min":
            sensor = arguments.get("sensor_name", "LidarSensor1")
            data = client.getLidarData(sensor, vehicle)
            points = data.point_cloud
            minimum = None
            for index in range(0, len(points), 3):
                distance = math.sqrt(points[index] ** 2 + points[index + 1] ** 2 + points[index + 2] ** 2)
                if distance > 0 and (minimum is None or distance < minimum):
                    minimum = distance
            return {"minimum_m": minimum, "point_count": len(points) // 3}
        if operation == "reset":
            client.reset()
            return {"accepted": True}
        raise ValueError("unknown operation: %s" % operation)


def write_message(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    bridge = Bridge()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result = bridge.dispatch(request["op"], request.get("args", {}))
                write_message({"id": request["id"], "ok": True, "result": result})
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                request_id = request.get("id") if isinstance(request, dict) else None
                write_message({"id": request_id, "ok": False, "error": str(error)})
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
