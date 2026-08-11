from __future__ import annotations

import base64
import math
import struct
import zlib

import numpy as np

from .models import BoundingBox, CameraFrame, Quaternion, Vec3


class DepthLocalizationError(ValueError):
    pass


def _rotate(vector: np.ndarray, quaternion: Quaternion) -> np.ndarray:
    q = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z], dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise DepthLocalizationError("camera quaternion has zero length")
    w, x, y, z = q / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation @ vector


def decode_depth(frame: CameraFrame) -> np.ndarray:
    if not frame.depth_f32_zlib_b64:
        raise DepthLocalizationError("frame has no depth image")
    try:
        packed = base64.b64decode(frame.depth_f32_zlib_b64, validate=True)
        raw = zlib.decompress(packed)
    except Exception as error:
        raise DepthLocalizationError("depth payload is invalid") from error
    values = np.frombuffer(raw, dtype="<f4")
    expected = frame.width * frame.height
    if values.size != expected:
        raise DepthLocalizationError(f"depth payload has {values.size} values, expected {expected}")
    return values.reshape((frame.height, frame.width))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def depth_preview_payload(frame: CameraFrame, max_depth_m: float = 50.0) -> dict:
    """Render AirSim float depth as a bounded false-color PNG for the Web UI."""
    depth = decode_depth(frame)
    if frame.depth_source == "depth-vis-fallback":
        max_depth_m = 100.0
    stride = max(1, int(math.ceil(frame.width / 480)))
    sampled = depth[::stride, ::stride]
    valid = np.isfinite(sampled) & (sampled > 0.1)
    valid_values = sampled[valid]
    degenerate = bool(
        valid_values.size
        and float(
            np.percentile(valid_values, 95) - np.percentile(valid_values, 5)
        )
        < 0.05
    )
    clipped = np.clip(sampled, 0.5, max_depth_m)
    normalized = (clipped - 0.5) / max(0.1, max_depth_m - 0.5)
    # Reverse "jet": close geometry is red/yellow, distant geometry is blue.
    color_value = 1.0 - normalized
    red = np.clip(1.5 - np.abs(4 * color_value - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * color_value - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * color_value - 1), 0, 1)
    rgb = (np.stack([red, green, blue], axis=2) * 255).astype(np.uint8)
    rgb[~valid] = 0
    if degenerate:
        # A constant float depth frame is a failed UE post-process capture,
        # not a wall filling the image. Never present it as an all-red map.
        rgb[:] = 0
    height, width = rgb.shape[:2]
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, 4))
        + _png_chunk(b"IEND", b"")
    )
    return {
        "depth_data_url": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}",
        "depth_width": width,
        "depth_height": height,
        "depth_min_m": (
            float(valid_values.min()) if valid_values.size and not degenerate else None
        ),
        "depth_max_m": float(min(valid_values.max(), max_depth_m))
        if valid_values.size and not degenerate
        else None,
        "depth_scale_max_m": max_depth_m,
        "depth_source": frame.depth_source,
        "depth_metric_valid": not degenerate and bool(valid_values.size),
        "depth_warning": (
            "CityPark DepthPlanar was clamped; displaying the AirSim DepthVis 0-100 m fallback"
            if frame.depth_source == "depth-vis-fallback"
            else (
                "AirSim returned a constant depth frame; restart the scene with the updated settings"
                if degenerate
                else None
            )
        ),
    }


def quaternion_yaw_degrees(quaternion: Quaternion) -> float:
    norm = math.sqrt(
        quaternion.w**2 + quaternion.x**2 + quaternion.y**2 + quaternion.z**2
    )
    if norm < 1e-9:
        raise ValueError("camera quaternion has zero length")
    w = quaternion.w / norm
    x = quaternion.x / norm
    y = quaternion.y / norm
    z = quaternion.z / norm
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def frame_preview_payload(frame: CameraFrame, *, source: str | None = None) -> dict:
    payload = {
        "frame_id": frame.frame_id,
        "data_url": f"data:image/png;base64,{frame.scene_png_b64}",
        "width": frame.width,
        "height": frame.height,
        "depth_available": bool(frame.depth_f32_zlib_b64),
        "depth_source": frame.depth_source,
        "camera_position": frame.camera_position.model_dump(),
        "camera_orientation": frame.camera_orientation.model_dump(),
        "camera_yaw_degrees": quaternion_yaw_degrees(frame.camera_orientation),
    }
    if source:
        payload["source"] = source
    try:
        payload.update(depth_preview_payload(frame))
    except DepthLocalizationError:
        payload["depth_available"] = False
    return payload


def localize_bbox(frame: CameraFrame, bbox: BoundingBox) -> Vec3:
    depth = decode_depth(frame)
    x0 = max(0, min(frame.width - 1, int(bbox.x_min * frame.width)))
    x1 = max(x0 + 1, min(frame.width, int(math.ceil(bbox.x_max * frame.width))))
    y0 = max(0, min(frame.height - 1, int(bbox.y_min * frame.height)))
    y1 = max(y0 + 1, min(frame.height, int(math.ceil(bbox.y_max * frame.height))))
    crop = depth[y0:y1, x0:x1]
    valid = crop[np.isfinite(crop) & (crop > 0.1) & (crop < 1000)]
    if valid.size < 4:
        raise DepthLocalizationError("candidate bounding box has insufficient valid depth")
    forward = float(np.median(valid))
    u = ((bbox.x_min + bbox.x_max) / 2) * frame.width
    v = ((bbox.y_min + bbox.y_max) / 2) * frame.height
    horizontal_fov = math.radians(frame.fov_degrees)
    fx = frame.width / (2 * math.tan(horizontal_fov / 2))
    fy = fx
    camera_vector = np.array(
        [forward, (u - frame.width / 2) * forward / fx, (v - frame.height / 2) * forward / fy]
    )
    world_vector = _rotate(camera_vector, frame.camera_orientation)
    return Vec3(
        x=frame.camera_position.x + float(world_vector[0]),
        y=frame.camera_position.y + float(world_vector[1]),
        z=frame.camera_position.z + float(world_vector[2]),
    )


def distance(left: Vec3, right: Vec3) -> float:
    return math.sqrt((left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2)
