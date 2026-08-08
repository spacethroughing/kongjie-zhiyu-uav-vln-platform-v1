from __future__ import annotations

import base64
import math
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

