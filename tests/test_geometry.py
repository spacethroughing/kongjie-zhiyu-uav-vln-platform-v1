import base64
import zlib

import numpy as np
import pytest

from harness.geometry import DepthLocalizationError, localize_bbox
from harness.models import BoundingBox, CameraFrame, Quaternion, Vec3


def frame(depth_value: float = 10) -> CameraFrame:
    depth = np.full((4, 4), depth_value, dtype="<f4")
    return CameraFrame(
        frame_id="f",
        width=4,
        height=4,
        scene_png_b64="",
        depth_f32_zlib_b64=base64.b64encode(zlib.compress(depth.tobytes())).decode(),
        camera_position=Vec3(x=1, y=2, z=-5),
        camera_orientation=Quaternion(),
        fov_degrees=90,
    )


def test_center_bbox_projects_forward_in_ned():
    result = localize_bbox(frame(), BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75))
    assert result.x == pytest.approx(11)
    assert result.y == pytest.approx(2)
    assert result.z == pytest.approx(-5)


def test_missing_depth_is_rejected():
    value = frame().model_copy(update={"depth_f32_zlib_b64": None})
    with pytest.raises(DepthLocalizationError):
        localize_bbox(value, BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8))

