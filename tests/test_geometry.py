import base64
import zlib

import numpy as np
import pytest

from harness.geometry import (
    DepthLocalizationError,
    depth_preview_payload,
    frame_preview_payload,
    localize_bbox,
    quaternion_yaw_degrees,
)
from harness.models import BoundingBox, CameraFrame, Quaternion, Vec3


def frame(depth_value: float = 10) -> CameraFrame:
    depth = np.full((4, 4), depth_value, dtype="<f4")
    depth[0, 0] = depth_value + 1
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


def test_depth_preview_is_a_bounded_png_data_url():
    payload = depth_preview_payload(frame(12.5))
    assert payload["depth_data_url"].startswith("data:image/png;base64,")
    assert payload["depth_width"] == 4
    assert payload["depth_height"] == 4
    assert payload["depth_min_m"] == pytest.approx(12.5)
    assert payload["depth_max_m"] == pytest.approx(13.5)
    assert payload["depth_scale_max_m"] == 50.0
    assert payload["depth_metric_valid"] is True


def test_constant_citypark_depth_is_flagged_instead_of_rendered_all_red():
    value = frame(1).model_copy()
    depth = np.ones((4, 4), dtype="<f4")
    value.depth_f32_zlib_b64 = base64.b64encode(
        zlib.compress(depth.tobytes())
    ).decode()
    payload = depth_preview_payload(value)
    assert payload["depth_metric_valid"] is False
    assert payload["depth_min_m"] is None
    assert "constant depth frame" in payload["depth_warning"]


def test_depth_vis_fallback_exposes_its_bounded_metric_scale():
    value = frame(20).model_copy(update={"depth_source": "depth-vis-fallback"})
    payload = depth_preview_payload(value)
    assert payload["depth_source"] == "depth-vis-fallback"
    assert payload["depth_scale_max_m"] == 100.0
    assert "DepthVis" in payload["depth_warning"]


def test_preview_exposes_camera_yaw_for_the_map_arrow():
    value = frame().model_copy(
        update={"camera_orientation": Quaternion(w=2**-0.5, z=2**-0.5)}
    )
    assert quaternion_yaw_degrees(value.camera_orientation) == pytest.approx(90)
    assert frame_preview_payload(value)["camera_yaw_degrees"] == pytest.approx(90)
