import pytest
from pydantic import ValidationError

from harness.models import BoundingBox, DetectionAssessment, Polygon, SearchMissionRequest


def test_matching_detection_requires_bbox():
    with pytest.raises(ValidationError):
        DetectionAssessment(frame_id="f", is_match=True, confidence=0.9)


def test_bbox_must_be_ordered():
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.8, y_min=0.1, x_max=0.2, y_max=0.9)


def test_target_text_is_normalized():
    request = SearchMissionRequest(scene_id="s", zone_id="z", target_text="  red   cube ")
    assert request.target_text == "red cube"


def test_polygon_requires_area():
    with pytest.raises(ValidationError):
        Polygon(points=[(0, 0), (1, 1), (2, 2)])

