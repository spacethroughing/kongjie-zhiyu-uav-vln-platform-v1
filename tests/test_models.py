import pytest
from pydantic import ValidationError

from harness.models import (
    BoundingBox,
    DetectionAssessment,
    MissionPlanParameters,
    Polygon,
    SafetyBounds,
    SearchMissionRequest,
    SemanticObservation,
    Telemetry,
    Vec3,
    VlmChatDecision,
    VlmMissionIntent,
)


def test_telemetry_stream_carries_vehicle_orientation_for_the_digital_twin():
    telemetry = Telemetry(
        position=Vec3(x=1, y=2, z=-3),
        orientation={"w": 0.7071, "x": 0, "y": 0, "z": 0.7071},
    )
    assert telemetry.orientation.z == pytest.approx(0.7071)


def test_matching_detection_requires_bbox():
    with pytest.raises(ValidationError):
        DetectionAssessment(frame_id="f", is_match=True, confidence=0.9)


def test_detection_accepts_bounded_semantic_inventory():
    box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.8)
    assessment = DetectionAssessment(
        frame_id="f",
        is_match=False,
        confidence=0.2,
        observed_objects=[
            SemanticObservation(label="  橙色   球体 ", confidence=0.88, bbox_norm=box)
        ],
    )
    assert assessment.observed_objects[0].label == "橙色 球体"


def test_bbox_must_be_ordered():
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.8, y_min=0.1, x_max=0.2, y_max=0.9)


def test_target_text_is_normalized():
    request = SearchMissionRequest(scene_id="s", zone_id="z", target_text="  red   cube ")
    assert request.target_text == "red cube"


def test_plan_parameter_editor_uses_999_absolute_limit_but_rejects_negative_physics():
    valid = MissionPlanParameters(
        search_altitude_m=999,
        lane_spacing_m=999,
        max_speed_mps=999,
        approach_speed_mps=999,
        min_standoff_m=999,
        min_clearance_m=999,
        max_mission_seconds=999,
        mapping_coverage_target=0.85,
    )
    assert valid.max_speed_mps == 999
    with pytest.raises(ValidationError):
        MissionPlanParameters(
            **{**valid.model_dump(), "min_clearance_m": -1}
        )


def test_compound_mission_targets_are_normalized_and_deduplicated():
    request = SearchMissionRequest(
        scene_id="s",
        zone_id="z",
        target_text="圆锥体和橙色球体",
        targets=[" 圆锥体 ", "橙色   球体", "圆锥体"],
    )
    assert request.targets == ["圆锥体", "橙色 球体"]
    intent = VlmMissionIntent(
        kind="target_search",
        summary="探索两个目标",
        targets=["圆锥体", "橙色球体"],
    )
    assert len(intent.targets) == 2


def test_semantic_mapping_intent_has_no_visual_search_targets():
    intent = VlmMissionIntent(
        kind="semantic_mapping",
        summary="建立区域语义拓扑图",
        targets=["应被移除"],
        coverage_target=0.9,
    )
    assert intent.targets == []
    assert intent.coverage_target == 0.9


def test_polygon_requires_area():
    with pytest.raises(ValidationError):
        Polygon(points=[(0, 0), (1, 1), (2, 2)])


def test_manual_safety_bounds_require_a_finite_minimum_rectangle():
    with pytest.raises(ValidationError):
        SafetyBounds(x_min=0, x_max=3, y_min=-10, y_max=10)
    with pytest.raises(ValidationError):
        SafetyBounds(x_min=float("nan"), x_max=10, y_min=-10, y_max=10)


def test_land_at_target_is_a_valid_mission_end_policy():
    request = SearchMissionRequest(
        scene_id="blocks",
        zone_id="blocks-main",
        target_text="orange sphere",
        end_policy="land_at_target",
    )
    assert request.end_policy == "land_at_target"


def test_vlm_explore_action_requires_a_bounded_heading_and_distance():
    decision = VlmChatDecision(
        reply="向东北探索。",
        action="explore",
        heading_degrees=405,
        distance_m=10,
    )
    assert decision.heading_degrees == 45
    with pytest.raises(ValidationError):
        VlmChatDecision(reply="缺少距离", action="explore", heading_degrees=90)
    with pytest.raises(ValidationError):
        VlmChatDecision(
            reply="距离过大", action="explore", heading_degrees=90, distance_m=21
        )


def test_vlm_altitude_action_requires_exactly_one_bounded_altitude_field():
    relative = VlmChatDecision(
        reply="升高三米",
        action="change-altitude",
        altitude_delta_m=3,
    )
    assert relative.altitude_delta_m == 3
    absolute = VlmChatDecision(
        reply="飞到十二米高度",
        action="change-altitude",
        target_altitude_m=12,
    )
    assert absolute.target_altitude_m == 12
    with pytest.raises(ValidationError):
        VlmChatDecision(reply="缺少高度", action="change-altitude")
    with pytest.raises(ValidationError):
        VlmChatDecision(
            reply="字段冲突",
            action="change-altitude",
            altitude_delta_m=2,
            target_altitude_m=10,
        )
    with pytest.raises(ValidationError):
        VlmChatDecision(reply="越界", action="change-altitude", altitude_delta_m=11)
