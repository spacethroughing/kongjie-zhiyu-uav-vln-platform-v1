from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunState(str, Enum):
    READY = "READY"
    TAKEOFF = "TAKEOFF"
    SEARCHING = "SEARCHING"
    VERIFYING = "VERIFYING"
    APPROACHING = "APPROACHING"
    EVIDENCE = "EVIDENCE"
    RTH = "RTH"
    LANDING = "LANDING"
    SUCCEEDED = "SUCCEEDED"
    PAUSED = "PAUSED"
    SAFE_HOLD = "SAFE_HOLD"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"
    FAILED = "FAILED"
    NOT_FOUND = "NOT_FOUND"


TERMINAL_STATES = {
    RunState.SUCCEEDED,
    RunState.ABORTED,
    RunState.FAILED,
    RunState.NOT_FOUND,
}


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Polygon(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=3)

    @model_validator(mode="after")
    def ensure_area(self) -> "Polygon":
        area = 0.0
        for index, point in enumerate(self.points):
            next_point = self.points[(index + 1) % len(self.points)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        if abs(area) < 1e-6:
            raise ValueError("polygon must have non-zero area")
        return self


class SafetyBounds(BaseModel):
    """Axis-aligned, home-relative NED bounds selected for one mission."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @model_validator(mode="after")
    def valid_rectangle(self) -> "SafetyBounds":
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("safety bounds must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("safety bounds minimums must be smaller than maximums")
        if self.x_max - self.x_min < 4 or self.y_max - self.y_min < 4:
            raise ValueError("safety bounds must span at least 4 m on each axis")
        return self

    def polygon(self) -> Polygon:
        return Polygon(
            points=[
                (self.x_min, self.y_min),
                (self.x_max, self.y_min),
                (self.x_max, self.y_max),
                (self.x_min, self.y_max),
            ]
        )


class SafetyEnvelope(BaseModel):
    min_altitude_m: float = 2.0
    max_altitude_m: float = 30.0
    max_speed_mps: float = 3.0
    approach_speed_mps: float = 1.0
    min_standoff_m: float = 3.0
    min_clearance_m: float = 1.5
    obstacle_avoidance_enabled: bool = True
    avoidance_segment_m: float = 4.0
    avoidance_max_replans: int = 12
    max_mission_seconds: int = 900
    telemetry_stale_seconds: float = 1.5
    no_fly_zones: list[Polygon] = Field(default_factory=list)

    @field_validator(
        "min_altitude_m",
        "max_altitude_m",
        "max_speed_mps",
        "approach_speed_mps",
        "min_standoff_m",
        "min_clearance_m",
        "avoidance_segment_m",
        "telemetry_stale_seconds",
    )
    @classmethod
    def positive_safety_value(cls, value: float) -> float:
        if value <= 0 or not math.isfinite(value):
            raise ValueError("safety values must be finite and positive")
        return value

    @field_validator("avoidance_max_replans", "max_mission_seconds")
    @classmethod
    def positive_safety_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("safety counts must be positive")
        return value


class SearchZone(BaseModel):
    id: str
    name: str
    polygon: Polygon
    coverage_polygon: Polygon | None = None
    search_altitude_m: float = 5.0
    lane_spacing_m: float = 5.0
    observation_yaws_deg: list[float] = Field(default_factory=lambda: [0, 90, 180, 270])
    initial_panorama_yaws_deg: list[float] = Field(default_factory=list)

    @field_validator("search_altitude_m", "lane_spacing_m")
    @classmethod
    def positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return value


class SceneProfile(BaseModel):
    id: str
    name: str
    mode: Literal["mock", "editor", "packaged"]
    executable: str | None = None
    project: str | None = None
    map: str | None = None
    settings: str | None = None
    vehicle_name: str = "Drone1"
    bridge_conda_env: str = "airsim"
    smoke_cleanup: Literal["land", "reset"] = "land"
    launch_args: list[str] = Field(default_factory=list)
    checksum_paths: list[str] = Field(default_factory=list)
    manual_safety_bounds: SafetyBounds | None = None
    zones: list[SearchZone]
    safety: SafetyEnvelope = Field(default_factory=SafetyEnvelope)

    @model_validator(mode="after")
    def require_launch_paths(self) -> "SceneProfile":
        if self.mode != "mock" and not self.executable:
            raise ValueError("non-mock scene requires executable")
        if self.mode == "editor" and not self.project:
            raise ValueError("editor scene requires project")
        if self.manual_safety_bounds:
            bounds = self.manual_safety_bounds
            if not (bounds.x_min <= 0 <= bounds.x_max and bounds.y_min <= 0 <= bounds.y_max):
                raise ValueError("scene manual safety limit must include NED home (0, 0)")
            for zone in self.zones:
                if any(
                    x < bounds.x_min
                    or x > bounds.x_max
                    or y < bounds.y_min
                    or y > bounds.y_max
                    for x, y in zone.polygon.points
                ):
                    raise ValueError(
                        f"zone {zone.id!r} extends beyond the scene manual safety limit"
                    )
        return self


class SearchMissionRequest(BaseModel):
    scene_id: str
    zone_id: str
    target_text: str = Field(min_length=2, max_length=500)
    mission_mode: Literal["target_search", "semantic_mapping"] = "target_search"
    targets: list[str] = Field(default_factory=list, max_length=6)
    mapping_coverage_target: float = Field(default=0.85, ge=0.5, le=0.98)
    end_policy: Literal["review_then_rth", "auto_rth", "land_at_target"] = (
        "review_then_rth"
    )
    safety_bounds: SafetyBounds | None = None

    @field_validator("target_text")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("target_text is required")
        return normalized

    @field_validator("targets")
    @classmethod
    def normalize_targets(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = " ".join(value.strip().split())
            if len(item) < 2:
                raise ValueError("each target must contain at least two characters")
            if len(item) > 120:
                raise ValueError("each target must contain at most 120 characters")
            if item not in normalized:
                normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def validate_mission_payload(self) -> "SearchMissionRequest":
        if self.mission_mode == "semantic_mapping":
            self.targets = []
        elif not self.targets:
            self.targets = [self.target_text]
        return self


class RoutePoint(BaseModel):
    index: int
    position: Vec3
    observe: bool = True


class MissionTask(BaseModel):
    id: str
    kind: Literal["target_search", "semantic_mapping"]
    label: str
    target_text: str | None = None
    coverage_target: float | None = Field(default=None, ge=0.5, le=0.98)


class MissionPlanParameters(BaseModel):
    """User-reviewable parameters frozen into one immutable plan version."""

    search_altitude_m: float = Field(gt=0, le=999)
    lane_spacing_m: float = Field(gt=0, le=999)
    max_speed_mps: float = Field(gt=0, le=999)
    approach_speed_mps: float = Field(gt=0, le=999)
    min_standoff_m: float = Field(gt=0, le=999)
    min_clearance_m: float = Field(gt=0, le=999)
    max_mission_seconds: int = Field(ge=30, le=999)
    mapping_coverage_target: float = Field(ge=0.5, le=0.98)

    @model_validator(mode="after")
    def validate_speed_order(self) -> "MissionPlanParameters":
        if self.approach_speed_mps > self.max_speed_mps:
            raise ValueError("approach speed cannot exceed maximum speed")
        return self


class MissionPlanRevisionRequest(BaseModel):
    base_version: int = Field(ge=1)
    parameters: MissionPlanParameters | None = None


class MissionPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    request: SearchMissionRequest
    parameters: MissionPlanParameters
    tasks: list[MissionTask] = Field(default_factory=list, max_length=6)
    route: list[RoutePoint]
    observation_yaws_deg: list[float]
    safety: SafetyEnvelope
    safety_summary: list[str]
    approved_at: datetime | None = None


class BoundingBox(BaseModel):
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> "BoundingBox":
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bounding box coordinates are not ordered")
        return self


class SemanticObservation(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0, le=1)
    bbox_norm: BoundingBox

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("semantic label is required")
        return normalized


class DetectionAssessment(BaseModel):
    frame_id: str
    is_match: bool
    confidence: float = Field(ge=0, le=1)
    bbox_norm: BoundingBox | None = None
    evidence: str = Field(default="", max_length=500)
    observed_objects: list[SemanticObservation] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def match_requires_box(self) -> "DetectionAssessment":
        if self.is_match and self.bbox_norm is None:
            raise ValueError("a matching assessment requires bbox_norm")
        return self


class ExplorationRouteDecision(BaseModel):
    waypoint_ids: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(default="", max_length=500)

    @field_validator("waypoint_ids")
    @classmethod
    def unique_waypoint_ids(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            normalized = item.strip()
            if normalized and normalized not in result:
                result.append(normalized)
        return result


class Telemetry(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    position: Vec3
    velocity: Vec3 = Field(default_factory=lambda: Vec3(x=0, y=0, z=0))
    orientation: Quaternion = Field(default_factory=Quaternion)
    armed: bool = False
    landed: bool = True
    collision: bool = False


class CameraFrame(BaseModel):
    frame_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    width: int
    height: int
    scene_png_b64: str
    depth_f32_zlib_b64: str | None = None
    depth_source: Literal["depth-planar", "depth-vis-fallback"] = "depth-planar"
    camera_position: Vec3
    camera_orientation: Quaternion
    fov_degrees: float


class MissionTaskProgress(BaseModel):
    task_id: str
    kind: Literal["target_search", "semantic_mapping"]
    label: str
    state: Literal["pending", "running", "succeeded", "not_found", "failed"] = "pending"
    target_position: Vec3 | None = None
    coverage_ratio: float | None = Field(default=None, ge=0, le=1)
    message: str = ""


class RunRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    plan_id: str
    state: RunState = RunState.READY
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    home_position: Vec3 | None = None
    target_position: Vec3 | None = None
    current_task_index: int = 0
    task_progress: list[MissionTaskProgress] = Field(default_factory=list)
    mapping_coverage_ratio: float | None = Field(default=None, ge=0, le=1)
    error: str | None = None
    artifact_dir: str


class EventEnvelope(BaseModel):
    topic: str
    run_id: str | None = None
    sequence: int
    timestamp: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any]


class SimulatorStartRequest(BaseModel):
    scene_id: str


class HealthResponse(BaseModel):
    ok: bool
    simulator_state: str
    active_scene_id: str | None = None
    provider: str


class ApiMessage(BaseModel):
    message: str


class ArtifactsResponse(BaseModel):
    run_id: str
    files: list[str]


class CandidateDecision(BaseModel):
    decision: Literal["accept", "continue"]


class JsonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


VlmControlAction = Literal[
    "pause",
    "resume",
    "return-home",
    "land",
    "abort",
    "explore",
    "change-altitude",
]


class VlmMissionIntent(JsonModel):
    kind: Literal["target_search", "semantic_mapping"]
    summary: str = Field(min_length=1, max_length=500)
    targets: list[str] = Field(default_factory=list, max_length=6)
    coverage_target: float = Field(default=0.85, ge=0.5, le=0.98)

    @field_validator("targets")
    @classmethod
    def normalize_intent_targets(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            item = " ".join(value.strip().split())
            if len(item) < 2 or len(item) > 120:
                raise ValueError("mission targets must contain 2 to 120 characters")
            if item not in result:
                result.append(item)
        return result

    @model_validator(mode="after")
    def validate_intent(self) -> "VlmMissionIntent":
        if self.kind == "target_search" and not self.targets:
            raise ValueError("target search mission requires at least one target")
        if self.kind == "semantic_mapping":
            self.targets = []
        return self


class VlmChatMessage(JsonModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("chat content is required")
        return normalized


class VlmChatRequest(JsonModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[VlmChatMessage] = Field(default_factory=list, max_length=12)
    run_id: str | None = None
    target_text: str | None = Field(default=None, max_length=500)
    scene_id: str | None = None
    zone_id: str | None = None
    end_policy: Literal["review_then_rth", "auto_rth", "land_at_target"] = "auto_rth"
    safety_bounds: SafetyBounds | None = None
    include_frame: bool = True
    execute_command: bool = True

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("chat message is required")
        return normalized

    @field_validator("target_text")
    @classmethod
    def normalize_target_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        return normalized or None


class VlmChatDecision(JsonModel):
    reply: str = Field(min_length=1, max_length=2000)
    action: VlmControlAction | None = None
    action_reason: str = Field(default="", max_length=500)
    heading_degrees: float | None = None
    distance_m: float | None = Field(default=None, gt=0, le=20)
    altitude_delta_m: float | None = Field(default=None, ge=-10, le=10)
    target_altitude_m: float | None = Field(default=None, ge=1, le=100)
    mission: VlmMissionIntent | None = None

    @model_validator(mode="after")
    def validate_control_fields(self) -> "VlmChatDecision":
        if self.mission is not None and self.action is not None:
            raise ValueError("mission planning and immediate flight control are mutually exclusive")
        if self.action == "explore":
            if self.heading_degrees is None or self.distance_m is None:
                raise ValueError("explore action requires heading_degrees and distance_m")
            if not math.isfinite(self.heading_degrees):
                raise ValueError("heading_degrees must be finite")
            self.heading_degrees %= 360
        elif self.heading_degrees is not None or self.distance_m is not None:
            raise ValueError("heading and distance are only valid for explore action")
        if self.action == "change-altitude":
            supplied = (self.altitude_delta_m is not None) + (
                self.target_altitude_m is not None
            )
            if supplied != 1:
                raise ValueError(
                    "change-altitude requires exactly one altitude target or delta"
                )
            if (
                self.altitude_delta_m is not None
                and abs(self.altitude_delta_m) < 0.1
            ):
                raise ValueError("altitude delta is too small")
        elif self.altitude_delta_m is not None or self.target_altitude_m is not None:
            raise ValueError(
                "altitude fields are only valid for change-altitude action"
            )
        return self


class VlmChatResponse(JsonModel):
    reply: str
    requested_action: VlmControlAction | None = None
    executed_action: VlmControlAction | None = None
    command_error: str | None = None
    run_id: str | None = None
    frame_used: bool = False
    heading_degrees: float | None = None
    distance_m: float | None = None
    altitude_delta_m: float | None = None
    target_altitude_m: float | None = None
    context_target_text: str | None = None
    mission_intent: VlmMissionIntent | None = None
    mission_plan: MissionPlan | None = None
    task_breakdown: list[MissionTask] = Field(default_factory=list)
    command_status: Literal["none", "planned", "executed", "queued", "rejected"] = "none"


ZhipuVisionModel = Literal[
    "glm-4.6v-flashx",
    "glm-4.6v-flash",
    "glm-4.6v",
    "glm-5v-turbo",
]


class ProviderConfigRequest(JsonModel):
    model: ZhipuVisionModel
    api_key: SecretStr | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def empty_key_keeps_current_key(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ProviderModelOption(JsonModel):
    id: ZhipuVisionModel
    name: str
    description: str
    billing: Literal["free", "paid"]


class ProviderConfigResponse(JsonModel):
    provider: str
    base_url: str
    model: str
    api_key_configured: bool
    runtime_only: bool = True
    models: list[ProviderModelOption]
