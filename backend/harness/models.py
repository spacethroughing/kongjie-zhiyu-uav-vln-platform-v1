from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    max_mission_seconds: int = 900
    telemetry_stale_seconds: float = 1.5
    no_fly_zones: list[Polygon] = Field(default_factory=list)


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
    end_policy: Literal["review_then_rth", "auto_rth"] = "review_then_rth"
    safety_bounds: SafetyBounds | None = None

    @field_validator("target_text")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("target_text is required")
        return normalized


class RoutePoint(BaseModel):
    index: int
    position: Vec3
    observe: bool = True


class MissionPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    request: SearchMissionRequest
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


class DetectionAssessment(BaseModel):
    frame_id: str
    is_match: bool
    confidence: float = Field(ge=0, le=1)
    bbox_norm: BoundingBox | None = None
    evidence: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def match_requires_box(self) -> "DetectionAssessment":
        if self.is_match and self.bbox_norm is None:
            raise ValueError("a matching assessment requires bbox_norm")
        return self


class Telemetry(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    position: Vec3
    velocity: Vec3 = Field(default_factory=lambda: Vec3(x=0, y=0, z=0))
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
    camera_position: Vec3
    camera_orientation: Quaternion
    fov_degrees: float


class RunRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    plan_id: str
    state: RunState = RunState.READY
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    target_position: Vec3 | None = None
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
