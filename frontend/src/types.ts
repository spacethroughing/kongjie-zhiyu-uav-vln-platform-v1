export type Vec3 = { x: number; y: number; z: number };

export type SafetyBounds = {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
};

export type Zone = {
  id: string;
  name: string;
  polygon: { points: [number, number][] };
  search_altitude_m: number;
  lane_spacing_m: number;
};

export type Scene = {
  id: string;
  name: string;
  mode: "mock" | "editor" | "packaged";
  manual_safety_bounds?: SafetyBounds;
  zones: Zone[];
  safety: {
    max_speed_mps: number;
    max_altitude_m: number;
    min_standoff_m: number;
    max_mission_seconds: number;
  };
};

export type MissionPlan = {
  id: string;
  version: number;
  request: {
    scene_id: string;
    zone_id: string;
    target_text: string;
    safety_bounds?: SafetyBounds;
  };
  route: { index: number; position: Vec3; observe: boolean }[];
  observation_yaws_deg: number[];
  safety_summary: string[];
};

export type RunState =
  | "READY"
  | "TAKEOFF"
  | "SEARCHING"
  | "VERIFYING"
  | "APPROACHING"
  | "EVIDENCE"
  | "RTH"
  | "LANDING"
  | "SUCCEEDED"
  | "PAUSED"
  | "SAFE_HOLD"
  | "ABORTING"
  | "ABORTED"
  | "FAILED"
  | "NOT_FOUND";

export type Run = {
  id: string;
  plan_id: string;
  state: RunState;
  started_at: string;
  ended_at?: string;
  error?: string;
  target_position?: Vec3;
};

export type Telemetry = {
  timestamp: string;
  position: Vec3;
  velocity: Vec3;
  armed: boolean;
  landed: boolean;
  collision: boolean;
};

export type HarnessEvent = {
  topic: string;
  run_id?: string;
  sequence: number;
  timestamp?: string;
  payload: Record<string, unknown>;
};

export type ProviderModelOption = {
  id: "glm-4.6v-flashx" | "glm-4.6v-flash" | "glm-4.6v" | "glm-5v-turbo";
  name: string;
  description: string;
  billing: "free" | "paid";
};

export type ProviderConfig = {
  provider: string;
  base_url: string;
  model: string;
  api_key_configured: boolean;
  runtime_only: boolean;
  models: ProviderModelOption[];
};
