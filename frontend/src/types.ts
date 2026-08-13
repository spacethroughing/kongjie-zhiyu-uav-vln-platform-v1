export type Vec3 = { x: number; y: number; z: number };
export type Quaternion = { w: number; x: number; y: number; z: number };

export type SafetyBounds = {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
};

export type EndPolicy = "review_then_rth" | "auto_rth" | "land_at_target";
export type MissionMode = "target_search" | "semantic_mapping";

export type MissionPlanParameters = {
  search_altitude_m: number;
  lane_spacing_m: number;
  max_speed_mps: number;
  approach_speed_mps: number;
  min_standoff_m: number;
  min_clearance_m: number;
  max_mission_seconds: number;
  mapping_coverage_target: number;
};

export type MissionTask = {
  id: string;
  kind: MissionMode;
  label: string;
  target_text?: string;
  coverage_target?: number;
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
    min_altitude_m: number;
    max_speed_mps: number;
    approach_speed_mps: number;
    max_altitude_m: number;
    min_standoff_m: number;
    min_clearance_m: number;
    max_mission_seconds: number;
  };
};

export type MissionPlan = {
  id: string;
  version: number;
  approved_at?: string | null;
  request: {
    scene_id: string;
    zone_id: string;
    target_text: string;
    mission_mode: MissionMode;
    targets: string[];
    mapping_coverage_target: number;
    end_policy: EndPolicy;
    safety_bounds?: SafetyBounds;
  };
  parameters: MissionPlanParameters;
  tasks: MissionTask[];
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
  home_position?: Vec3;
  target_position?: Vec3;
  current_task_index: number;
  mapping_coverage_ratio?: number;
  task_progress: Array<{
    task_id: string;
    kind: MissionMode;
    label: string;
    state: "pending" | "running" | "succeeded" | "not_found" | "failed";
    target_position?: Vec3;
    coverage_ratio?: number;
    message: string;
  }>;
};

export type Telemetry = {
  timestamp: string;
  position: Vec3;
  velocity: Vec3;
  orientation?: Quaternion;
  armed: boolean;
  landed: boolean;
  collision: boolean;
};

export type LidarPointCloudFrame = {
  timestamp: string;
  data_frame: "VehicleInertialFrame";
  point_count: number;
  sampled_point_count: number;
  vehicle_position: Vec3;
  points: [number, number, number][];
};

export type TopologyNode = {
  id: string;
  kind: "place" | "object";
  position: Vec3;
  label?: string;
  confidence?: number;
  observations?: number;
};

export type TopologyEdge = {
  source: string;
  target: string;
  kind: "traversal" | "observed";
};

export type SemanticMap = {
  revision: number;
  obstacles: Array<Vec3 & { hits: number }>;
  explored: Array<{ x: number; y: number; scans: number }>;
  coverage_cell_size_m: number;
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  stats: {
    occupancy_cells: number;
    occupancy_tracks?: number;
    explored_cells: number;
    place_nodes: number;
    semantic_objects: number;
    semantic_tracks?: number;
  };
};

export type VlmChatMessage = {
  role: "user" | "assistant";
  content: string;
  action?: string;
  error?: string;
  heading_degrees?: number;
  distance_m?: number;
  altitude_delta_m?: number;
  target_altitude_m?: number;
  command_status?: "none" | "planned" | "executed" | "queued" | "rejected";
  context_target_text?: string;
  mission_mode?: MissionMode;
  mission_plan_id?: string;
  task_breakdown?: MissionTask[];
};

export type VlmMissionIntent = {
  kind: MissionMode;
  summary: string;
  targets: string[];
  coverage_target: number;
};

export type VlmChatResponse = {
  reply: string;
  requested_action?: "pause" | "resume" | "return-home" | "land" | "abort" | "explore" | "change-altitude";
  executed_action?: "pause" | "resume" | "return-home" | "land" | "abort" | "explore" | "change-altitude";
  command_error?: string;
  run_id?: string;
  frame_used: boolean;
  heading_degrees?: number;
  distance_m?: number;
  altitude_delta_m?: number;
  target_altitude_m?: number;
  context_target_text?: string;
  mission_intent?: VlmMissionIntent;
  mission_plan?: MissionPlan;
  task_breakdown: MissionTask[];
  command_status: "none" | "planned" | "executed" | "queued" | "rejected";
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
