import type { EndPolicy, MissionPlan, MissionPlanParameters, ProviderConfig, Run, SafetyBounds, Scene, VlmChatMessage, VlmChatResponse } from "./types";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail ?? response.statusText);
  }
  return response.json();
}

export const api = {
  health: () => request<{ simulator_state: string; active_scene_id?: string; provider: string }>("/api/health"),
  providerConfig: () => request<ProviderConfig>("/api/provider/config"),
  configureProvider: (model: string, api_key?: string) =>
    request<ProviderConfig>("/api/provider/config", {
      method: "PUT",
      body: JSON.stringify({ model, ...(api_key ? { api_key } : {}) }),
    }),
  scenes: () => request<Scene[]>("/api/scenes"),
  runs: () => request<Run[]>("/api/runs"),
  start: (scene_id: string) =>
    request<{ state: string }>("/api/simulator/start", { method: "POST", body: JSON.stringify({ scene_id }) }),
  stop: () => request("/api/simulator/stop", { method: "POST" }),
  plan: (
    scene_id: string,
    zone_id: string,
    target_text: string,
    end_policy: EndPolicy,
    safety_bounds?: SafetyBounds,
  ) =>
    request<MissionPlan>("/api/missions/plan", {
      method: "POST",
      body: JSON.stringify({
        scene_id,
        zone_id,
        target_text,
        end_policy,
        ...(safety_bounds ? { safety_bounds } : {}),
      }),
    }),
  approve: (planId: string) => request<Run>(`/api/missions/${planId}/approve`, { method: "POST" }),
  revisePlan: (planId: string, base_version: number, parameters: MissionPlanParameters) =>
    request<MissionPlan>(`/api/missions/${planId}`, {
      method: "PATCH",
      body: JSON.stringify({ base_version, parameters }),
    }),
  control: (runId: string, action: string) =>
    request<Run>(`/api/runs/${runId}/${action}`, { method: "POST" }),
  candidate: (runId: string, decision: "accept" | "continue") =>
    request(`/api/runs/${runId}/candidate`, { method: "POST", body: JSON.stringify({ decision }) }),
  vlmChat: (
    message: string,
    history: VlmChatMessage[],
    target_text: string,
    scene_id: string,
    zone_id: string,
    end_policy: EndPolicy,
    safety_bounds?: SafetyBounds,
    run_id?: string,
    execute_command = true,
  ) => request<VlmChatResponse>("/api/vlm/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      history: history.slice(-12).map(({ role, content }) => ({ role, content })),
      target_text,
      scene_id,
      zone_id,
      end_policy,
      ...(safety_bounds ? { safety_bounds } : {}),
      ...(run_id ? { run_id } : {}),
      include_frame: true,
      execute_command,
    }),
  }),
};
