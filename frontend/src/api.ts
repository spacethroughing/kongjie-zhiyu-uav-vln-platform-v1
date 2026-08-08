import type { MissionPlan, Run, Scene } from "./types";

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
  scenes: () => request<Scene[]>("/api/scenes"),
  runs: () => request<Run[]>("/api/runs"),
  start: (scene_id: string) =>
    request<{ state: string }>("/api/simulator/start", { method: "POST", body: JSON.stringify({ scene_id }) }),
  stop: () => request("/api/simulator/stop", { method: "POST" }),
  plan: (scene_id: string, zone_id: string, target_text: string) =>
    request<MissionPlan>("/api/missions/plan", {
      method: "POST",
      body: JSON.stringify({ scene_id, zone_id, target_text, end_policy: "review_then_rth" }),
    }),
  approve: (planId: string) => request<Run>(`/api/missions/${planId}/approve`, { method: "POST" }),
  control: (runId: string, action: string) =>
    request<Run>(`/api/runs/${runId}/${action}`, { method: "POST" }),
  candidate: (runId: string, decision: "accept" | "continue") =>
    request(`/api/runs/${runId}/candidate`, { method: "POST", body: JSON.stringify({ decision }) }),
};

