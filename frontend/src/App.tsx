import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { DigitalTwinView } from "./DigitalTwinView";
import { MapView } from "./MapView";
import "./digital-twin.css";
import "./semantic.css";
import type { EndPolicy, HarnessEvent, LidarPointCloudFrame, MissionPlan, MissionPlanParameters, ProviderConfig, Run, SafetyBounds, Scene, SemanticMap, Telemetry, VlmChatMessage, Zone } from "./types";

const terminal = new Set(["SUCCEEDED", "FAILED", "ABORTED", "NOT_FOUND"]);

function boundsFromZone(zone: Zone | undefined): SafetyBounds | undefined {
  if (!zone) return undefined;
  const xs = zone.polygon.points.map(([x]) => x);
  const ys = zone.polygon.points.map(([, y]) => y);
  return {
    x_min: Math.min(...xs),
    x_max: Math.max(...xs),
    y_min: Math.min(...ys),
    y_max: Math.max(...ys),
  };
}

function safetyBoundsIssue(bounds: SafetyBounds | undefined, limit: SafetyBounds | undefined): string | undefined {
  if (!bounds || !Object.values(bounds).every(Number.isFinite)) return "请完整填写四个有限数值";
  if (bounds.x_min >= bounds.x_max || bounds.y_min >= bounds.y_max) return "最小值必须小于最大值";
  if (bounds.x_max - bounds.x_min < 4 || bounds.y_max - bounds.y_min < 4) return "X、Y 方向跨度均不能小于 4 m";
  if (!(bounds.x_min <= 0 && bounds.x_max >= 0 && bounds.y_min <= 0 && bounds.y_max >= 0)) {
    return "任务安全范围必须包含返航起点 (0, 0)";
  }
  if (limit && (
    bounds.x_min < limit.x_min || bounds.x_max > limit.x_max ||
    bounds.y_min < limit.y_min || bounds.y_max > limit.y_max
  )) return "任务范围不能超出场景硬限制";
  return undefined;
}

function planParameterIssue(parameters: MissionPlanParameters | undefined, scene: Scene | undefined): string | undefined {
  if (!parameters || !scene) return "计划参数尚未加载";
  if (!Object.values(parameters).every(Number.isFinite)) return "请完整填写全部有限数值";
  if (parameters.search_altitude_m <= 0 || parameters.search_altitude_m > 999) return "搜索高度必须大于 0 且不超过 999 m";
  if (parameters.max_speed_mps <= 0 || parameters.max_speed_mps > 999) return "最大速度必须大于 0 且不超过 999 m/s";
  if (parameters.approach_speed_mps <= 0 || parameters.approach_speed_mps > 999) return "接近速度必须大于 0 且不超过 999 m/s";
  if (parameters.approach_speed_mps > parameters.max_speed_mps) return "接近速度不能高于最大速度";
  if (parameters.min_standoff_m <= 0 || parameters.min_standoff_m > 999) return "目标距离必须大于 0 且不超过 999 m";
  if (parameters.min_clearance_m <= 0 || parameters.min_clearance_m > 999) return "避障净空必须大于 0 且不超过 999 m";
  if (parameters.lane_spacing_m <= 0 || parameters.lane_spacing_m > 999) return "航线间距必须大于 0 且不超过 999 m";
  if (parameters.max_mission_seconds < 30 || parameters.max_mission_seconds > 999) return "任务时限必须位于 30～999 s";
  if (parameters.mapping_coverage_target < 0.5 || parameters.mapping_coverage_target > 0.98) return "覆盖率必须位于 50%～98%";
  return undefined;
}

function eventSummary(event: HarnessEvent): string {
  const payload = event.payload;
  if (event.topic === "snapshot") {
    const runCount = Array.isArray(payload.runs) ? payload.runs.length : 0;
    return `WebSocket 已连接 · 仿真 ${payload.simulator_state ?? "UNKNOWN"} · 历史运行 ${runCount}`;
  }
  if (typeof payload.message === "string") return payload.message;
  if (typeof payload.error === "string" && payload.error) return payload.error;
  if (typeof payload.state === "string") {
    const scene = typeof payload.scene_id === "string" ? ` · ${payload.scene_id}` : "";
    return `状态切换为 ${payload.state}${scene}`;
  }
  if (typeof payload.phase === "string") return `阶段：${payload.phase}`;
  if (typeof payload.action === "string") return `控制指令：${payload.action}`;
  if (event.topic === "frame.preview") return `收到 ${payload.width ?? "?"}×${payload.height ?? "?"} 画面`;
  if (event.topic === "telemetry") return "遥测已更新";
  const compact = JSON.stringify(payload);
  return compact.length > 150 ? `${compact.slice(0, 147)}…` : compact;
}

function eventLevel(event: HarnessEvent): "info" | "warning" | "error" | "success" {
  if (event.payload.level === "error" || event.topic.includes("failed")) return "error";
  if (event.payload.level === "warning" || event.topic.includes("rejected")) return "warning";
  if (event.topic.includes("confirmed") || event.payload.phase === "completed") return "success";
  return "info";
}

const svgStyleProperties = [
  "fill", "fill-opacity", "stroke", "stroke-width", "stroke-opacity", "stroke-dasharray",
  "stroke-linecap", "stroke-linejoin", "opacity", "font-family", "font-size", "font-weight",
  "paint-order", "filter", "vector-effect", "shape-rendering",
] as const;

function serializedMapSvg(): string {
  const source = document.querySelector<SVGSVGElement>("#semantic-topology-map");
  if (!source) throw new Error("实时地图尚未就绪，无法导出");
  const clone = source.cloneNode(true) as SVGSVGElement;
  const sourceNodes = [source, ...source.querySelectorAll<SVGElement>("*")];
  const cloneNodes = [clone, ...clone.querySelectorAll<SVGElement>("*")];
  sourceNodes.forEach((node, index) => {
    const target = cloneNodes[index];
    if (!target) return;
    const computed = getComputedStyle(node);
    const inlineStyle = svgStyleProperties
      .map((property) => `${property}:${computed.getPropertyValue(property)}`)
      .join(";");
    target.setAttribute("style", inlineStyle);
  });
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", "1600");
  clone.setAttribute("height", "1000");
  clone.setAttribute("preserveAspectRatio", "xMidYMid meet");
  return `<?xml version="1.0" encoding="UTF-8"?>${new XMLSerializer().serializeToString(clone)}`;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function mapExportFilename(sceneId: string, extension: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `semantic-map-${sceneId}-${timestamp}.${extension}`;
}

export default function App() {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [sceneId, setSceneId] = useState("mock");
  const [zoneId, setZoneId] = useState("");
  const [simState, setSimState] = useState("STOPPED");
  const [provider, setProvider] = useState("-");
  const [providerConfig, setProviderConfig] = useState<ProviderConfig>();
  const [selectedModel, setSelectedModel] = useState("glm-4.6v-flashx");
  const [apiKey, setApiKey] = useState("");
  const [targetText, setTargetText] = useState("明亮的红色立方体");
  const [endPolicy, setEndPolicy] = useState<EndPolicy>("review_then_rth");
  const [plan, setPlan] = useState<MissionPlan>();
  const [planParameters, setPlanParameters] = useState<MissionPlanParameters>();
  const [run, setRun] = useState<Run>();
  const [telemetryPath, setTelemetryPath] = useState<Telemetry[]>([]);
  const [lidarFrames, setLidarFrames] = useState<LidarPointCloudFrame[]>([]);
  const [groundZ, setGroundZ] = useState<number>();
  const [events, setEvents] = useState<HarnessEvent[]>([]);
  const [preview, setPreview] = useState<string>();
  const [depthPreview, setDepthPreview] = useState<string>();
  const [depthRange, setDepthRange] = useState<{
    min?: number;
    max?: number;
    scaleMax?: number;
    source?: string;
    metricValid?: boolean;
    warning?: string;
  }>({});
  const [cameraYawDegrees, setCameraYawDegrees] = useState(0);
  const [semanticMap, setSemanticMap] = useState<SemanticMap>();
  const [candidateBox, setCandidateBox] = useState<{x_min:number;y_min:number;x_max:number;y_max:number}>();
  const [manualSafetyEnabled, setManualSafetyEnabled] = useState(false);
  const [manualSafetyBounds, setManualSafetyBounds] = useState<SafetyBounds>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [chatMessages, setChatMessages] = useState<VlmChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [allowVlmControl, setAllowVlmControl] = useState(true);
  const reconnect = useRef<number>();
  const missionConfigRevision = useRef(0);

  const scene = useMemo(() => scenes.find((item) => item.id === sceneId), [scenes, sceneId]);
  const zone = scene?.zones.find((item) => item.id === zoneId) ?? scene?.zones[0];
  const zonePresetBounds = useMemo(() => boundsFromZone(zone), [zone]);
  const hardSafetyBounds = scene?.manual_safety_bounds ?? zonePresetBounds;
  const manualSafetyError = manualSafetyEnabled
    ? safetyBoundsIssue(manualSafetyBounds, hardSafetyBounds)
    : undefined;
  const planParametersError = planParameterIssue(planParameters, scene);
  const planParametersDirty = Boolean(
    plan && planParameters && JSON.stringify(plan.parameters) !== JSON.stringify(planParameters),
  );

  function invalidatePlan() {
    missionConfigRevision.current += 1;
    setPlan(undefined);
    setPlanParameters(undefined);
  }

  useEffect(() => {
    setPlanParameters(plan?.parameters);
  }, [plan?.id]);

  useEffect(() => {
    Promise.all([api.scenes(), api.health(), api.runs(), api.providerConfig()])
      .then(([loadedScenes, health, runs, loadedProviderConfig]) => {
        setScenes(loadedScenes);
        setSimState(health.simulator_state);
        setProvider(health.provider);
        setProviderConfig(loadedProviderConfig);
        setSelectedModel(loadedProviderConfig.model);
        if (health.active_scene_id) setSceneId(health.active_scene_id);
        if (runs[0]) setRun(runs[0]);
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (scene?.zones[0] && !scene.zones.some((item) => item.id === zoneId)) setZoneId(scene.zones[0].id);
  }, [scene, zoneId]);

  useEffect(() => {
    if (!zonePresetBounds) return;
    setManualSafetyBounds(zonePresetBounds);
    invalidatePlan();
  }, [scene?.id, zone?.id]);

  useEffect(() => {
    let socket: WebSocket;
    const connect = () => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${location.host}/api/ws`);
      socket.onmessage = ({ data }) => {
        const event = JSON.parse(data) as HarnessEvent;
        // Image, map, telemetry and point-cloud events already have dedicated
        // state below. Retaining their large payloads in the generic log caused
        // needless full-app renders and could exhaust the tab during flight.
        if (!["telemetry", "frame.preview", "map.update", "lidar.points"].includes(event.topic)) {
          setEvents((items) => [event, ...items].slice(0, 120));
        }
        if (event.topic === "snapshot") {
          if (event.payload.map) setSemanticMap(event.payload.map as unknown as SemanticMap);
          if (event.payload.lidar) {
            setLidarFrames([{
              ...(event.payload.lidar as unknown as Omit<LidarPointCloudFrame, "timestamp">),
              timestamp: new Date().toISOString(),
            }]);
          }
        } else if (event.topic === "simulator.state") {
          setSimState(String(event.payload.state));
          if (event.payload.state === "STOPPED") {
            setPreview(undefined);
            setDepthPreview(undefined);
            setSemanticMap(undefined);
            setGroundZ(undefined);
            setTelemetryPath([]);
            setLidarFrames([]);
          }
        } else if (event.topic === "run.created") {
          setRun(event.payload.run as unknown as Run);
          setTelemetryPath([]);
          setLidarFrames([]);
        } else if (event.topic === "run.home") {
          const homePosition = event.payload.home_position as unknown as Run["home_position"];
          setRun((current) => current && current.id === event.run_id
            ? { ...current, home_position: homePosition }
            : current);
        } else if (event.topic === "run.state") {
          setRun((current) => current ? { ...current, state: String(event.payload.state) as Run["state"], error: event.payload.error as string } : current);
        } else if (event.topic === "mission.progress" && event.payload.run) {
          setRun(event.payload.run as unknown as Run);
        } else if (event.topic === "telemetry") {
          const telemetry = event.payload as unknown as Telemetry;
          if (telemetry.landed) setGroundZ(telemetry.position.z);
          setTelemetryPath((path) => [...path, telemetry].slice(-800));
        } else if (event.topic === "frame.preview") {
          if (typeof event.payload.data_url === "string") setPreview(event.payload.data_url);
          if (typeof event.payload.camera_yaw_degrees === "number") {
            setCameraYawDegrees(event.payload.camera_yaw_degrees);
          }
          if (typeof event.payload.depth_data_url === "string") {
            setDepthPreview(event.payload.depth_data_url);
            setDepthRange({
              min: typeof event.payload.depth_min_m === "number" ? event.payload.depth_min_m : undefined,
              max: typeof event.payload.depth_max_m === "number" ? event.payload.depth_max_m : undefined,
              scaleMax: typeof event.payload.depth_scale_max_m === "number" ? event.payload.depth_scale_max_m : undefined,
              source: typeof event.payload.depth_source === "string" ? event.payload.depth_source : undefined,
              metricValid: typeof event.payload.depth_metric_valid === "boolean" ? event.payload.depth_metric_valid : undefined,
              warning: typeof event.payload.depth_warning === "string" ? event.payload.depth_warning : undefined,
            });
          }
        } else if (event.topic === "map.update") {
          setSemanticMap(event.payload as unknown as SemanticMap);
        } else if (event.topic === "lidar.points") {
          const lidarFrame = {
            ...(event.payload as unknown as Omit<LidarPointCloudFrame, "timestamp">),
            timestamp: event.timestamp ?? new Date().toISOString(),
          };
          setLidarFrames((frames) => [...frames, lidarFrame].slice(-3));
        } else if (event.topic === "vision.assessment") {
          setCandidateBox((event.payload.bbox_norm as typeof candidateBox) ?? undefined);
        } else if (event.topic === "provider.configured") {
          const model = String(event.payload.model);
          setProvider(String(event.payload.provider));
          setSelectedModel(model);
          setProviderConfig((current) => current ? {
            ...current,
            provider: String(event.payload.provider),
            model,
            api_key_configured: Boolean(event.payload.api_key_configured),
          } : current);
        }
      };
      socket.onclose = () => { reconnect.current = window.setTimeout(connect, 1500); };
    };
    connect();
    return () => { window.clearTimeout(reconnect.current); socket?.close(); };
  }, []);

  async function act(task: () => Promise<unknown>) {
    setBusy(true); setError(undefined);
    try { await task(); } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }

  function updateSafetyBound(key: keyof SafetyBounds, value: number) {
    if (Number.isFinite(value) && hardSafetyBounds) {
      const axis = key.startsWith("x") ? "x" : "y";
      value = Math.min(hardSafetyBounds[`${axis}_max`], Math.max(hardSafetyBounds[`${axis}_min`], value));
    }
    setManualSafetyBounds((current) => ({ ...(current ?? zonePresetBounds!), [key]: value }));
    invalidatePlan();
  }

  function replaceSafetyBounds(bounds: SafetyBounds) {
    setManualSafetyBounds(bounds);
    invalidatePlan();
  }

  async function createPlan() {
    const revision = missionConfigRevision.current;
    const created = await api.plan(
      sceneId,
      zone?.id ?? "",
      targetText,
      endPolicy,
      manualSafetyEnabled ? manualSafetyBounds : undefined,
    );
    if (revision === missionConfigRevision.current) setPlan(created);
  }

  function updatePlanParameter(key: keyof MissionPlanParameters, value: number) {
    setPlanParameters((current) => current ? { ...current, [key]: value } : current);
  }

  async function applyPlanParameters() {
    if (!plan || !planParameters) throw new Error("计划参数尚未加载");
    if (planParametersError) throw new Error(planParametersError);
    const revised = await api.revisePlan(plan.id, plan.version, planParameters);
    setPlan(revised);
  }

  async function saveProviderConfig() {
    const configured = await api.configureProvider(selectedModel, apiKey.trim() || undefined);
    setProviderConfig(configured);
    setProvider(configured.provider);
    setSelectedModel(configured.model);
    setApiKey("");
    invalidatePlan();
  }

  async function exportMapPng() {
    const svg = serializedMapSvg();
    const svgUrl = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
    try {
      const image = new Image();
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error("地图图像渲染失败"));
        image.src = svgUrl;
      });
      const canvas = document.createElement("canvas");
      canvas.width = 1600;
      canvas.height = 1000;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("浏览器不支持地图图像导出");
      context.fillStyle = "#060c17";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const png = await new Promise<Blob>((resolve, reject) => canvas.toBlob(
        (blob) => blob ? resolve(blob) : reject(new Error("地图 PNG 编码失败")),
        "image/png",
      ));
      downloadBlob(png, mapExportFilename(sceneId, "png"));
    } finally {
      URL.revokeObjectURL(svgUrl);
    }
  }

  async function exportMapJson() {
    const payload = {
      schema: "airsim-llm-harness.semantic-map.v1",
      exported_at: new Date().toISOString(),
      scene: { id: sceneId, name: scene?.name },
      zone,
      mission: {
        run_id: run?.id,
        state: currentState,
        target_text: targetText,
        target_position: run?.target_position,
      },
      vehicle: {
        camera_yaw_degrees: cameraYawDegrees,
        latest_telemetry: telemetryPath.at(-1),
        telemetry_path: telemetryPath,
      },
      safety: {
        coordinate_frame: "run-home-relative-ned",
        frame_origin: mapFrameOrigin,
        hard_bounds: hardSafetyBounds,
        task_bounds: manualSafetyEnabled ? manualSafetyBounds : undefined,
      },
      semantic_map: semanticMap,
    };
    downloadBlob(
      new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" }),
      mapExportFilename(sceneId, "json"),
    );
  }

  const latest = telemetryPath.at(-1);
  const active = run && !terminal.has(run.state);
  const mapFrameOrigin = active
    ? (run.home_position ?? telemetryPath.at(0)?.position ?? latest?.position)
    : latest?.position;
  const currentState = active ? run.state : simState === "READY" ? "READY" : "IDLE";

  async function submitVlmChat() {
    const message = chatInput.trim();
    if (!message || chatBusy) return;
    const userEntry: VlmChatMessage = { role: "user", content: message };
    const history = [...chatMessages, userEntry].slice(-12);
    setChatMessages(history);
    setChatInput("");
    setChatBusy(true);
    try {
      const response = await api.vlmChat(
        message,
        history,
        targetText,
        sceneId,
        zone?.id ?? "",
        endPolicy,
        manualSafetyEnabled ? manualSafetyBounds : undefined,
        active ? run.id : undefined,
        allowVlmControl,
      );
      if (response.mission_plan) {
        setPlan(response.mission_plan);
        setTargetText(
          response.mission_plan.request.targets.length
            ? response.mission_plan.request.targets.join("、")
            : response.mission_plan.request.target_text,
        );
      }
      const assistantEntry: VlmChatMessage = {
        role: "assistant",
        content: response.reply,
        action: response.executed_action ?? response.requested_action ?? (response.mission_plan ? "plan-mission" : undefined),
        error: response.command_error,
        heading_degrees: response.heading_degrees,
        distance_m: response.distance_m,
        altitude_delta_m: response.altitude_delta_m,
        target_altitude_m: response.target_altitude_m,
        command_status: response.command_status,
        context_target_text: response.context_target_text,
        mission_mode: response.mission_intent?.kind,
        mission_plan_id: response.mission_plan?.id,
        task_breakdown: response.task_breakdown,
      };
      setChatMessages((items) => [...items, assistantEntry].slice(-30));
    } catch (reason) {
      const assistantEntry: VlmChatMessage = {
        role: "assistant",
        content: "VLM 暂时无法响应。",
        error: reason instanceof Error ? reason.message : String(reason),
      };
      setChatMessages((items) => [...items, assistantEntry].slice(-30));
    } finally {
      setChatBusy(false);
    }
  }
  const logEvents = events;

  return (
    <main className="app-shell">
      <header className="command-header">
        <div className="brand">
          <span className="mark" aria-hidden="true">
            <svg viewBox="0 0 48 48" focusable="false">
              <g className="drone-rotors">
                <circle cx="9" cy="12" r="5" />
                <circle cx="39" cy="12" r="5" />
                <circle cx="9" cy="36" r="5" />
                <circle cx="39" cy="36" r="5" />
              </g>
              <path className="drone-arms" d="M19 20 12.5 15.5M29 20l6.5-4.5M19 28l-6.5 4.5M29 28l6.5 4.5" />
              <path className="drone-body" d="m18 19 6-3 6 3 3 5-4 6H19l-4-6Z" />
              <circle className="drone-camera" cx="24" cy="24" r="2.4" />
              <path className="drone-heading" d="M24 15V8m-3 3 3-3 3 3" />
            </svg>
          </span>
          <div className="brand-copy">
            <p className="eyebrow">SKYBOUND · VISION FLIGHT DECK</p>
            <h1>空界智语</h1>
            <p className="brand-subtitle">无人机视觉语言导航平台</p>
          </div>
        </div>
        <div className="flight-horizon" aria-hidden="true">
          <span>−30</span><i /><span>−15</span><i /><b>0°</b><i /><span>15</span><i /><span>30</span>
        </div>
        <div className="status-cluster">
          <div className="status-item"><small>SIM LINK</small><strong><span className={`dot ${simState === "READY" ? "online" : ""}`} />{simState}</strong></div>
          <div className="status-item provider"><small>VISION CORE</small><strong>{providerConfig?.model ?? provider}</strong></div>
        </div>
      </header>

      <section className="ops-ribbon" aria-label="任务态势摘要">
        <div><small>SCENE / 场景</small><strong>{scene?.name ?? "未选择"}</strong></div>
        <div><small>MISSION / 任务</small><strong>{currentState}</strong></div>
        <div className="ribbon-target"><small>OBJECTIVE / 开放词汇目标</small><strong>{targetText || "未设定"}</strong></div>
        <div><small>DATALINK / 数据链</small><strong className={preview ? "signal-live" : ""}>{preview ? "VIDEO ACTIVE" : "STANDBY"}</strong></div>
      </section>

      {error && <div className="error"><b>操作失败</b><span>{error}</span><button onClick={() => setError(undefined)}>×</button></div>}

      <section className="layout">
        <aside className="left-stack">
          <article className="panel provider-panel">
            <div className="panel-title"><span>VLM</span><h2>模型配置</h2><em>{providerConfig?.api_key_configured ? "KEY READY" : "NO KEY"}</em></div>
            <label>视觉模型
              <select
                aria-label="视觉模型"
                value={selectedModel}
                disabled={busy || Boolean(active)}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                {providerConfig?.models.map((model) => <option value={model.id} key={model.id}>
                  {model.name} · {model.billing === "free" ? "免费" : "计费"}
                </option>)}
              </select>
            </label>
            <p className="model-description">
              {providerConfig?.models.find((model) => model.id === selectedModel)?.description ?? "选择智谱视觉模型"}
            </p>
            <label>API Key
              <input
                className="api-key-input"
                type="password"
                name="runtime-api-key"
                autoComplete="off"
                spellCheck={false}
                aria-label="智谱 API Key"
                value={apiKey}
                placeholder={providerConfig?.api_key_configured ? "已配置；留空保留当前密钥" : "输入智谱 API Key"}
                disabled={busy || Boolean(active)}
                onChange={(event) => setApiKey(event.target.value)}
              />
            </label>
            <button
              className="provider-save"
              disabled={busy || Boolean(active) || !selectedModel || (!providerConfig?.api_key_configured && !apiKey.trim())}
              onClick={() => act(saveProviderConfig)}
            >保存并探测能力</button>
            <p className="secret-note">仅当前后端进程内存保存；页面不会回显密钥，任务运行中禁止切换。</p>
          </article>

          <article className="panel mission-panel">
            <div className="panel-title"><span>MSN</span><h2>任务配置</h2><em>MISSION AUTHORING</em></div>
            <label>仿真场景<select value={sceneId} disabled={simState !== "STOPPED"} onChange={(e) => { setSceneId(e.target.value); invalidatePlan(); }}>{scenes.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label>
            <label>搜索区域<select value={zone?.id ?? ""} onChange={(e) => { setZoneId(e.target.value); invalidatePlan(); }}>{scene?.zones.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label>
            <section className={`safety-editor ${manualSafetyEnabled ? "enabled" : ""}`} aria-label="任务安全范围">
              <div className="safety-editor-head">
                <label className="safety-toggle">
                  <span>手动安全范围</span>
                  <input
                    type="checkbox"
                    role="switch"
                    checked={manualSafetyEnabled}
                    onChange={(event) => {
                      const enabled = event.target.checked;
                      setManualSafetyEnabled(enabled);
                      if (enabled && zonePresetBounds) setManualSafetyBounds(zonePresetBounds);
                      invalidatePlan();
                    }}
                  />
                  <i aria-hidden="true" />
                </label>
                <button
                  type="button"
                  className="restore-bounds"
                  disabled={!manualSafetyEnabled || !zonePresetBounds}
                  onClick={() => zonePresetBounds && replaceSafetyBounds(zonePresetBounds)}
                >恢复区域预设</button>
              </div>
              {hardSafetyBounds && <p className="safety-limit">
                允许范围：X {hardSafetyBounds.x_min.toFixed(1)} ～ {hardSafetyBounds.x_max.toFixed(1)} m · Y {hardSafetyBounds.y_min.toFixed(1)} ～ {hardSafetyBounds.y_max.toFixed(1)} m
                <br />坐标相对本次任务起飞点；地图与数字孪生会平移到 AirSim NED 实际位置。
              </p>}
              {manualSafetyEnabled && manualSafetyBounds && <>
                <div className="bounds-grid">
                  {(["x_min", "x_max", "y_min", "y_max"] as const).map((key) => {
                    const axis = key.startsWith("x") ? "x" : "y";
                    const edge = key.endsWith("min") ? "最小" : "最大";
                    return <label key={key}>{axis.toUpperCase()} {edge}
                      <input
                        type="number"
                        step="0.5"
                        min={hardSafetyBounds?.[`${axis}_min`]}
                        max={hardSafetyBounds?.[`${axis}_max`]}
                        value={Number.isFinite(manualSafetyBounds[key]) ? manualSafetyBounds[key] : ""}
                        aria-label={`${axis.toUpperCase()} ${edge}值`}
                        onChange={(event) => updateSafetyBound(key, event.currentTarget.valueAsNumber)}
                      />
                    </label>;
                  })}
                </div>
                <p className={`bounds-validation ${manualSafetyError ? "invalid" : "valid"}`}>
                  {manualSafetyError ?? "范围有效，将仅对此任务生效"}
                </p>
              </>}
            </section>
            <label>开放词汇目标<textarea rows={3} value={targetText} onChange={(e) => { setTargetText(e.target.value); setChatMessages([]); invalidatePlan(); }} /></label>
            <label>任务结束方式
              <select
                aria-label="任务结束方式"
                value={endPolicy}
                onChange={(event) => {
                  setEndPolicy(event.target.value as EndPolicy);
                  invalidatePlan();
                }}
              >
                <option value="review_then_rth">人工确认后返航</option>
                <option value="auto_rth">自动返航</option>
                <option value="land_at_target">目标安全距离处原地降落</option>
              </select>
            </label>
            <div className="button-row">
              {simState === "STOPPED" ? <button className="primary" disabled={busy} onClick={() => act(() => api.start(sceneId))}>启动场景</button> : <button disabled={busy || Boolean(active)} onClick={() => act(() => api.stop())}>停止场景</button>}
              <button disabled={busy || simState !== "READY" || !targetText.trim() || !zone || Boolean(manualSafetyError)} onClick={() => act(createPlan)}>生成计划</button>
            </div>
          </article>

          <article className="panel plan-panel">
            <div className="panel-title"><span>CHK</span><h2>计划审核</h2><em>SAFETY GATE</em></div>
            {plan ? <>
              <div className="plan-metrics"><div><b>{plan.route.length}</b><small>观察航点</small></div><div><b>{plan.observation_yaws_deg.length}</b><small>扫描方向</small></div><div><b>v{plan.version}</b><small>计划版本</small></div></div>
              {planParameters && <section className={`plan-parameter-editor ${planParametersDirty ? "dirty" : ""}`} aria-label="计划关键参数">
                <div className="plan-parameter-head">
                  <div><b>关键参数</b><small>输入范围 -999～999；距离、速度和时限须为正</small></div>
                  <span>{planParametersDirty ? "UNAPPLIED" : `FROZEN · V${plan.version}`}</span>
                </div>
                <div className="plan-parameter-grid">
                  <label>搜索高度 <small>m</small><input type="number" step="0.5" min="-999" max="999" value={Number.isFinite(planParameters.search_altitude_m) ? planParameters.search_altitude_m : ""} disabled={busy || Boolean(active)} onChange={(event) => updatePlanParameter("search_altitude_m", event.currentTarget.valueAsNumber)} /></label>
                  <label>航线间距 <small>m</small><input type="number" step="0.5" min="-999" max="999" value={Number.isFinite(planParameters.lane_spacing_m) ? planParameters.lane_spacing_m : ""} disabled={busy || Boolean(active)} onChange={(event) => updatePlanParameter("lane_spacing_m", event.currentTarget.valueAsNumber)} /></label>
                  <label>最大速度 <small>m/s</small><input type="number" step="0.2" min="-999" max="999" value={Number.isFinite(planParameters.max_speed_mps) ? planParameters.max_speed_mps : ""} disabled={busy || Boolean(active)} onChange={(event) => updatePlanParameter("max_speed_mps", event.currentTarget.valueAsNumber)} /></label>
                  <label>接近速度 <small>m/s</small><input type="number" step="0.1" min="-999" max="999" value={Number.isFinite(planParameters.approach_speed_mps) ? planParameters.approach_speed_mps : ""} disabled={busy || Boolean(active)} onChange={(event) => updatePlanParameter("approach_speed_mps", event.currentTarget.valueAsNumber)} /></label>
                  <label>目标距离 <small>m</small><input type="number" step="0.5" min="-999" max="999" value={Number.isFinite(planParameters.min_standoff_m) ? planParameters.min_standoff_m : ""} disabled={busy || Boolean(active)} onChange={(event) => updatePlanParameter("min_standoff_m", event.currentTarget.valueAsNumber)} /></label>
                  <label>避障净空 <small>m</small><input type="number" step="0.25" min="-999" max="999" value={Number.isFinite(planParameters.min_clearance_m) ? planParameters.min_clearance_m : ""} disabled={busy || Boolean(active)} onChange={(event) => updatePlanParameter("min_clearance_m", event.currentTarget.valueAsNumber)} /></label>
                  <label>任务时限 <small>s</small><input type="number" step="30" min="-999" max="999" value={Number.isFinite(planParameters.max_mission_seconds) ? planParameters.max_mission_seconds : ""} disabled={busy || Boolean(active)} onChange={(event) => updatePlanParameter("max_mission_seconds", event.currentTarget.valueAsNumber)} /></label>
                  {plan.request.mission_mode === "semantic_mapping" && <label>覆盖率 <small>%</small><input type="number" step="1" min="50" max="98" value={Number.isFinite(planParameters.mapping_coverage_target) ? Math.round(planParameters.mapping_coverage_target * 100) : ""} disabled={busy || Boolean(active)} onChange={(event) => updatePlanParameter("mapping_coverage_target", event.currentTarget.valueAsNumber / 100)} /></label>}
                </div>
                <p className={`plan-parameter-status ${planParametersError ? "invalid" : planParametersDirty ? "pending" : "valid"}`}>
                  {planParametersError ?? (planParametersDirty ? "存在未应用修改；重新计算前禁止批准执行" : `参数已冻结在计划 v${plan.version}`)}
                </p>
                <div className="plan-parameter-actions">
                  <button type="button" disabled={busy || Boolean(active) || !planParametersDirty || Boolean(planParametersError)} onClick={() => act(applyPlanParameters)}>应用并重算航线</button>
                  <button type="button" disabled={busy || Boolean(active) || !planParametersDirty} onClick={() => setPlanParameters(plan.parameters)}>撤销未应用修改</button>
                </div>
              </section>}
              <div className="mission-task-list" aria-label="复合任务步骤">
                {plan.tasks.map((task, index) => <div key={task.id}><b>{String(index + 1).padStart(2, "0")}</b><span>{task.label}</span><small>{task.kind === "semantic_mapping" ? `覆盖率目标 ${Math.round((task.coverage_target ?? .85) * 100)}%` : "独立视觉查询"}</small></div>)}
              </div>
              <ul>{plan.safety_summary.map((item) => <li key={item}>{item}</li>)}</ul>
              <button className="approve" disabled={busy || Boolean(active) || simState !== "READY" || planParametersDirty || Boolean(planParametersError)} onClick={() => act(async () => setRun(await api.approve(plan.id)))}>批准并执行 v{plan.version}</button>
            </> : <div className="empty">生成计划后，必须在此审核安全摘要才能起飞。</div>}
          </article>
        </aside>

        <section className="center-stack">
          <article className="panel video-panel">
            <div className="panel-title"><span>LIVE</span><h2>机载 RGB / 深度</h2><em>{latest && groundZ !== undefined ? `${(groundZ - latest.position.z).toFixed(1)} m AGL` : "NO FEED"}</em></div>
            <div className="sensor-grid">
              <div className="video-stage">
                <b className="sensor-badge">RGB</b>
                {preview ? <img src={preview} alt="无人机相机实时关键帧" /> : <div className="camera-empty"><div className="reticle" /><span>等待观察关键帧</span></div>}
                {candidateBox && <div className="bbox" style={{left:`${candidateBox.x_min*100}%`,top:`${candidateBox.y_min*100}%`,width:`${(candidateBox.x_max-candidateBox.x_min)*100}%`,height:`${(candidateBox.y_max-candidateBox.y_min)*100}%`}}><span>VLM CANDIDATE</span></div>}
              </div>
              <div className="video-stage depth-stage">
                <b className="sensor-badge">{depthRange.source === "depth-vis-fallback" ? "DEPTH VIS · 0–100m" : "DEPTH PLANAR"}</b>
                {depthPreview ? <img src={depthPreview} alt="无人机实时深度图" /> : <div className="camera-empty"><span>等待深度帧</span></div>}
                <div className="depth-scale"><span>近 0.5m</span><i /><span>远 {depthRange.scaleMax?.toFixed(0) ?? "50"}m</span></div>
                {depthRange.min !== undefined && <small className="depth-range">当前 {depthRange.min.toFixed(1)}–{depthRange.max?.toFixed(1) ?? "?"}m</small>}
                {depthRange.warning && <small className={`depth-warning ${depthRange.metricValid === false ? "invalid" : ""}`}>{depthRange.warning}</small>}
              </div>
            </div>
            <div className="telemetry-strip">
              <span>X <b>{latest?.position.x.toFixed(1) ?? "—"}</b></span><span>Y <b>{latest?.position.y.toFixed(1) ?? "—"}</b></span><span>Z <b>{latest?.position.z.toFixed(1) ?? "—"}</b></span><span>V <b>{latest ? Math.hypot(latest.velocity.x, latest.velocity.y, latest.velocity.z).toFixed(1) : "—"}</b> m/s</span><span>CAM <b>{cameraYawDegrees.toFixed(0)}°</b></span><span>{latest?.armed ? "ARMED" : "DISARMED"}</span>
            </div>
          </article>

          <article className="panel twin-panel">
            <div className="panel-title">
              <span>TWIN</span><h2>无人机实时数字孪生</h2>
              <em>{latest ? `${latest.position.x.toFixed(1)}, ${latest.position.y.toFixed(1)}, ${latest.position.z.toFixed(1)} NED` : "3D AIRSPACE"}</em>
            </div>
            <DigitalTwinView
              zone={zone}
              safetyBounds={manualSafetyEnabled ? manualSafetyBounds : zonePresetBounds}
              telemetryPath={telemetryPath}
              semanticMap={semanticMap}
              lidarFrames={lidarFrames}
              cameraYawDegrees={cameraYawDegrees}
              frameOrigin={mapFrameOrigin}
              groundZ={groundZ}
              target={active ? run?.target_position : undefined}
              state={currentState}
            />
          </article>

          <article className="panel map-panel">
            <div className="panel-title map-panel-title">
              <span>MAP</span><h2>实时占据与语义拓扑图</h2><em>{zone?.name}</em>
              <div className="map-export-actions" aria-label="地图导出">
                <button type="button" disabled={busy || !zone} onClick={() => act(exportMapPng)} title="导出当前地图为 PNG 图像">PNG</button>
                <button type="button" disabled={busy || !zone} onClick={() => act(exportMapJson)} title="导出地图、轨迹和语义数据为 JSON">JSON</button>
              </div>
            </div>
            <MapView
              zone={zone}
              hardBounds={hardSafetyBounds}
              safetyBounds={manualSafetyEnabled ? manualSafetyBounds : undefined}
              telemetryPath={telemetryPath}
              semanticMap={semanticMap}
              cameraYawDegrees={cameraYawDegrees}
              frameOrigin={mapFrameOrigin}
              target={active ? run?.target_position : undefined}
              onSafetyBoundsChange={manualSafetyEnabled ? replaceSafetyBounds : undefined}
            />
          </article>
        </section>

        <aside className="right-stack">
          <article className="panel run-panel">
            <div className="panel-title"><span>RUN</span><h2>执行控制</h2></div>
            <div className={`state-card state-${currentState.toLowerCase()}`}><small>CURRENT STATE</small><strong>{currentState}</strong><code>{active ? run.id.slice(0, 13) : run ? `历史 ${run.state} · ${run.id.slice(0, 13)}` : "no active run"}</code></div>
            {run?.task_progress?.length ? <div className="run-task-progress" aria-label="复合任务执行进度">
              {run.task_progress.map((task, index) => <div className={`task-${task.state}`} key={task.task_id}>
                <b>{index + 1}</b><span>{task.label}</span><small>{task.coverage_ratio !== undefined ? `${Math.round(task.coverage_ratio * 100)}%` : task.state}</small>
              </div>)}
            </div> : null}
            <div className="controls">
              <button disabled={!active || run?.state === "PAUSED"} onClick={() => act(() => api.control(run!.id, "pause"))}>暂停</button>
              <button disabled={run?.state !== "PAUSED"} onClick={() => act(() => api.control(run!.id, "resume"))}>继续</button>
              <button disabled={!active} onClick={() => act(() => api.control(run!.id, "return-home"))}>返航</button>
              <button disabled={!active} onClick={() => act(() => api.control(run!.id, "land"))}>就地降落</button>
              <button className="danger" disabled={!active} onClick={() => act(() => api.control(run!.id, "abort"))}>终止任务</button>
              <button className="hard" disabled={!active} onClick={() => confirm("硬停会立即终止 UE 仿真进程。确认继续？") && act(() => api.control(run!.id, "hard-stop"))}>仿真硬停</button>
            </div>
            {run?.state === "EVIDENCE" && <div className="candidate-actions"><b>候选目标等待复核</b><button className="approve" onClick={() => act(() => api.candidate(run.id, "accept"))}>确认并返航</button><button onClick={() => act(() => api.candidate(run.id, "continue"))}>误检，继续搜索</button></div>}
          </article>

          <article className="panel chat-panel">
            <div className="panel-title"><span>VLM</span><h2>实时对话与控制</h2><em>{chatBusy ? "THINKING" : "LIVE"}</em></div>
            <div className="chat-context" aria-label="VLM 对话上下文">
              <span>{currentState}</span>
              <span>{preview ? "机载画面" : "无画面"}</span>
              <span>探索 {semanticMap?.stats.explored_cells ?? 0}</span>
              <span title={targetText}>目标：{targetText || "未设置"}</span>
            </div>
            <div className="chat-messages" role="log" aria-label="VLM 实时反馈">
              {chatMessages.length ? chatMessages.map((message, index) => <div
                className={`chat-message chat-${message.role}`}
                key={`${message.role}-${index}-${message.content.slice(0, 12)}`}
              >
                <b>{message.role === "user" ? "YOU" : "VLM"}</b>
                <p>{message.content}</p>
                {message.action && <small>
                  指令：{message.action}
                  {message.action === "explore" && message.heading_degrees !== undefined && message.distance_m !== undefined
                    ? ` · 航向 ${message.heading_degrees.toFixed(0)}° · ${message.distance_m.toFixed(1)} m`
                    : ""}
                  {message.action === "change-altitude" && message.target_altitude_m !== undefined
                    ? ` · 目标高度 ${message.target_altitude_m.toFixed(1)} m`
                    : ""}
                  {message.action === "change-altitude" && message.altitude_delta_m !== undefined
                    ? ` · 高度变化 ${message.altitude_delta_m > 0 ? "+" : ""}${message.altitude_delta_m.toFixed(1)} m`
                    : ""}
                  {message.command_status === "queued" ? " · 已进入安全队列" : ""}
                  {message.command_status === "planned" ? " · 已生成计划，等待审核" : ""}
                </small>}
                {message.task_breakdown?.length ? <div className="chat-task-breakdown">
                  {message.task_breakdown.map((task, taskIndex) => <span key={task.id}><b>{taskIndex + 1}</b>{task.label}</span>)}
                </div> : null}
                {message.context_target_text && <small>本次上下文目标：{message.context_target_text}</small>}
                {message.error && <small className="chat-error">未执行：{message.error}</small>}
              </div>) : <div className="chat-empty">支持自然语言复合任务：例如“探索圆锥体和橙色球体”，或“探索整片区域并建立占据与语义拓扑图”。也可发送方向、距离、高度和紧急控制指令。</div>}
              {chatBusy && <div className="chat-message chat-assistant pending"><b>VLM</b><p>正在结合机载画面与实时地图分析…</p></div>}
            </div>
            <form className="chat-form" onSubmit={(event) => { event.preventDefault(); void submitVlmChat(); }}>
              <textarea
                aria-label="发送给 VLM 的消息"
                rows={2}
                value={chatInput}
                placeholder="例如：探索圆锥体和橙色球体；或覆盖整片区域建立语义拓扑图"
                onChange={(event) => setChatInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void submitVlmChat();
                  }
                }}
              />
              <div className="chat-actions">
                <label className="chat-control-toggle">
                  <input type="checkbox" checked={allowVlmControl} onChange={(event) => setAllowVlmControl(event.target.checked)} />
                  允许生成任务并执行白名单飞行指令
                </label>
                <button className="primary" type="submit" disabled={chatBusy || !chatInput.trim()}>发送</button>
              </div>
            </form>
          </article>

          <article className="panel events-panel">
            <div className="panel-title"><span>LOG</span><h2>实时日志</h2><button className="log-clear" onClick={() => setEvents([])}>清空</button></div>
            <div className="event-list">{logEvents.length ? logEvents.map((event) => <div className={`event level-${eventLevel(event)}`} key={`${event.sequence}-${event.topic}`}><div className="event-head"><time>{event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "—"}</time><span>{event.topic}</span><small>#{event.sequence}</small></div><p>{eventSummary(event)}</p></div>) : <div className="empty">等待系统日志</div>}</div>
          </article>
        </aside>
      </section>
    </main>
  );
}
