import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { MapView } from "./MapView";
import type { HarnessEvent, MissionPlan, Run, Scene, Telemetry } from "./types";

const terminal = new Set(["SUCCEEDED", "FAILED", "ABORTED", "NOT_FOUND"]);

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

export default function App() {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [sceneId, setSceneId] = useState("mock");
  const [zoneId, setZoneId] = useState("");
  const [simState, setSimState] = useState("STOPPED");
  const [provider, setProvider] = useState("-");
  const [targetText, setTargetText] = useState("明亮的红色立方体");
  const [plan, setPlan] = useState<MissionPlan>();
  const [run, setRun] = useState<Run>();
  const [telemetryPath, setTelemetryPath] = useState<Telemetry[]>([]);
  const [events, setEvents] = useState<HarnessEvent[]>([]);
  const [preview, setPreview] = useState<string>();
  const [candidateBox, setCandidateBox] = useState<{x_min:number;y_min:number;x_max:number;y_max:number}>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const reconnect = useRef<number>();

  const scene = useMemo(() => scenes.find((item) => item.id === sceneId), [scenes, sceneId]);
  const zone = scene?.zones.find((item) => item.id === zoneId) ?? scene?.zones[0];

  useEffect(() => {
    Promise.all([api.scenes(), api.health(), api.runs()])
      .then(([loadedScenes, health, runs]) => {
        setScenes(loadedScenes);
        setSimState(health.simulator_state);
        setProvider(health.provider);
        if (health.active_scene_id) setSceneId(health.active_scene_id);
        if (runs[0]) setRun(runs[0]);
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (scene?.zones[0] && !scene.zones.some((item) => item.id === zoneId)) setZoneId(scene.zones[0].id);
  }, [scene, zoneId]);

  useEffect(() => {
    let socket: WebSocket;
    const connect = () => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${location.host}/api/ws`);
      socket.onmessage = ({ data }) => {
        const event = JSON.parse(data) as HarnessEvent;
        setEvents((items) => [event, ...items].slice(0, 120));
        if (event.topic === "simulator.state") {
          setSimState(String(event.payload.state));
        } else if (event.topic === "run.created") {
          setRun(event.payload.run as unknown as Run);
          setTelemetryPath([]);
        } else if (event.topic === "run.state") {
          setRun((current) => current ? { ...current, state: String(event.payload.state) as Run["state"], error: event.payload.error as string } : current);
        } else if (event.topic === "telemetry") {
          const telemetry = event.payload as unknown as Telemetry;
          setTelemetryPath((path) => [...path, telemetry].slice(-800));
        } else if (event.topic === "frame.preview") {
          setPreview(String(event.payload.data_url));
        } else if (event.topic === "vision.assessment") {
          setCandidateBox((event.payload.bbox_norm as typeof candidateBox) ?? undefined);
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

  const latest = telemetryPath.at(-1);
  const active = run && !terminal.has(run.state);
  const currentState = active ? run.state : simState === "READY" ? "READY" : "IDLE";
  const logEvents = events.filter((event) => !["telemetry", "frame.preview"].includes(event.topic));

  return (
    <main>
      <header>
        <div className="brand"><span className="mark">A</span><div><h1>AIRSIM MISSION HARNESS</h1><p>LLM-assisted · deterministic flight safety</p></div></div>
        <div className="status-line"><span className={`dot ${simState === "READY" ? "online" : ""}`} />{simState}<span className="provider">VLM · {provider}</span></div>
      </header>

      {error && <div className="error"><b>操作失败</b><span>{error}</span><button onClick={() => setError(undefined)}>×</button></div>}

      <section className="layout">
        <aside className="left-stack">
          <article className="panel mission-panel">
            <div className="panel-title"><span>01</span><h2>任务配置</h2></div>
            <label>仿真场景<select value={sceneId} disabled={simState !== "STOPPED"} onChange={(e) => setSceneId(e.target.value)}>{scenes.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label>
            <label>搜索区域<select value={zone?.id ?? ""} onChange={(e) => setZoneId(e.target.value)}>{scene?.zones.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label>
            <label>开放词汇目标<textarea rows={3} value={targetText} onChange={(e) => setTargetText(e.target.value)} /></label>
            <div className="button-row">
              {simState === "STOPPED" ? <button className="primary" disabled={busy} onClick={() => act(() => api.start(sceneId))}>启动场景</button> : <button disabled={busy || Boolean(active)} onClick={() => act(() => api.stop())}>停止场景</button>}
              <button disabled={busy || simState !== "READY" || !targetText.trim()} onClick={() => act(async () => setPlan(await api.plan(sceneId, zone?.id ?? "", targetText)))}>生成计划</button>
            </div>
          </article>

          <article className="panel plan-panel">
            <div className="panel-title"><span>02</span><h2>计划审核</h2></div>
            {plan ? <>
              <div className="plan-metrics"><div><b>{plan.route.length}</b><small>观察航点</small></div><div><b>{plan.observation_yaws_deg.length}</b><small>扫描方向</small></div><div><b>v{plan.version}</b><small>计划版本</small></div></div>
              <ul>{plan.safety_summary.map((item) => <li key={item}>{item}</li>)}</ul>
              <button className="approve" disabled={busy || Boolean(active) || simState !== "READY"} onClick={() => act(async () => setRun(await api.approve(plan.id)))}>批准并执行</button>
            </> : <div className="empty">生成计划后，必须在此审核安全摘要才能起飞。</div>}
          </article>
        </aside>

        <section className="center-stack">
          <article className="panel video-panel">
            <div className="panel-title"><span>LIVE</span><h2>机载视觉</h2><em>{latest ? `${(-latest.position.z).toFixed(1)} m AGL` : "NO FEED"}</em></div>
            <div className="video-stage">
              {preview ? <img src={preview} alt="无人机相机实时关键帧" /> : <div className="camera-empty"><div className="reticle" /><span>等待观察关键帧</span></div>}
              {candidateBox && <div className="bbox" style={{left:`${candidateBox.x_min*100}%`,top:`${candidateBox.y_min*100}%`,width:`${(candidateBox.x_max-candidateBox.x_min)*100}%`,height:`${(candidateBox.y_max-candidateBox.y_min)*100}%`}}><span>VLM CANDIDATE</span></div>}
            </div>
            <div className="telemetry-strip">
              <span>X <b>{latest?.position.x.toFixed(1) ?? "—"}</b></span><span>Y <b>{latest?.position.y.toFixed(1) ?? "—"}</b></span><span>Z <b>{latest?.position.z.toFixed(1) ?? "—"}</b></span><span>V <b>{latest ? Math.hypot(latest.velocity.x, latest.velocity.y, latest.velocity.z).toFixed(1) : "—"}</b> m/s</span><span>{latest?.armed ? "ARMED" : "DISARMED"}</span>
            </div>
          </article>

          <article className="panel map-panel">
            <div className="panel-title"><span>NED</span><h2>搜索区与飞行轨迹</h2><em>{zone?.name}</em></div>
            <MapView zone={zone} telemetryPath={telemetryPath} target={run?.target_position} />
          </article>
        </section>

        <aside className="right-stack">
          <article className="panel run-panel">
            <div className="panel-title"><span>RUN</span><h2>执行控制</h2></div>
            <div className={`state-card state-${currentState.toLowerCase()}`}><small>CURRENT STATE</small><strong>{currentState}</strong><code>{active ? run.id.slice(0, 13) : run ? `历史 ${run.state} · ${run.id.slice(0, 13)}` : "no active run"}</code></div>
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

          <article className="panel events-panel">
            <div className="panel-title"><span>LOG</span><h2>实时日志</h2><button className="log-clear" onClick={() => setEvents([])}>清空</button></div>
            <div className="event-list">{logEvents.length ? logEvents.map((event) => <div className={`event level-${eventLevel(event)}`} key={`${event.sequence}-${event.topic}`}><div className="event-head"><time>{event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "—"}</time><span>{event.topic}</span><small>#{event.sequence}</small></div><p>{eventSummary(event)}</p></div>) : <div className="empty">等待系统日志</div>}</div>
          </article>
        </aside>
      </section>
    </main>
  );
}
