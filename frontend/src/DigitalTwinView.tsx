import { useEffect, useMemo, useRef } from "react";
import type { LidarPointCloudFrame, Quaternion, RunState, SafetyBounds, SemanticMap, Telemetry, Vec3, Zone } from "./types";

type Point3 = { x: number; y: number; z: number };
type ViewState = { azimuth: number; elevation: number; zoom: number };

type DigitalTwinViewProps = {
  zone?: Zone;
  safetyBounds?: SafetyBounds;
  telemetryPath: Telemetry[];
  lidarFrames: LidarPointCloudFrame[];
  semanticMap?: SemanticMap;
  cameraYawDegrees: number;
  frameOrigin?: Vec3;
  groundZ?: number;
  target?: Vec3;
  state: RunState | "IDLE" | "READY";
};

const identityQuaternion: Quaternion = { w: 1, x: 0, y: 0, z: 0 };
const defaultView: ViewState = { azimuth: -0.72, elevation: 0.58, zoom: 1 };

function zoneBounds(zone?: Zone): SafetyBounds {
  if (!zone?.polygon.points.length) return { x_min: -20, x_max: 20, y_min: -20, y_max: 20 };
  const xs = zone.polygon.points.map(([x]) => x);
  const ys = zone.polygon.points.map(([, y]) => y);
  return {
    x_min: Math.min(...xs),
    x_max: Math.max(...xs),
    y_min: Math.min(...ys),
    y_max: Math.max(...ys),
  };
}

export function quaternionYawDegrees(value?: Quaternion): number {
  if (!value) return 0;
  const norm = Math.hypot(value.w, value.x, value.y, value.z) || 1;
  const w = value.w / norm;
  const x = value.x / norm;
  const y = value.y / norm;
  const z = value.z / norm;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * 180 / Math.PI;
}

export function nedToTwinWorld(position: Vec3, groundZ: number, center: { x: number; y: number }): Point3 {
  return {
    x: position.y - center.y,
    y: groundZ - position.z,
    z: -(position.x - center.x),
  };
}

function normalizedQuaternion(value?: Quaternion, fallbackYawDegrees = 0): Quaternion {
  if (!value) {
    const yaw = fallbackYawDegrees * Math.PI / 180;
    return { w: Math.cos(yaw / 2), x: 0, y: 0, z: Math.sin(yaw / 2) };
  }
  const norm = Math.hypot(value.w, value.x, value.y, value.z) || 1;
  return { w: value.w / norm, x: value.x / norm, y: value.y / norm, z: value.z / norm };
}

function rotateNed(vector: Vec3, quaternion: Quaternion): Vec3 {
  const { w, x, y, z } = quaternion;
  const tx = 2 * (y * vector.z - z * vector.y);
  const ty = 2 * (z * vector.x - x * vector.z);
  const tz = 2 * (x * vector.y - y * vector.x);
  return {
    x: vector.x + w * tx + (y * tz - z * ty),
    y: vector.y + w * ty + (z * tx - x * tz),
    z: vector.z + w * tz + (x * ty - y * tx),
  };
}

function niceGridStep(span: number): number {
  if (span > 120) return 20;
  if (span > 60) return 10;
  if (span > 24) return 5;
  return 2;
}

function speed(telemetry?: Telemetry): number {
  return telemetry ? Math.hypot(telemetry.velocity.x, telemetry.velocity.y, telemetry.velocity.z) : 0;
}

export function DigitalTwinView({
  zone,
  safetyBounds,
  telemetryPath,
  lidarFrames,
  semanticMap,
  cameraYawDegrees,
  frameOrigin,
  groundZ,
  target,
  state,
}: DigitalTwinViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<ViewState>({ ...defaultView });
  const dragRef = useRef<{ x: number; y: number; pointerId: number }>();
  const latest = telemetryPath.at(-1);
  const effectiveGroundZ = groundZ ?? 0;
  const bounds = useMemo(() => {
    const relative = safetyBounds ?? zoneBounds(zone);
    return {
      x_min: relative.x_min + (frameOrigin?.x ?? 0),
      x_max: relative.x_max + (frameOrigin?.x ?? 0),
      y_min: relative.y_min + (frameOrigin?.y ?? 0),
      y_max: relative.y_max + (frameOrigin?.y ?? 0),
    };
  }, [frameOrigin?.x, frameOrigin?.y, safetyBounds, zone]);
  const altitude = latest ? Math.max(0, effectiveGroundZ - latest.position.z) : 0;
  const bodyYaw = quaternionYawDegrees(latest?.orientation);
  const liveDataRef = useRef({
    latest,
    telemetryPath,
    lidarFrames,
    semanticMap,
    cameraYawDegrees,
    target,
  });
  // React receives telemetry, image yaw, maps and LiDAR at different rates.
  // Keep those values fresh without rebuilding the canvas animation loop. A
  // canvas width/height assignment clears its pixels, so restarting this effect
  // for every packet used to produce a visible blank frame between redraws.
  liveDataRef.current = {
    latest,
    telemetryPath,
    lidarFrames,
    semanticMap,
    cameraYawDegrees,
    target,
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    let animationFrame = 0;
    let width = 1;
    let height = 1;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const center = {
      x: (bounds.x_min + bounds.x_max) / 2,
      y: (bounds.y_min + bounds.y_max) / 2,
    };
    const spanX = Math.max(8, bounds.x_max - bounds.x_min);
    const spanY = Math.max(8, bounds.y_max - bounds.y_min);
    const extent = Math.max(spanX, spanY);
    const fenceAltitude = Math.max(10, zone?.search_altitude_m ? zone.search_altitude_m * 1.8 : 12);

    const project = (point: Point3) => {
      const view = viewRef.current;
      const cosA = Math.cos(view.azimuth);
      const sinA = Math.sin(view.azimuth);
      const cosE = Math.cos(view.elevation);
      const sinE = Math.sin(view.elevation);
      const rotatedX = point.x * cosA - point.z * sinA;
      const rotatedZ = point.x * sinA + point.z * cosA;
      const rotatedY = point.y * cosE - rotatedZ * sinE;
      const depth = point.y * sinE + rotatedZ * cosE;
      const scale = Math.min(width / (extent * 1.56), height / (extent * 1.08)) * view.zoom;
      const perspective = Math.max(0.78, Math.min(1.2, 1 + depth / (extent * 7)));
      return {
        x: width * 0.5 + rotatedX * scale * perspective,
        y: height * 0.63 - rotatedY * scale * perspective,
        depth,
        scale: scale * perspective,
      };
    };

    const nedWorld = (point: Vec3) => nedToTwinWorld(point, effectiveGroundZ, center);
    const stroke3d = (
      first: Point3,
      second: Point3,
      color: string,
      lineWidth = 1,
      dash: number[] = [],
    ) => {
      const a = project(first);
      const b = project(second);
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.setLineDash(dash);
      context.strokeStyle = color;
      context.lineWidth = lineWidth;
      context.stroke();
      context.setLineDash([]);
    };

    const drawPolyline = (points: Point3[], color: string, lineWidth: number, dash: number[] = []) => {
      if (points.length < 2) return;
      context.beginPath();
      points.forEach((point, index) => {
        const projected = project(point);
        if (index === 0) context.moveTo(projected.x, projected.y);
        else context.lineTo(projected.x, projected.y);
      });
      context.setLineDash(dash);
      context.strokeStyle = color;
      context.lineWidth = lineWidth;
      context.stroke();
      context.setLineDash([]);
    };

    const drawGround = () => {
      const step = niceGridStep(extent);
      const minX = Math.floor(bounds.x_min / step) * step;
      const maxX = Math.ceil(bounds.x_max / step) * step;
      const minY = Math.floor(bounds.y_min / step) * step;
      const maxY = Math.ceil(bounds.y_max / step) * step;
      for (let north = minX; north <= maxX + 0.001; north += step) {
        const major = Math.abs(north) < 0.001;
        stroke3d(
          nedWorld({ x: north, y: minY, z: effectiveGroundZ }),
          nedWorld({ x: north, y: maxY, z: effectiveGroundZ }),
          major ? "rgba(107,231,255,.32)" : "rgba(66,115,143,.15)",
          major ? 1 : 0.65,
        );
      }
      for (let east = minY; east <= maxY + 0.001; east += step) {
        const major = Math.abs(east) < 0.001;
        stroke3d(
          nedWorld({ x: minX, y: east, z: effectiveGroundZ }),
          nedWorld({ x: maxX, y: east, z: effectiveGroundZ }),
          major ? "rgba(155,140,255,.32)" : "rgba(66,115,143,.15)",
          major ? 1 : 0.65,
        );
      }
    };

    const drawFence = () => {
      const ground = [
        { x: bounds.x_min, y: bounds.y_min, z: effectiveGroundZ },
        { x: bounds.x_min, y: bounds.y_max, z: effectiveGroundZ },
        { x: bounds.x_max, y: bounds.y_max, z: effectiveGroundZ },
        { x: bounds.x_max, y: bounds.y_min, z: effectiveGroundZ },
      ];
      const top = ground.map((point) => ({ ...point, z: effectiveGroundZ - fenceAltitude }));
      drawPolyline([...ground, ground[0]].map(nedWorld), "rgba(244,201,93,.72)", 1.15, [5, 4]);
      drawPolyline([...top, top[0]].map(nedWorld), "rgba(244,201,93,.22)", 0.75, [3, 5]);
      ground.forEach((point, index) => {
        stroke3d(nedWorld(point), nedWorld(top[index]), "rgba(244,201,93,.17)", 0.7, [3, 5]);
      });
    };

    const drawMapObjects = () => {
      const map = liveDataRef.current.semanticMap;
      const obstacles = map?.obstacles ?? [];
      const stride = Math.max(1, Math.ceil(obstacles.length / 550));
      for (let index = 0; index < obstacles.length; index += stride) {
        const obstacle = obstacles[index];
        const point = project(nedWorld({ x: obstacle.x, y: obstacle.y, z: effectiveGroundZ - 0.18 }));
        context.fillStyle = "rgba(255,101,111,.52)";
        context.fillRect(point.x - 1, point.y - 1, 2, 2);
      }
      const objects = (map?.nodes ?? []).filter((node) => node.kind === "object").slice(-12);
      objects.forEach((node) => {
        const base = nedWorld({ x: node.position.x, y: node.position.y, z: effectiveGroundZ });
        const tip = nedWorld(node.position);
        stroke3d(base, tip, "rgba(155,140,255,.58)", 1, [2, 3]);
        const label = project(tip);
        context.fillStyle = "rgba(6,10,20,.88)";
        context.fillRect(label.x + 5, label.y - 14, Math.min(86, (node.label?.length ?? 4) * 10 + 10), 17);
        context.fillStyle = "#cfc7ff";
        context.font = "8px 'Microsoft YaHei UI', sans-serif";
        context.fillText(node.label ?? "语义目标", label.x + 9, label.y - 3);
      });
    };

    const drawPointCloud = () => {
      const frames = liveDataRef.current.lidarFrames;
      frames.forEach((frame, frameIndex) => {
        const recency = (frameIndex + 1) / Math.max(1, frames.length);
        const alpha = 0.14 + recency * 0.68;
        const pointSize = 0.7 + recency * 1.15;
        frame.points.forEach(([x, y, z]) => {
          if (![x, y, z].every(Number.isFinite)) return;
          const heightAboveGround = effectiveGroundZ - z;
          const point = project(nedWorld({ x, y, z }));
          let color: string;
          if (heightAboveGround < 0.6) color = `rgba(49,112,255,${alpha * 0.62})`;
          else if (heightAboveGround < 3) color = `rgba(74,225,255,${alpha})`;
          else if (heightAboveGround < 8) color = `rgba(155,140,255,${alpha})`;
          else color = `rgba(244,201,93,${alpha * 0.9})`;
          context.fillStyle = color;
          context.fillRect(
            point.x - pointSize / 2,
            point.y - pointSize / 2,
            pointSize,
            pointSize,
          );
        });
      });
    };

    const drawTrack = () => {
      const path = liveDataRef.current.telemetryPath;
      const sampleStride = Math.max(1, Math.ceil(path.length / 500));
      const samples = path.filter((_, index) => index % sampleStride === 0);
      drawPolyline(
        samples.map((item) => nedWorld({ ...item.position, z: effectiveGroundZ })),
        "rgba(107,231,255,.18)",
        0.8,
        [3, 4],
      );
      drawPolyline(samples.map((item) => nedWorld(item.position)), "rgba(107,231,255,.9)", 1.65);
    };

    const drawTarget = () => {
      const currentTarget = liveDataRef.current.target;
      if (!currentTarget) return;
      const base = nedWorld({ x: currentTarget.x, y: currentTarget.y, z: effectiveGroundZ });
      const tip = nedWorld(currentTarget);
      stroke3d(base, tip, "rgba(244,201,93,.72)", 1.2, [3, 3]);
      const point = project(tip);
      context.beginPath();
      context.arc(point.x, point.y, 7, 0, Math.PI * 2);
      context.strokeStyle = "#f4c95d";
      context.lineWidth = 1.2;
      context.stroke();
      context.beginPath();
      context.arc(point.x, point.y, 2.2, 0, Math.PI * 2);
      context.fillStyle = "#f4c95d";
      context.fill();
    };

    const drawDrone = (time: number) => {
      const telemetry = liveDataRef.current.latest;
      const position = telemetry?.position ?? { x: 0, y: 0, z: effectiveGroundZ };
      const liveCameraYawDegrees = liveDataRef.current.cameraYawDegrees;
      const quaternion = normalizedQuaternion(telemetry?.orientation, liveCameraYawDegrees);
      const size = Math.max(0.75, extent * 0.018);
      const toDroneWorld = (local: Vec3) => {
        const rotated = rotateNed(local, quaternion);
        return nedWorld({
          x: position.x + rotated.x,
          y: position.y + rotated.y,
          z: position.z + rotated.z,
        });
      };
      const centerPoint = nedWorld(position);
      const projectedCenter = project(centerPoint);
      const groundPoint = nedWorld({ ...position, z: effectiveGroundZ });

      stroke3d(centerPoint, groundPoint, "rgba(220,244,255,.2)", 0.8, [2, 4]);
      const shadow = project(groundPoint);
      context.beginPath();
      context.ellipse(shadow.x, shadow.y, 8, 3, 0, 0, Math.PI * 2);
      context.fillStyle = "rgba(0,0,0,.36)";
      context.fill();

      const cameraYaw = liveCameraYawDegrees * Math.PI / 180;
      const coneRange = Math.max(5, Math.min(12, extent * 0.2));
      const coneHalf = 22 * Math.PI / 180;
      const conePoints = [
        centerPoint,
        nedWorld({
          x: position.x + Math.cos(cameraYaw - coneHalf) * coneRange,
          y: position.y + Math.sin(cameraYaw - coneHalf) * coneRange,
          z: position.z,
        }),
        nedWorld({
          x: position.x + Math.cos(cameraYaw + coneHalf) * coneRange,
          y: position.y + Math.sin(cameraYaw + coneHalf) * coneRange,
          z: position.z,
        }),
      ].map(project);
      context.beginPath();
      context.moveTo(conePoints[0].x, conePoints[0].y);
      context.lineTo(conePoints[1].x, conePoints[1].y);
      context.lineTo(conePoints[2].x, conePoints[2].y);
      context.closePath();
      context.fillStyle = "rgba(107,231,255,.065)";
      context.fill();
      context.strokeStyle = "rgba(107,231,255,.46)";
      context.lineWidth = 0.85;
      context.stroke();

      const rotorCenters: Vec3[] = [
        { x: size, y: size, z: 0 },
        { x: size, y: -size, z: 0 },
        { x: -size, y: size, z: 0 },
        { x: -size, y: -size, z: 0 },
      ];
      rotorCenters.forEach((rotor, rotorIndex) => {
        stroke3d(centerPoint, toDroneWorld(rotor), "rgba(220,248,255,.9)", 1.55);
        const ring: Point3[] = [];
        for (let index = 0; index <= 18; index += 1) {
          const angle = index / 18 * Math.PI * 2;
          ring.push(toDroneWorld({
            x: rotor.x + Math.cos(angle) * size * 0.38,
            y: rotor.y + Math.sin(angle) * size * 0.38,
            z: 0,
          }));
        }
        drawPolyline(ring, "rgba(107,231,255,.72)", 1);
        const bladeAngle = (reducedMotion ? 0 : time * 0.018) + rotorIndex * Math.PI / 2;
        const blade = { x: Math.cos(bladeAngle) * size * 0.34, y: Math.sin(bladeAngle) * size * 0.34 };
        stroke3d(
          toDroneWorld({ x: rotor.x - blade.x, y: rotor.y - blade.y, z: 0 }),
          toDroneWorld({ x: rotor.x + blade.x, y: rotor.y + blade.y, z: 0 }),
          telemetry?.armed ? "rgba(238,253,255,.92)" : "rgba(117,148,170,.55)",
          1.1,
        );
      });

      const body = [
        { x: size * 0.7, y: 0, z: 0 },
        { x: 0, y: size * 0.52, z: 0 },
        { x: -size * 0.58, y: 0, z: 0 },
        { x: 0, y: -size * 0.52, z: 0 },
        { x: size * 0.7, y: 0, z: 0 },
      ].map(toDroneWorld);
      const bodyScreen = body.map(project);
      context.beginPath();
      bodyScreen.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
      context.closePath();
      context.fillStyle = telemetry?.collision ? "rgba(255,92,105,.82)" : "rgba(13,45,64,.94)";
      context.fill();
      context.strokeStyle = "#e9fbff";
      context.lineWidth = 1.45;
      context.stroke();

      const nose = project(toDroneWorld({ x: size * 1.16, y: 0, z: 0 }));
      stroke3d(centerPoint, toDroneWorld({ x: size * 1.16, y: 0, z: 0 }), "#9b8cff", 1.8);
      context.beginPath();
      context.arc(nose.x, nose.y, 2.3, 0, Math.PI * 2);
      context.fillStyle = "#9b8cff";
      context.fill();

      if (telemetry) {
        const velocityScale = 1.5;
        const velocityTip = nedWorld({
          x: position.x + telemetry.velocity.x * velocityScale,
          y: position.y + telemetry.velocity.y * velocityScale,
          z: position.z + telemetry.velocity.z * velocityScale,
        });
        stroke3d(centerPoint, velocityTip, "rgba(244,201,93,.9)", 1.3);
      }

      context.beginPath();
      context.arc(projectedCenter.x, projectedCenter.y, 18, 0, Math.PI * 2);
      context.strokeStyle = telemetry?.armed ? "rgba(107,231,255,.18)" : "rgba(107,231,255,.08)";
      context.lineWidth = 1;
      context.stroke();
    };

    const drawLabels = () => {
      const north = project(nedWorld({ x: bounds.x_max, y: center.y, z: effectiveGroundZ }));
      const east = project(nedWorld({ x: center.x, y: bounds.y_max, z: effectiveGroundZ }));
      context.font = "8px ui-monospace, monospace";
      context.fillStyle = "rgba(107,231,255,.7)";
      context.fillText("N+", north.x + 5, north.y - 3);
      context.fillStyle = "rgba(155,140,255,.72)";
      context.fillText("E+", east.x + 5, east.y - 3);
    };

    const draw = (time: number) => {
      context.clearRect(0, 0, width, height);
      const gradient = context.createRadialGradient(width * 0.5, height * 0.5, 12, width * 0.5, height * 0.5, width * 0.7);
      gradient.addColorStop(0, "rgba(14,32,49,.72)");
      gradient.addColorStop(1, "rgba(2,6,13,.98)");
      context.fillStyle = gradient;
      context.fillRect(0, 0, width, height);
      drawGround();
      drawFence();
      drawMapObjects();
      drawPointCloud();
      drawTrack();
      drawTarget();
      drawDrone(time);
      drawLabels();
      animationFrame = requestAnimationFrame(draw);
    };
    animationFrame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
    };
  }, [bounds, effectiveGroundZ, zone?.search_altitude_m]);

  function resetView() {
    viewRef.current = { ...defaultView };
  }

  return <div className="digital-twin" data-state={state.toLowerCase()}>
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="无人机三维实时数字孪生视图"
      onPointerDown={(event) => {
        dragRef.current = { x: event.clientX, y: event.clientY, pointerId: event.pointerId };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const drag = dragRef.current;
        if (!drag || drag.pointerId !== event.pointerId) return;
        const dx = event.clientX - drag.x;
        const dy = event.clientY - drag.y;
        viewRef.current.azimuth += dx * 0.008;
        viewRef.current.elevation = Math.max(0.18, Math.min(1.22, viewRef.current.elevation - dy * 0.006));
        dragRef.current = { ...drag, x: event.clientX, y: event.clientY };
      }}
      onPointerUp={(event) => {
        dragRef.current = undefined;
        event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onWheel={(event) => {
        event.preventDefault();
        viewRef.current.zoom = Math.max(0.62, Math.min(2.2, viewRef.current.zoom * Math.exp(-event.deltaY * 0.001)));
      }}
    />
    <div className="twin-scanline" aria-hidden="true" />
    <div className="twin-hud twin-hud-left">
      <small>NED POSITION</small>
      <span><b>X</b>{latest?.position.x.toFixed(1) ?? "—"}</span>
      <span><b>Y</b>{latest?.position.y.toFixed(1) ?? "—"}</span>
      <span><b>ALT</b>{latest ? altitude.toFixed(1) : "—"}<i>m</i></span>
    </div>
    <div className="twin-hud twin-hud-right">
      <small>FLIGHT VECTOR</small>
      <span><b>SPD</b>{latest ? speed(latest).toFixed(1) : "—"}<i>m/s</i></span>
      <span><b>BODY</b>{latest ? bodyYaw.toFixed(0) : "—"}<i>°</i></span>
      <span><b>CAM</b>{cameraYawDegrees.toFixed(0)}<i>°</i></span>
    </div>
    <div className="twin-status">
      <span className={latest ? "live" : ""}><i />{latest ? "TWIN SYNC" : "WAITING TELEMETRY"}</span>
      <b>{state}</b>
    </div>
    <div className="twin-legend" aria-hidden="true">
      <span><i className="legend-flight" />实际轨迹</span>
      <span><i className="legend-cloud" />3D 点云</span>
      <span><i className="legend-camera" />相机视锥</span>
      <span><i className="legend-fence" />安全围栏</span>
    </div>
    <div className={`twin-cloud-status ${lidarFrames.length ? "live" : ""}`}>
      <span>LiDAR 3D</span>
      <b>{lidarFrames.at(-1)?.sampled_point_count ?? 0}</b>
      <small>/ {lidarFrames.at(-1)?.point_count ?? 0} PTS</small>
      <i aria-hidden="true" />
    </div>
    <button className="twin-reset" type="button" onClick={resetView} title="恢复默认三维视角">RESET VIEW</button>
    <p className="twin-help">拖拽旋转 · 滚轮缩放</p>
  </div>;
}
