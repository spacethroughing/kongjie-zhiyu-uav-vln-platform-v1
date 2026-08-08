import { useRef, type PointerEvent as ReactPointerEvent } from "react";
import type { SafetyBounds, Telemetry, Zone } from "./types";

type Props = {
  zone?: Zone;
  hardBounds?: SafetyBounds;
  safetyBounds?: SafetyBounds;
  telemetryPath: Telemetry[];
  target?: { x: number; y: number };
  onSafetyBoundsChange?: (bounds: SafetyBounds) => void;
};

function boundsFromZone(zone: Zone): SafetyBounds {
  const xs = zone.polygon.points.map(([x]) => x);
  const ys = zone.polygon.points.map(([, y]) => y);
  return {
    x_min: Math.min(...xs),
    x_max: Math.max(...xs),
    y_min: Math.min(...ys),
    y_max: Math.max(...ys),
  };
}

function canDraw(bounds: SafetyBounds | undefined): bounds is SafetyBounds {
  return Boolean(
    bounds &&
    Object.values(bounds).every(Number.isFinite) &&
    bounds.x_min < bounds.x_max &&
    bounds.y_min < bounds.y_max,
  );
}

export function MapView({
  zone,
  hardBounds,
  safetyBounds,
  telemetryPath,
  target,
  onSafetyBoundsChange,
}: Props) {
  const dragStart = useRef<[number, number]>();
  if (!zone) return <div className="empty">选择搜索区后显示 NED 轨迹</div>;

  const zoneBounds = boundsFromZone(zone);
  const limit = hardBounds ?? zoneBounds;
  const points = zone.polygon.points;
  const pad = 3;
  const minX = Math.min(zoneBounds.x_min, limit.x_min) - pad;
  const maxX = Math.max(zoneBounds.x_max, limit.x_max) + pad;
  const minY = Math.min(zoneBounds.y_min, limit.y_min) - pad;
  const maxY = Math.max(zoneBounds.y_max, limit.y_max) + pad;
  const map = ([x, y]: [number, number]) => [
    ((x - minX) / Math.max(1, maxX - minX)) * 100,
    100 - ((y - minY) / Math.max(1, maxY - minY)) * 100,
  ];
  const rectangle = (bounds: SafetyBounds) => {
    const [left, top] = map([bounds.x_min, bounds.y_max]);
    const [right, bottom] = map([bounds.x_max, bounds.y_min]);
    return { x: left, y: top, width: right - left, height: bottom - top };
  };
  const hardRectangle = rectangle(limit);
  const manualRectangle = canDraw(safetyBounds) ? rectangle(safetyBounds) : undefined;
  const polygon = points.map(map).map((point) => point.join(",")).join(" ");
  const trace = telemetryPath.map((item) => map([item.position.x, item.position.y]).join(",")).join(" ");
  const latest = telemetryPath.at(-1);
  const drone = latest ? map([latest.position.x, latest.position.y]) : undefined;
  const targetPoint = target ? map([target.x, target.y]) : undefined;
  const home = map([0, 0]);

  const eventPoint = (event: ReactPointerEvent<SVGSVGElement>): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    const svgX = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 100;
    const svgY = ((event.clientY - rect.top) / Math.max(1, rect.height)) * 100;
    const x = minX + (svgX / 100) * (maxX - minX);
    const y = minY + ((100 - svgY) / 100) * (maxY - minY);
    return [
      Math.min(limit.x_max, Math.max(limit.x_min, x)),
      Math.min(limit.y_max, Math.max(limit.y_min, y)),
    ];
  };

  const updateDrag = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!dragStart.current || !onSafetyBoundsChange) return;
    const [startX, startY] = dragStart.current;
    const [endX, endY] = eventPoint(event);
    onSafetyBoundsChange({
      x_min: Math.min(startX, endX),
      x_max: Math.max(startX, endX),
      y_min: Math.min(startY, endY),
      y_max: Math.max(startY, endY),
    });
  };

  return (
    <div className="map-wrap">
      <svg
        className={`map ${onSafetyBoundsChange ? "map-selectable" : ""}`}
        viewBox="0 0 100 100"
        role="img"
        aria-label="无人机 NED 轨迹图"
        onPointerDown={(event) => {
          if (!onSafetyBoundsChange || event.button > 0) return;
          event.preventDefault();
          dragStart.current = eventPoint(event);
          event.currentTarget.setPointerCapture?.(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!dragStart.current) return;
          event.preventDefault();
          updateDrag(event);
        }}
        onPointerUp={(event) => {
          if (!dragStart.current) return;
          updateDrag(event);
          dragStart.current = undefined;
          event.currentTarget.releasePointerCapture?.(event.pointerId);
        }}
        onPointerCancel={() => { dragStart.current = undefined; }}
      >
        <defs>
          <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(103,235,187,.08)" strokeWidth=".5" />
          </pattern>
        </defs>
        <rect width="100" height="100" fill="url(#grid)" />
        <rect {...hardRectangle} className="hard-limit" data-testid="hard-limit" />
        <polygon points={polygon} className="zone" data-testid="default-zone" />
        {manualRectangle && <rect {...manualRectangle} className="manual-bounds" data-testid="manual-bounds" />}
        {trace && <polyline points={trace} className="trace" />}
        <g className="home" transform={`translate(${home[0]} ${home[1]})`} aria-label="NED 起点">
          <circle r="1.5" />
          <path d="M -2.5 0 H 2.5 M 0 -2.5 V 2.5" />
        </g>
        {targetPoint && <circle cx={targetPoint[0]} cy={targetPoint[1]} r="2.6" className="target" />}
        {drone && <g transform={`translate(${drone[0]} ${drone[1]})`}><path d="M 0 -3 L 2 2 L 0 1 L -2 2 Z" className="drone" /></g>}
        <text x="3" y="7" className="map-label">N ↑</text>
        <text x="91" y="96" className="map-label">E →</text>
      </svg>
      <div className="map-legend" aria-hidden="true">
        <span><i className="legend-hard" />硬限制</span>
        <span><i className="legend-zone" />区域预设</span>
        {manualRectangle && <span><i className="legend-manual" />任务范围</span>}
      </div>
      {onSafetyBoundsChange && <div className="map-select-hint">在地图上拖拽可重新框选任务范围</div>}
    </div>
  );
}
