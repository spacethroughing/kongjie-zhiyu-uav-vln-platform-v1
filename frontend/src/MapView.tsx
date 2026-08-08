import type { Telemetry, Zone } from "./types";

type Props = { zone?: Zone; telemetryPath: Telemetry[]; target?: { x: number; y: number } };

export function MapView({ zone, telemetryPath, target }: Props) {
  if (!zone) return <div className="empty">选择搜索区后显示 NED 轨迹</div>;
  const points = zone.polygon.points;
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const pad = 3;
  const minX = Math.min(...xs) - pad;
  const maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad;
  const maxY = Math.max(...ys) + pad;
  const map = ([x, y]: [number, number]) => [
    ((x - minX) / Math.max(1, maxX - minX)) * 100,
    100 - ((y - minY) / Math.max(1, maxY - minY)) * 100,
  ];
  const polygon = points.map(map).map((point) => point.join(",")).join(" ");
  const trace = telemetryPath.map((item) => map([item.position.x, item.position.y]).join(",")).join(" ");
  const latest = telemetryPath.at(-1);
  const drone = latest ? map([latest.position.x, latest.position.y]) : undefined;
  const targetPoint = target ? map([target.x, target.y]) : undefined;
  return (
    <svg className="map" viewBox="0 0 100 100" role="img" aria-label="无人机 NED 轨迹图">
      <defs>
        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(103,235,187,.08)" strokeWidth=".5" />
        </pattern>
      </defs>
      <rect width="100" height="100" fill="url(#grid)" />
      <polygon points={polygon} className="zone" />
      {trace && <polyline points={trace} className="trace" />}
      {targetPoint && <circle cx={targetPoint[0]} cy={targetPoint[1]} r="2.6" className="target" />}
      {drone && <g transform={`translate(${drone[0]} ${drone[1]})`}><path d="M 0 -3 L 2 2 L 0 1 L -2 2 Z" className="drone" /></g>}
      <text x="3" y="7" className="map-label">N ↑</text>
    </svg>
  );
}

