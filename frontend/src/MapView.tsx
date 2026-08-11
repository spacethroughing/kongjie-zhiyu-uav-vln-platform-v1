import { useRef, type PointerEvent as ReactPointerEvent } from "react";
import type { SafetyBounds, SemanticMap, Telemetry, Vec3, Zone } from "./types";

type Props = {
  zone?: Zone;
  hardBounds?: SafetyBounds;
  safetyBounds?: SafetyBounds;
  telemetryPath: Telemetry[];
  semanticMap?: SemanticMap;
  cameraYawDegrees?: number;
  frameOrigin?: Vec3;
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
  semanticMap,
  cameraYawDegrees,
  frameOrigin,
  target,
  onSafetyBoundsChange,
}: Props) {
  const dragStart = useRef<[number, number]>();
  const dragViewport = useRef<SafetyBounds>();
  if (!zone) return <div className="empty">选择搜索区后显示 NED 轨迹</div>;

  const originX = frameOrigin?.x ?? 0;
  const originY = frameOrigin?.y ?? 0;
  const shiftBounds = (bounds: SafetyBounds): SafetyBounds => ({
    x_min: bounds.x_min + originX,
    x_max: bounds.x_max + originX,
    y_min: bounds.y_min + originY,
    y_max: bounds.y_max + originY,
  });
  const relativeZoneBounds = boundsFromZone(zone);
  const relativeLimit = hardBounds ?? relativeZoneBounds;
  const zoneBounds = shiftBounds(relativeZoneBounds);
  const limit = shiftBounds(relativeLimit);
  const displayedSafetyBounds = canDraw(safetyBounds) ? shiftBounds(safetyBounds) : undefined;
  const points = zone.polygon.points.map(([x, y]) => [x + originX, y + originY] as [number, number]);
  // The scene hard limit is a safety constraint, not a useful camera viewport.
  // Fit the map to the mission selection so a large configurable hard limit
  // (for example +/-999 m) cannot collapse the active area into a tiny dot.
  // Keep the viewport fixed while dragging to avoid feedback between pointer
  // coordinates and the continuously updated selection.
  const viewportBounds = dragViewport.current ?? displayedSafetyBounds ?? zoneBounds;
  const viewportSpanX = viewportBounds.x_max - viewportBounds.x_min;
  const viewportSpanY = viewportBounds.y_max - viewportBounds.y_min;
  const padX = Math.max(0.5, viewportSpanX * 0.06);
  const padY = Math.max(0.5, viewportSpanY * 0.06);
  const minX = viewportBounds.x_min - padX;
  const maxX = viewportBounds.x_max + padX;
  const minY = viewportBounds.y_min - padY;
  const maxY = viewportBounds.y_max + padY;
  // NED convention on screen: North (+X) is up and East (+Y) is right.
  const map = ([x, y]: [number, number]) => [
    ((y - minY) / Math.max(1, maxY - minY)) * 100,
    100 - ((x - minX) / Math.max(1, maxX - minX)) * 100,
  ];
  const rectangle = (bounds: SafetyBounds) => {
    const corners = [
      map([bounds.x_min, bounds.y_min]),
      map([bounds.x_min, bounds.y_max]),
      map([bounds.x_max, bounds.y_min]),
      map([bounds.x_max, bounds.y_max]),
    ];
    const xs = corners.map(([x]) => x);
    const ys = corners.map(([, y]) => y);
    const left = Math.min(...xs);
    const right = Math.max(...xs);
    const top = Math.min(...ys);
    const bottom = Math.max(...ys);
    return { x: left, y: top, width: right - left, height: bottom - top };
  };
  const hardRectangle = rectangle(limit);
  const manualRectangle = displayedSafetyBounds ? rectangle(displayedSafetyBounds) : undefined;
  const polygon = points.map(map).map((point) => point.join(",")).join(" ");
  const trace = telemetryPath.map((item) => map([item.position.x, item.position.y]).join(",")).join(" ");
  const latest = telemetryPath.at(-1);
  const drone = latest ? map([latest.position.x, latest.position.y]) : undefined;
  const targetPoint = target ? map([target.x, target.y]) : undefined;
  const home = map([originX, originY]);
  const topologyNodes = semanticMap?.nodes ?? [];
  const topologyNodeById = new Map(topologyNodes.map((node) => [node.id, node]));
  const placeNodes = topologyNodes.filter((node) => node.kind === "place");
  const objectNodes = topologyNodes.filter((node) => node.kind === "object").slice(-30);
  const topologyEdges = (semanticMap?.edges ?? []).flatMap((edge) => {
    const source = topologyNodeById.get(edge.source);
    const destination = topologyNodeById.get(edge.target);
    if (!source || !destination) return [];
    return [{ ...edge, sourcePoint: map([source.position.x, source.position.y]), destinationPoint: map([destination.position.x, destination.position.y]) }];
  });

  const eventPoint = (event: ReactPointerEvent<SVGSVGElement>): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    const svgX = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 100;
    const svgY = ((event.clientY - rect.top) / Math.max(1, rect.height)) * 100;
    const worldX = minX + ((100 - svgY) / 100) * (maxX - minX);
    const worldY = minY + (svgX / 100) * (maxY - minY);
    const x = worldX - originX;
    const y = worldY - originY;
    return [
      Math.min(relativeLimit.x_max, Math.max(relativeLimit.x_min, x)),
      Math.min(relativeLimit.y_max, Math.max(relativeLimit.y_min, y)),
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
        id="semantic-topology-map"
        className={`map ${onSafetyBoundsChange ? "map-selectable" : ""}`}
        viewBox="0 0 100 100"
        data-testid="semantic-map"
        data-viewport-x-min={viewportBounds.x_min}
        data-viewport-x-max={viewportBounds.x_max}
        data-viewport-y-min={viewportBounds.y_min}
        data-viewport-y-max={viewportBounds.y_max}
        data-frame-origin-x={originX}
        data-frame-origin-y={originY}
        role="img"
        aria-label="无人机 NED 轨迹图"
        onPointerDown={(event) => {
          if (!onSafetyBoundsChange || event.button > 0) return;
          event.preventDefault();
          dragViewport.current = { ...viewportBounds };
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
          dragViewport.current = undefined;
          event.currentTarget.releasePointerCapture?.(event.pointerId);
        }}
        onPointerCancel={() => {
          dragStart.current = undefined;
          dragViewport.current = undefined;
        }}
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
        <g aria-label="LiDAR 已探索范围">
          {(semanticMap?.explored ?? []).slice(0, 2500).map((cell) => {
            const [cx, cy] = map([cell.x, cell.y]);
            const size = semanticMap?.coverage_cell_size_m ?? 2;
            const width = (size / Math.max(1, maxY - minY)) * 100;
            const height = (size / Math.max(1, maxX - minX)) * 100;
            return <rect
              key={`${Math.round(cell.x * 10)}-${Math.round(cell.y * 10)}`}
              x={cx - width / 2}
              y={cy - height / 2}
              width={width}
              height={height}
              opacity={Math.min(0.48, 0.16 + Math.log2(cell.scans + 1) * 0.06)}
              className="explored-cell"
            />;
          })}
        </g>
        {(semanticMap?.obstacles ?? []).slice(0, 1000).map((point, index) => {
          const [cx, cy] = map([point.x, point.y]);
          return <circle key={`${Math.round(point.x * 10)}-${Math.round(point.y * 10)}-${index}`} cx={cx} cy={cy} r={Math.min(0.75, 0.22 + Math.log2(point.hits + 1) * 0.12)} opacity={Math.min(0.88, 0.2 + point.hits * 0.08)} className="occupancy-point" />;
        })}
        {topologyEdges.map((edge) => <line
          key={`${edge.source}-${edge.target}`}
          x1={edge.sourcePoint[0]}
          y1={edge.sourcePoint[1]}
          x2={edge.destinationPoint[0]}
          y2={edge.destinationPoint[1]}
          className={`topology-edge topology-${edge.kind}`}
        />)}
        {placeNodes.map((node) => {
          const [cx, cy] = map([node.position.x, node.position.y]);
          return <circle key={node.id} cx={cx} cy={cy} r="0.75" className="place-node" />;
        })}
        {trace && <polyline points={trace} className="trace" />}
        <g className="home" data-testid="home-marker" transform={`translate(${home[0]} ${home[1]})`} aria-label="任务起飞点">
          <circle r="1.5" />
          <path d="M -2.5 0 H 2.5 M 0 -2.5 V 2.5" />
        </g>
        {targetPoint && <circle cx={targetPoint[0]} cy={targetPoint[1]} r="2.6" className="target" />}
        {objectNodes.map((node) => {
          const [cx, cy] = map([node.position.x, node.position.y]);
          const label = node.label ?? "物体";
          const displayLabel = `${label} · ${node.position.x.toFixed(1)},${node.position.y.toFixed(1)}`;
          return <g key={node.id} className="semantic-node" aria-label={`语义物体 ${displayLabel}`}>
            <circle cx={cx} cy={cy} r="1.8" />
            <text x={cx + 2.4} y={cy - 1.5}>{displayLabel}</text>
          </g>;
        })}
        {drone && <g data-testid="drone-marker" aria-label={`相机航向 ${Math.round(cameraYawDegrees ?? 0)} 度`} transform={`translate(${drone[0]} ${drone[1]}) rotate(${cameraYawDegrees ?? 0})`}><path d="M 0 -3 L 2 2 L 0 1 L -2 2 Z" className="drone" /></g>}
        <text x="3" y="7" className="map-label">N ↑</text>
        <text x="91" y="96" className="map-label">E →</text>
      </svg>
      <div className="map-viewport-status" aria-label="地图自适应视野">
        <b>AUTO FIT</b>
        <small>HOME {originX.toFixed(1)}, {originY.toFixed(1)}</small>
        <span>X {viewportSpanX.toFixed(1)} m · Y {viewportSpanY.toFixed(1)} m</span>
      </div>
      <div className="map-legend" aria-hidden="true">
        <span><i className="legend-hard" />硬限制</span>
        <span><i className="legend-zone" />区域预设</span>
        {manualRectangle && <span><i className="legend-manual" />任务范围</span>}
        <span><i className="legend-explored" />LiDAR 已探索</span>
        <span><i className="legend-obstacle" />LiDAR 占据</span>
        <span><i className="legend-semantic" />VLM 物体</span>
      </div>
      {semanticMap && <div className="map-stats" aria-label="实时地图统计">
        已探索 {semanticMap.stats.explored_cells} · 占据 {semanticMap.stats.occupancy_cells} · 拓扑 {semanticMap.stats.place_nodes} · 物体 {semanticMap.stats.semantic_objects}
      </div>}
      {onSafetyBoundsChange && <div className="map-select-hint">在地图上拖拽可重新框选任务范围</div>}
    </div>
  );
}
