import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MapView } from "./MapView";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const zone = {
  id: "z",
  name: "Zone",
  polygon: { points: [[0, 0], [10, 0], [10, 10], [0, 10]] as [number, number][] },
  search_altitude_m: 5,
  lane_spacing_m: 2,
};

describe("MapView", () => {
  it("renders the configured geofence", () => {
    render(<MapView zone={zone} telemetryPath={[]} />);
    expect(screen.getByRole("img", { name: "无人机 NED 轨迹图" })).toBeInTheDocument();
  });

  it("renders LiDAR occupancy and VLM semantic topology", () => {
    render(
      <MapView
        zone={zone}
        telemetryPath={[]}
        semanticMap={{
          revision: 3,
          obstacles: [{ x: 3, y: 4, z: -5, hits: 2 }],
          explored: [{ x: 2, y: 2, scans: 3 }],
          coverage_cell_size_m: 2,
          nodes: [
            { id: "place-1", kind: "place", position: { x: 1, y: 1, z: -5 } },
            { id: "object-1", kind: "object", position: { x: 4, y: 4, z: -3 }, label: "橙色球体", confidence: 0.9 },
          ],
          edges: [{ source: "place-1", target: "object-1", kind: "observed" }],
          stats: { occupancy_cells: 1, explored_cells: 1, place_nodes: 1, semantic_objects: 1 },
        }}
      />,
    );
    expect(screen.getByLabelText("语义物体 橙色球体 · 4.0,4.0")).toBeInTheDocument();
    expect(screen.getByLabelText("LiDAR 已探索范围")).toBeInTheDocument();
    expect(screen.getByLabelText("实时地图统计")).toHaveTextContent("物体 1");
  });

  it("rotates the drone marker with camera yaw in the NED map", () => {
    render(
      <MapView
        zone={zone}
        cameraYawDegrees={90}
        telemetryPath={[{
          timestamp: "2026-01-01T00:00:00Z",
          position: { x: 5, y: 2, z: -5 },
          velocity: { x: 0, y: 0, z: 0 },
          armed: true,
          landed: false,
          collision: false,
        }]}
      />,
    );
    expect(screen.getByTestId("drone-marker")).toHaveAttribute(
      "transform",
      expect.stringContaining("rotate(90)"),
    );
    expect(screen.getByLabelText("相机航向 90 度")).toBeInTheDocument();
  });

  it("renders the scene hard limit and mission bounds separately", () => {
    render(
      <MapView
        zone={zone}
        hardBounds={{ x_min: -20, x_max: 20, y_min: -20, y_max: 20 }}
        safetyBounds={{ x_min: -5, x_max: 12, y_min: -6, y_max: 13 }}
        telemetryPath={[]}
      />,
    );
    expect(screen.getByTestId("hard-limit")).toBeInTheDocument();
    expect(screen.getByTestId("default-zone")).toBeInTheDocument();
    expect(screen.getByTestId("manual-bounds")).toBeInTheDocument();
  });

  it("auto-fits the viewport to the selected mission bounds instead of the hard limit", () => {
    const view = render(
      <MapView
        zone={zone}
        hardBounds={{ x_min: -999, x_max: 999, y_min: -999, y_max: 999 }}
        safetyBounds={{ x_min: -5, x_max: 12, y_min: -6, y_max: 13 }}
        telemetryPath={[]}
      />,
    );
    const map = screen.getByTestId("semantic-map");
    expect(map).toHaveAttribute("data-viewport-x-min", "-5");
    expect(map).toHaveAttribute("data-viewport-x-max", "12");
    expect(map).toHaveAttribute("data-viewport-y-min", "-6");
    expect(map).toHaveAttribute("data-viewport-y-max", "13");
    expect(Number(screen.getByTestId("manual-bounds").getAttribute("width"))).toBeGreaterThan(80);
    expect(screen.getByLabelText("地图自适应视野")).toHaveTextContent("X 17.0 m · Y 19.0 m");

    view.rerender(
      <MapView
        zone={zone}
        hardBounds={{ x_min: -999, x_max: 999, y_min: -999, y_max: 999 }}
        safetyBounds={{ x_min: -80, x_max: 80, y_min: -40, y_max: 40 }}
        telemetryPath={[]}
      />,
    );
    expect(map).toHaveAttribute("data-viewport-x-min", "-80");
    expect(map).toHaveAttribute("data-viewport-x-max", "80");
    expect(screen.getByLabelText("地图自适应视野")).toHaveTextContent("X 160.0 m · Y 80.0 m");
  });

  it("anchors relative mission bounds to the persisted AirSim NED run home", () => {
    render(
      <MapView
        zone={zone}
        frameOrigin={{ x: 100, y: 50, z: 0 }}
        hardBounds={{ x_min: -20, x_max: 20, y_min: -20, y_max: 20 }}
        safetyBounds={{ x_min: -5, x_max: 12, y_min: -6, y_max: 13 }}
        telemetryPath={[{
          timestamp: "2026-01-01T00:00:00Z",
          position: { x: 100, y: 50, z: -5 },
          velocity: { x: 0, y: 0, z: 0 },
          armed: true,
          landed: false,
          collision: false,
        }]}
      />,
    );
    const map = screen.getByTestId("semantic-map");
    expect(map).toHaveAttribute("data-frame-origin-x", "100");
    expect(map).toHaveAttribute("data-frame-origin-y", "50");
    expect(map).toHaveAttribute("data-viewport-x-min", "95");
    expect(map).toHaveAttribute("data-viewport-x-max", "112");
    expect(map).toHaveAttribute("data-viewport-y-min", "44");
    expect(map).toHaveAttribute("data-viewport-y-max", "63");
    expect(screen.getByLabelText("地图自适应视野")).toHaveTextContent("HOME 100.0, 50.0");
    expect(screen.getByLabelText("任务起飞点")).toBeInTheDocument();
  });

  it("converts a drag selection to NED bounds clipped by the hard limit", () => {
    vi.stubGlobal("PointerEvent", MouseEvent);
    const onChange = vi.fn();
    render(
      <MapView
        zone={zone}
        frameOrigin={{ x: 100, y: 50, z: 0 }}
        hardBounds={{ x_min: -20, x_max: 20, y_min: -20, y_max: 20 }}
        safetyBounds={{ x_min: 0, x_max: 10, y_min: 0, y_max: 10 }}
        telemetryPath={[]}
        onSafetyBoundsChange={onChange}
      />,
    );
    const map = screen.getByRole("img", { name: "无人机 NED 轨迹图" });
    Object.defineProperty(map, "getBoundingClientRect", {
      value: () => ({ left: 0, top: 0, width: 200, height: 200, right: 200, bottom: 200, x: 0, y: 0, toJSON: () => ({}) }),
    });

    fireEvent.pointerDown(map, { clientX: -400, clientY: -400, button: 0, pointerId: 1 });
    fireEvent.pointerMove(map, { clientX: 600, clientY: 600, pointerId: 1 });
    fireEvent.pointerUp(map, { clientX: 600, clientY: 600, pointerId: 1 });

    expect(onChange).toHaveBeenCalled();
    expect(onChange).toHaveBeenLastCalledWith({ x_min: -20, x_max: 20, y_min: -20, y_max: 20 });
  });
});
