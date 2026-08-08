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

  it("converts a drag selection to NED bounds clipped by the hard limit", () => {
    vi.stubGlobal("PointerEvent", MouseEvent);
    const onChange = vi.fn();
    render(
      <MapView
        zone={zone}
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

    fireEvent.pointerDown(map, { clientX: -20, clientY: -20, button: 0, pointerId: 1 });
    fireEvent.pointerMove(map, { clientX: 220, clientY: 220, pointerId: 1 });
    fireEvent.pointerUp(map, { clientX: 220, clientY: 220, pointerId: 1 });

    expect(onChange).toHaveBeenCalled();
    expect(onChange).toHaveBeenLastCalledWith({ x_min: -20, x_max: 20, y_min: -20, y_max: 20 });
  });
});
