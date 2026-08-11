import React from "react";
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { nedToTwinWorld, quaternionYawDegrees } from "./DigitalTwinView";
import { DigitalTwinView } from "./DigitalTwinView";
import type { Telemetry } from "./types";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("DigitalTwinView transforms", () => {
  it("maps AirSim NED axes into an east-up-north 3D scene", () => {
    expect(nedToTwinWorld({ x: 12, y: 7, z: -5 }, 0, { x: 2, y: 3 })).toEqual({
      x: 4,
      y: 5,
      z: -10,
    });
  });

  it("derives body heading from the streamed quaternion", () => {
    const halfTurn = Math.PI / 4;
    expect(quaternionYawDegrees({
      w: Math.cos(halfTurn),
      x: 0,
      y: 0,
      z: Math.sin(halfTurn),
    })).toBeCloseTo(90);
  });

  it("keeps one canvas loop alive across high-frequency telemetry rerenders", () => {
    const requestFrame = vi.fn(() => 17);
    const cancelFrame = vi.fn();
    const disconnect = vi.fn();
    let resizeObservers = 0;
    vi.stubGlobal("requestAnimationFrame", requestFrame);
    vi.stubGlobal("cancelAnimationFrame", cancelFrame);
    vi.stubGlobal("ResizeObserver", class {
      constructor(_callback: ResizeObserverCallback) { resizeObservers += 1; }
      observe() {}
      unobserve() {}
      disconnect() { disconnect(); }
    });
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn(() => ({ matches: false })),
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      setTransform: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
    const first: Telemetry = {
      timestamp: "2026-08-11T00:00:00Z",
      position: { x: 0, y: 0, z: -5 },
      velocity: { x: 1, y: 0, z: 0 },
      armed: true,
      landed: false,
      collision: false,
    };
    const second: Telemetry = {
      ...first,
      timestamp: "2026-08-11T00:00:00.2Z",
      position: { x: 0.2, y: 0, z: -5 },
    };
    const baseProps = {
      lidarFrames: [],
      cameraYawDegrees: 0,
      state: "SEARCHING" as const,
    };
    const view = render(React.createElement(DigitalTwinView, {
      ...baseProps,
      telemetryPath: [first],
    }));

    expect(requestFrame).toHaveBeenCalledTimes(1);
    expect(resizeObservers).toBe(1);
    view.rerender(React.createElement(DigitalTwinView, {
      ...baseProps,
      cameraYawDegrees: 15,
      telemetryPath: [first, second],
      lidarFrames: [{
        timestamp: second.timestamp,
        data_frame: "VehicleInertialFrame",
        point_count: 1,
        sampled_point_count: 1,
        vehicle_position: second.position,
        points: [[1, 2, -3]],
      }],
    }));

    expect(requestFrame).toHaveBeenCalledTimes(1);
    expect(cancelFrame).not.toHaveBeenCalled();
    expect(resizeObservers).toBe(1);
    view.unmount();
    expect(cancelFrame).toHaveBeenCalledWith(17);
    expect(disconnect).toHaveBeenCalledTimes(1);
  });
});
