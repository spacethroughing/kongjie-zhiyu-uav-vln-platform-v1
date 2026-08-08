import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MapView } from "./MapView";

describe("MapView", () => {
  it("renders the configured geofence", () => {
    render(<MapView zone={{id:"z",name:"Zone",polygon:{points:[[0,0],[10,0],[10,10],[0,10]]},search_altitude_m:5,lane_spacing_m:2}} telemetryPath={[]} />);
    expect(screen.getByRole("img", { name: "无人机 NED 轨迹图" })).toBeInTheDocument();
  });
});

