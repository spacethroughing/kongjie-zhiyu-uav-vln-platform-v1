from fastapi.testclient import TestClient


def test_health_and_plan_require_ready_simulator(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_PROVIDER", "mock")
    monkeypatch.setenv("HARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HARNESS_RUNS_DIR", str(tmp_path / "runs"))
    from harness.app import app

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["simulator_state"] == "STOPPED"
        with client.websocket_connect("/api/ws") as websocket:
            assert websocket.receive_json()["topic"] == "snapshot"
            assert client.post("/api/simulator/start", json={"scene_id": "mock"}).status_code == 200
            for _ in range(10):
                event = websocket.receive_json()
                if event["topic"] == "frame.preview":
                    assert event["payload"]["source"] == "live_preview"
                    assert event["payload"]["data_url"].startswith("data:image/png;base64,")
                    break
            else:
                raise AssertionError("continuous simulator preview was not published")
        smoke = client.post("/api/simulator/smoke")
        assert smoke.status_code == 200
        assert smoke.json()["frame"]["width"] == 64
        assert smoke.json()["telemetry"]["landed"] is True
        plan = client.post(
            "/api/missions/plan",
            json={"scene_id": "mock", "zone_id": "mock-fixture", "target_text": "red cube"},
        )
        assert plan.status_code == 200
        assert client.post(f"/api/missions/{plan.json()['id']}/approve").status_code == 200
