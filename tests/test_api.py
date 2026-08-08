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
        assert client.post("/api/simulator/start", json={"scene_id": "mock"}).status_code == 200
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
