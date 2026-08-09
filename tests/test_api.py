import json

from fastapi.testclient import TestClient

from harness.llm import MockProvider
from harness.models import RunState


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
        assert smoke.json()["lidar"]["data_frame"] == "VehicleInertialFrame"
        assert smoke.json()["lidar"]["sampled_point_count"] > 0
        assert smoke.json()["telemetry"]["landed"] is True
        plan = client.post(
            "/api/missions/plan",
            json={
                "scene_id": "mock",
                "zone_id": "mock-fixture",
                "target_text": "red cube",
                "safety_bounds": {"x_min": -20, "x_max": 20, "y_min": -20, "y_max": 20},
            },
        )
        assert plan.status_code == 200
        assert plan.json()["request"]["safety_bounds"]["x_min"] == -20
        assert client.post(f"/api/missions/{plan.json()['id']}/approve").status_code == 200


class ConfigurableMockProvider(MockProvider):
    name = "openai-compatible"

    def __init__(self, model: str):
        super().__init__()
        self.model = model
        self.closed = False

    async def probe(self):
        return {
            "provider": self.name,
            "model": self.model,
            "vision": True,
            "structured_output": True,
            "schema_valid": True,
        }

    async def close(self):
        self.closed = True


def test_runtime_provider_configuration_never_echoes_or_persists_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_PROVIDER", "mock")
    monkeypatch.setenv("HARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HARNESS_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    import harness.app as app_module

    with TestClient(app_module.app) as client:
        initial = client.get("/api/provider/config")
        assert initial.status_code == 200
        assert not initial.json()["api_key_configured"]
        assert initial.json()["models"][0]["id"] == "glm-4.6v-flashx"

        active = app_module.app.state.services.store.create_run("provider-lock-test")
        blocked = client.put(
            "/api/provider/config",
            json={"model": "glm-4.6v-flashx", "api_key": "must-not-appear"},
        )
        assert blocked.status_code == 409
        app_module.app.state.services.store.update_run(
            active, RunState.FAILED, error="test cleanup", ended=True
        )

        captured_settings = []
        candidate = ConfigurableMockProvider("glm-4.6v-flashx")

        def configured_provider_factory(settings):
            captured_settings.append(settings)
            return candidate

        monkeypatch.setattr(app_module, "create_provider", configured_provider_factory)
        secret = "runtime-secret-value"
        with client.websocket_connect("/api/ws") as websocket:
            assert websocket.receive_json()["topic"] == "snapshot"
            configured = client.put(
                "/api/provider/config",
                json={"model": "glm-4.6v-flashx", "api_key": secret},
            )
            assert configured.status_code == 200
            event = websocket.receive_json()
            assert event["topic"] == "provider.configured"

        response_text = configured.text
        event_text = json.dumps(event)
        assert secret not in response_text
        assert secret not in event_text
        assert "api_key" not in configured.json()
        assert configured.json()["model"] == "glm-4.6v-flashx"
        assert configured.json()["api_key_configured"] is True
        assert captured_settings[0].llm_api_key == secret
        assert app_module.app.state.services.settings.llm_api_key == secret

        status = client.get("/api/provider/config")
        assert status.status_code == 200
        assert secret not in status.text
        assert status.json()["api_key_configured"] is True

        invalid = client.put(
            "/api/provider/config",
            json={"model": "untrusted-model", "api_key": "another-secret"},
        )
        assert invalid.status_code == 422
