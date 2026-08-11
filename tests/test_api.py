import json

from fastapi.testclient import TestClient

from harness.llm import MockProvider
from harness.models import RunState, VlmChatDecision


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
            snapshot = websocket.receive_json()
            assert snapshot["topic"] == "snapshot"
            assert snapshot["payload"]["map"]["stats"]["occupancy_cells"] == 0
            assert client.post("/api/simulator/start", json={"scene_id": "mock"}).status_code == 200
            seen_preview = False
            seen_map = False
            for _ in range(20):
                event = websocket.receive_json()
                if event["topic"] == "frame.preview":
                    assert event["payload"]["source"] == "live_preview"
                    assert event["payload"]["data_url"].startswith("data:image/png;base64,")
                    assert event["payload"]["depth_data_url"].startswith("data:image/png;base64,")
                    seen_preview = True
                elif event["topic"] == "map.update":
                    assert "nodes" in event["payload"]
                    assert "obstacles" in event["payload"]
                    seen_map = True
                if seen_preview and seen_map:
                    break
            if not seen_preview or not seen_map:
                raise AssertionError("continuous simulator preview was not published")
        smoke = client.post("/api/simulator/smoke")
        assert smoke.status_code == 200
        assert smoke.json()["frame"]["width"] == 64
        assert smoke.json()["lidar"]["data_frame"] == "VehicleInertialFrame"
        assert smoke.json()["lidar"]["sampled_point_count"] > 0
        assert smoke.json()["telemetry"]["landed"] is True
        chat = client.post(
            "/api/vlm/chat",
            json={
                "message": "汇报当前地图覆盖",
                "target_text": "橙色球体",
                "execute_command": False,
            },
        )
        assert chat.status_code == 200
        assert "LiDAR 已探索" in chat.json()["reply"]
        assert chat.json()["requested_action"] is None
        assert chat.json()["context_target_text"] == "橙色球体"
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
        initial = plan.json()
        parameters = {
            **initial["parameters"],
            "search_altitude_m": 7,
            "lane_spacing_m": 4,
            "max_speed_mps": 2,
            "approach_speed_mps": 0.8,
            "min_standoff_m": 4,
            "min_clearance_m": 2,
            "max_mission_seconds": 300,
        }
        revised = client.patch(
            f"/api/missions/{initial['id']}",
            json={"base_version": initial["version"], "parameters": parameters},
        )
        assert revised.status_code == 200, revised.text
        revised_payload = revised.json()
        assert revised_payload["id"] != initial["id"]
        assert revised_payload["version"] == 2
        assert revised_payload["parameters"]["search_altitude_m"] == 7
        assert revised_payload["safety"]["max_speed_mps"] == 2
        assert all(point["position"]["z"] == -7 for point in revised_payload["route"])
        assert client.post(f"/api/missions/{revised_payload['id']}/approve").status_code == 200
        rejected = client.patch(
            f"/api/missions/{revised_payload['id']}",
            json={"base_version": 2, "parameters": parameters},
        )
        assert rejected.status_code == 409


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


def test_vlm_chat_executes_only_a_whitelisted_mission_control(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_PROVIDER", "mock")
    monkeypatch.setenv("HARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HARNESS_RUNS_DIR", str(tmp_path / "runs"))
    import harness.app as app_module

    with TestClient(app_module.app) as client:
        service = app_module.app.state.services
        active = service.store.create_run("vlm-chat-control")
        calls = []

        seen_context = []

        async def chat_command(message, context, history, frame):
            seen_context.append(context)
            return VlmChatDecision(
                reply="已解析为向东探索 8 米。",
                action="explore",
                action_reason="用户明确要求定向探索",
                heading_degrees=90,
                distance_m=8,
            )

        async def record_exploration(run_id, heading_degrees, distance_m):
            calls.append((run_id, heading_degrees, distance_m))

        service.provider.chat = chat_command
        service.missions.queue_exploration = record_exploration
        response = client.post(
            "/api/vlm/chat",
            json={
                "message": "向东探索 8 米",
                "target_text": "新的橙色圆锥体",
                "run_id": active.id,
                "execute_command": True,
            },
        )

        assert response.status_code == 200
        assert response.json()["executed_action"] == "explore"
        assert response.json()["command_status"] == "queued"
        assert response.json()["heading_degrees"] == 90
        assert response.json()["distance_m"] == 8
        assert response.json()["context_target_text"] == "新的橙色圆锥体"
        assert calls == [(active.id, 90, 8)]
        assert seen_context[0]["target_text"] == "新的橙色圆锥体"

        altitude_calls = []

        async def chat_altitude(message, context, history, frame):
            return VlmChatDecision(
                reply="已解析为升高 3 米。",
                action="change-altitude",
                action_reason="用户明确要求调整高度",
                altitude_delta_m=3,
            )

        async def record_altitude(
            run_id, *, altitude_delta_m=None, target_altitude_m=None
        ):
            altitude_calls.append(
                (run_id, altitude_delta_m, target_altitude_m)
            )

        service.provider.chat = chat_altitude
        service.missions.queue_altitude = record_altitude
        altitude_response = client.post(
            "/api/vlm/chat",
            json={
                "message": "升高 3 米",
                "target_text": "新的橙色圆锥体",
                "run_id": active.id,
                "execute_command": True,
            },
        )
        assert altitude_response.status_code == 200
        assert altitude_response.json()["executed_action"] == "change-altitude"
        assert altitude_response.json()["command_status"] == "queued"
        assert altitude_response.json()["altitude_delta_m"] == 3
        assert altitude_calls == [(active.id, 3, None)]
        service.store.update_run(
            active, RunState.ABORTED, error="test cleanup", ended=True
        )


def test_vlm_chat_creates_reviewable_compound_and_mapping_plans(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_PROVIDER", "mock")
    monkeypatch.setenv("HARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HARNESS_RUNS_DIR", str(tmp_path / "runs"))
    from harness.app import app

    with TestClient(app) as client:
        multi = client.post(
            "/api/vlm/chat",
            json={
                "message": "探索圆锥体和橙色球体",
                "scene_id": "mock",
                "zone_id": "mock-fixture",
                "target_text": "",
                "execute_command": True,
            },
        )
        assert multi.status_code == 200
        assert multi.json()["command_status"] == "planned"
        assert [
            task["target_text"] for task in multi.json()["mission_plan"]["tasks"]
        ] == ["圆锥体", "橙色球体"]

        mapping = client.post(
            "/api/vlm/chat",
            json={
                "message": "探索整片区域建立区域占据与语义拓扑图",
                "scene_id": "mock",
                "zone_id": "mock-fixture",
                "execute_command": True,
            },
        )
        assert mapping.status_code == 200
        assert mapping.json()["command_status"] == "planned"
        plan = mapping.json()["mission_plan"]
        assert plan["request"]["mission_mode"] == "semantic_mapping"
        assert plan["tasks"][0]["kind"] == "semantic_mapping"
