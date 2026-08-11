from pathlib import Path

import httpx
import pytest

from harness.config import Settings
from harness.llm import MockProvider, OpenAICompatibleProvider
from harness.models import CameraFrame, Quaternion, Telemetry, Vec3, VlmChatMessage


class StubResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"is_match":false,"confidence":1,"bbox_norm":null,'
                            '"evidence":"none","observed_objects":['
                            '{"label":"tower","confidence":0.8,"bbox_norm":'
                            '{"x_min":0.1,"y_min":0.1,"x_max":0.3,"y_max":0.4}},null]}'
                        )
                    }
                }
            ]
        }


class CapturingClient:
    def __init__(self):
        self.body = None

    async def post(self, url, *, headers, json):
        self.body = json
        return StubResponse()

    async def aclose(self):
        return None


class RateLimitedClient(CapturingClient):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def post(self, url, *, headers, json):
        self.calls += 1
        if self.calls < 3:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                request=httpx.Request("POST", url),
            )
        return await super().post(url, headers=headers, json=json)


class PlanningResponse(StubResponse):
    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"waypoint_ids":["route-1","route-2"],"rationale":"连续探索未访问区域"}'
                    }
                }
            ]
        }


class PlanningClient(CapturingClient):
    async def post(self, url, *, headers, json):
        self.body = json
        return PlanningResponse()


class ChatResponse(StubResponse):
    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"reply":"前方地图仍有未探索区域。","action":"return-home",'
                            '"action_reason":"用户明确要求返航"}'
                        )
                    }
                }
            ]
        }


class ChatClient(CapturingClient):
    async def post(self, url, *, headers, json):
        self.body = json
        return ChatResponse()


@pytest.mark.asyncio
async def test_zhipu_requests_native_json_without_thinking(tmp_path: Path):
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        scenes_file=tmp_path / "scenes.json",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        provider="openai-compatible",
        llm_base_url="https://open.bigmodel.cn/api/paas/v4",
        llm_model="glm-5v-turbo",
        llm_api_key="test-only",
        llm_timeout_seconds=10,
    )
    provider = OpenAICompatibleProvider(settings)
    await provider._client.aclose()
    client = CapturingClient()
    provider._client = client
    frame = CameraFrame(
        frame_id="frame-1",
        width=1,
        height=1,
        scene_png_b64="AA==",
        camera_position=Vec3(x=0, y=0, z=-5),
        camera_orientation=Quaternion(),
        fov_degrees=90,
    )

    assessment = await provider.inspect(
        frame,
        "red cube",
        Telemetry(position=Vec3(x=0, y=0, z=-5)),
        1,
    )

    assert not assessment.is_match
    assert [item.label for item in assessment.observed_objects] == ["tower"]
    assert client.body["thinking"] == {"type": "disabled"}
    assert client.body["response_format"] == {"type": "json_object"}
    assert client.body["max_tokens"] == 1024
    await provider.close()


@pytest.mark.asyncio
async def test_provider_retries_transient_rate_limits(tmp_path: Path, monkeypatch):
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        scenes_file=tmp_path / "scenes.json",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        provider="openai-compatible",
        llm_base_url="https://open.bigmodel.cn/api/paas/v4",
        llm_model="glm-4.6v-flash",
        llm_api_key="test-only",
        llm_timeout_seconds=10,
    )
    provider = OpenAICompatibleProvider(settings)
    await provider._client.aclose()
    client = RateLimitedClient()
    provider._client = client

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr("harness.llm.asyncio.sleep", no_wait)
    frame = CameraFrame(
        frame_id="frame-rate-limit",
        width=1,
        height=1,
        scene_png_b64="AA==",
        camera_position=Vec3(x=0, y=0, z=-5),
        camera_orientation=Quaternion(),
        fov_degrees=90,
    )
    assessment = await provider.inspect(
        frame,
        "orange cone",
        Telemetry(position=Vec3(x=0, y=0, z=-5)),
        1,
    )

    assert client.calls == 3
    assert not assessment.is_match
    await provider.close()


@pytest.mark.asyncio
async def test_provider_plans_only_named_topology_candidates(tmp_path: Path):
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        scenes_file=tmp_path / "scenes.json",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        provider="openai-compatible",
        llm_base_url="https://open.bigmodel.cn/api/paas/v4",
        llm_model="glm-4.6v-flashx",
        llm_api_key="test-only",
        llm_timeout_seconds=10,
    )
    provider = OpenAICompatibleProvider(settings)
    await provider._client.aclose()
    client = PlanningClient()
    provider._client = client
    decision = await provider.plan_exploration(
        "圆锥体",
        {"nodes": [], "edges": [], "stats": {}},
        [
            {"id": "route-1", "position": {"x": 0, "y": -8, "z": -5}},
            {"id": "route-2", "position": {"x": 0, "y": -16, "z": -5}},
        ],
    )
    assert decision.waypoint_ids == ["route-1", "route-2"]
    assert client.body["response_format"] == {"type": "json_object"}
    assert client.body["messages"][0]["role"] == "system"
    system_prompt = client.body["messages"][0]["content"]
    assert "最高优先级是扩大地图覆盖率" in system_prompt
    assert "优先探索 LiDAR 尚未覆盖的区域" in system_prompt
    assert "只要存在未建图或部分建图候选" in system_prompt
    assert "不得创造坐标" in system_prompt
    assert client.body["messages"][1]["role"] == "user"
    assert "搜索目标" in client.body["messages"][1]["content"]
    await provider.close()


@pytest.mark.asyncio
async def test_provider_chat_uses_live_context_and_structured_control_action(tmp_path: Path):
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        scenes_file=tmp_path / "scenes.json",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        provider="openai-compatible",
        llm_base_url="https://open.bigmodel.cn/api/paas/v4",
        llm_model="glm-4.6v-flashx",
        llm_api_key="test-only",
        llm_timeout_seconds=10,
    )
    provider = OpenAICompatibleProvider(settings)
    await provider._client.aclose()
    client = ChatClient()
    provider._client = client

    decision = await provider.chat(
        "现在返航",
        {
            "simulator_state": "READY",
            "run_state": "SEARCHING",
            "map": {"stats": {"explored_cells": 42}},
        },
        [VlmChatMessage(role="assistant", content="任务正在搜索")],
        None,
    )

    assert decision.action == "return-home"
    assert "未探索区域" in decision.reply
    assert client.body["messages"][0]["role"] == "system"
    assert "白名单动作" in client.body["messages"][0]["content"]
    assert "explored_cells" in client.body["messages"][1]["content"][0]["text"]
    await provider.close()


@pytest.mark.asyncio
async def test_mock_chat_converts_relative_direction_to_absolute_ned_heading():
    provider = MockProvider()
    decision = await provider.chat(
        "向右探索 10 米",
        {
            "simulator_state": "READY",
            "run_state": "SEARCHING",
            "camera_yaw_degrees": 30,
            "map": {"stats": {"explored_cells": 5}},
        },
        [],
        None,
    )
    assert decision.action == "explore"
    assert decision.heading_degrees == pytest.approx(120)
    assert decision.distance_m == pytest.approx(10)


@pytest.mark.asyncio
async def test_mock_chat_decomposes_multi_target_and_mapping_missions():
    provider = MockProvider()
    context = {"simulator_state": "READY", "map": {"stats": {"explored_cells": 0}}}
    targets = await provider.chat("探索圆锥体和橙色球体", context, [], None)
    assert targets.action is None
    assert targets.mission is not None
    assert targets.mission.kind == "target_search"
    assert targets.mission.targets == ["圆锥体", "橙色球体"]

    mapping = await provider.chat(
        "探索整片区域并建立区域占据与语义拓扑图", context, [], None
    )
    assert mapping.mission is not None
    assert mapping.mission.kind == "semantic_mapping"
    assert mapping.mission.targets == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_delta", "expected_target"),
    [
        ("升高 3 米", 3, None),
        ("下降 2 米", -2, None),
        ("飞到 12 米高度", None, 12),
    ],
)
async def test_mock_chat_parses_safe_altitude_controls(
    message, expected_delta, expected_target
):
    decision = await MockProvider().chat(
        message,
        {"simulator_state": "READY", "run_state": "SEARCHING", "map": {}},
        [],
        None,
    )
    assert decision.action == "change-altitude"
    assert decision.altitude_delta_m == expected_delta
    assert decision.target_altitude_m == expected_target
