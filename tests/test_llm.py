from pathlib import Path

import httpx
import pytest

from harness.config import Settings
from harness.llm import OpenAICompatibleProvider
from harness.models import CameraFrame, Quaternion, Telemetry, Vec3


class StubResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"is_match":false,"confidence":1,"bbox_norm":null,"evidence":"none"}'
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
