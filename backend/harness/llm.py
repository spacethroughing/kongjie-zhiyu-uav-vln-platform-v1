from __future__ import annotations

import base64
import json
import re
import struct
import zlib
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import httpx

from .config import Settings
from .models import BoundingBox, CameraFrame, DetectionAssessment, Quaternion, Telemetry, Vec3


class ModelProvider(ABC):
    name: str

    @abstractmethod
    async def inspect(
        self, frame: CameraFrame, target_text: str, telemetry: Telemetry, observation_index: int
    ) -> DetectionAssessment:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    async def probe(self) -> dict[str, Any]:
        return {"provider": self.name, "vision": True, "structured_output": True}


class MockProvider(ModelProvider):
    name = "mock"

    async def inspect(
        self, frame: CameraFrame, target_text: str, telemetry: Telemetry, observation_index: int
    ) -> DetectionAssessment:
        match = observation_index >= 2
        return DetectionAssessment(
            frame_id=frame.frame_id,
            is_match=match,
            confidence=0.94 if match else 0.12,
            bbox_norm=BoundingBox(x_min=0.38, y_min=0.30, x_max=0.62, y_max=0.70) if match else None,
            evidence=f"Mock observation {observation_index}: {target_text}",
        )


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


class OpenAICompatibleProvider(ModelProvider):
    name = "openai-compatible"

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_base_url or not settings.llm_model or not settings.llm_api_key:
            raise ValueError("LLM_BASE_URL, LLM_MODEL and LLM_API_KEY are required")
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model
        self._api_key = settings.llm_api_key
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
        self._probe_result: dict[str, Any] | None = None

    async def inspect(
        self, frame: CameraFrame, target_text: str, telemetry: Telemetry, observation_index: int
    ) -> DetectionAssessment:
        prompt = (
            "你是无人机视觉搜索系统中的只读感知模块。"
            f"请判断图像中是否存在目标：{target_text!r}。"
            "只能输出一个 JSON 对象，字段为 is_match(boolean)、confidence(0到1)、"
            "bbox_norm(null 或 {x_min,y_min,x_max,y_max}，坐标归一化到0到1)、evidence(简短中文)。"
            "不确定时 is_match=false；不得给飞行指令。"
        )
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{frame.scene_png_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            "stream": False,
        }
        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        parsed["frame_id"] = frame.frame_id
        return DetectionAssessment.model_validate(parsed)

    async def probe(self) -> dict[str, Any]:
        if self._probe_result:
            return self._probe_result

        def png_chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        raw = b"".join(b"\x00" + b"\xff\x00\x00" * 64 for _ in range(64))
        png = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
            + png_chunk(b"IDAT", zlib.compress(raw))
            + png_chunk(b"IEND", b"")
        )
        frame = CameraFrame(
            frame_id=str(uuid4()),
            width=64,
            height=64,
            scene_png_b64=base64.b64encode(png).decode("ascii"),
            camera_position=Vec3(x=0, y=0, z=0),
            camera_orientation=Quaternion(),
            fov_degrees=90,
        )
        assessment = await self.inspect(
            frame,
            "蓝色球体（能力探测图中不存在该目标）",
            Telemetry(position=Vec3(x=0, y=0, z=0)),
            0,
        )
        self._probe_result = {
            "provider": self.name,
            "model": self.model,
            "vision": True,
            "structured_output": True,
            "schema_valid": isinstance(assessment, DetectionAssessment),
        }
        return self._probe_result

    async def close(self) -> None:
        await self._client.aclose()


def create_provider(settings: Settings) -> ModelProvider:
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "openai-compatible":
        return OpenAICompatibleProvider(settings)
    raise ValueError(f"unknown HARNESS_PROVIDER: {settings.provider}")
