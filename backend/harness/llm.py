from __future__ import annotations

import asyncio
import base64
import json
import re
import struct
import zlib
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from .config import Settings
from .models import (
    BoundingBox,
    CameraFrame,
    DetectionAssessment,
    ExplorationRouteDecision,
    Quaternion,
    SemanticObservation,
    Telemetry,
    Vec3,
    VlmChatDecision,
    VlmChatMessage,
    VlmMissionIntent,
)


def _mock_mission_intent(message: str) -> VlmMissionIntent | None:
    compact = " ".join(message.strip().split())
    mapping_terms = (
        "整片区域", "整个区域", "全区域", "区域建图", "建立地图", "构建地图",
        "语义拓扑图", "占据与语义", "占据地图",
    )
    if any(term in compact for term in mapping_terms) and any(
        term in compact for term in ("探索", "建", "扫描", "覆盖")
    ):
        return VlmMissionIntent(
            kind="semantic_mapping",
            summary="覆盖给定安全区域，建立占据与语义拓扑图",
            coverage_target=0.85,
        )
    if not any(term in compact for term in ("探索", "搜索", "寻找", "查找")):
        return None
    # Directional movement with an explicit distance remains an immediate,
    # bounded explore command rather than a new mission plan.
    if re.search(r"\d+(?:\.\d+)?\s*(?:米|m)", compact, re.I):
        return None
    target_phrase = re.sub(
        r"^(?:请|请你|帮我|让无人机|开始|自动)?\s*(?:探索|搜索|寻找|查找)\s*",
        "",
        compact,
    )
    target_phrase = re.sub(
        r"(?:并完成|并建立|直到|然后|后再).*$", "", target_phrase
    )
    raw_targets = re.split(r"\s*(?:、|，|,|以及|还有|和|与|及)\s*", target_phrase)
    targets: list[str] = []
    for raw in raw_targets:
        target = re.sub(r"(?:这些)?(?:目标|物体|对象)$", "", raw.strip())
        if 2 <= len(target) <= 120 and target not in targets:
            targets.append(target)
    if not targets:
        return None
    return VlmMissionIntent(
        kind="target_search",
        summary=f"依次探索并取证 {len(targets)} 个开放词汇目标",
        targets=targets[:6],
    )


class ModelProvider(ABC):
    name: str

    @abstractmethod
    async def inspect(
        self, frame: CameraFrame, target_text: str, telemetry: Telemetry, observation_index: int
    ) -> DetectionAssessment:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    async def plan_exploration(
        self,
        target_text: str,
        topology: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> ExplorationRouteDecision:
        return ExplorationRouteDecision(
            waypoint_ids=[str(item["id"]) for item in candidates[:8]],
            rationale="deterministic candidate order",
        )

    async def chat(
        self,
        message: str,
        context: dict[str, Any],
        history: list[VlmChatMessage],
        frame: CameraFrame | None,
    ) -> VlmChatDecision:
        return VlmChatDecision(
            reply=f"Mock VLM：已收到“{message}”。当前仿真状态为 {context.get('simulator_state', 'UNKNOWN')}。"
        )

    async def probe(self) -> dict[str, Any]:
        return {"provider": self.name, "vision": True, "structured_output": True}


class MockProvider(ModelProvider):
    name = "mock"

    async def inspect(
        self, frame: CameraFrame, target_text: str, telemetry: Telemetry, observation_index: int
    ) -> DetectionAssessment:
        match = observation_index >= 2
        box = BoundingBox(x_min=0.38, y_min=0.30, x_max=0.62, y_max=0.70)
        return DetectionAssessment(
            frame_id=frame.frame_id,
            is_match=match,
            confidence=0.94 if match else 0.12,
            bbox_norm=box if match else None,
            evidence=f"Mock observation {observation_index}: {target_text}",
            observed_objects=[
                SemanticObservation(
                    label=target_text if match else "场景地标",
                    confidence=0.94 if match else 0.72,
                    bbox_norm=box,
                )
            ],
        )

    async def chat(
        self,
        message: str,
        context: dict[str, Any],
        history: list[VlmChatMessage],
        frame: CameraFrame | None,
    ) -> VlmChatDecision:
        action = None
        heading_degrees = None
        distance_m = None
        altitude_delta_m = None
        target_altitude_m = None
        mission = None
        if any(term in message for term in ("终止任务", "中止任务", "取消任务")):
            action = "abort"
        elif "返航" in message:
            action = "return-home"
        elif any(term in message for term in ("降落", "着陆")):
            action = "land"
        elif any(term in message for term in ("继续飞行", "继续任务", "恢复任务")):
            action = "resume"
        elif any(term in message for term in ("暂停", "悬停")):
            action = "pause"
        else:
            distance_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:米|m)", message, re.I)
            target_altitude_match = re.search(
                r"(?:飞到|升到|降到|高度(?:调整)?到)\s*(\d+(?:\.\d+)?)\s*(?:米|m)(?:高度)?",
                message,
                re.I,
            )
            climb_match = re.search(
                r"(?:升高|上升|爬升)\s*(\d+(?:\.\d+)?)\s*(?:米|m)",
                message,
                re.I,
            )
            descend_match = re.search(
                r"(?:降低|下降|下沉)\s*(\d+(?:\.\d+)?)\s*(?:米|m)",
                message,
                re.I,
            )
            if target_altitude_match:
                target_altitude_m = float(target_altitude_match.group(1))
                action = "change-altitude"
            elif climb_match:
                altitude_delta_m = min(10.0, float(climb_match.group(1)))
                action = "change-altitude"
            elif descend_match:
                altitude_delta_m = -min(10.0, float(descend_match.group(1)))
                action = "change-altitude"
            elif distance_match and any(term in message for term in ("探索", "飞", "前进")):
                camera_yaw = float(context.get("camera_yaw_degrees") or 0.0)
                direction_headings = (
                    ("东北", 45.0),
                    ("东南", 135.0),
                    ("西南", 225.0),
                    ("西北", 315.0),
                    ("向北", 0.0),
                    ("向东", 90.0),
                    ("向南", 180.0),
                    ("向西", 270.0),
                    ("前", camera_yaw),
                    ("右", camera_yaw + 90.0),
                    ("后", camera_yaw + 180.0),
                    ("左", camera_yaw - 90.0),
                )
                heading_degrees = next(
                    (heading for term, heading in direction_headings if term in message),
                    camera_yaw,
                ) % 360
                distance_m = min(20.0, float(distance_match.group(1)))
                action = "explore"
        if action is None:
            mission = _mock_mission_intent(message)
        stats = context.get("map", {}).get("stats", {})
        return VlmChatDecision(
            reply=(
                f"Mock VLM：当前状态 {context.get('run_state') or context.get('simulator_state', 'UNKNOWN')}，"
                f"LiDAR 已探索 {stats.get('explored_cells', 0)} 个栅格。"
                + (
                    (
                        f"已拆解任务：{mission.summary}，等待审核后执行。"
                    )
                    if mission is not None
                    else
                    f"建议按航向 {heading_degrees:.0f}° 探索 {distance_m:.1f} m。"
                    if action == "explore"
                    else (
                        f"建议将高度调整到 {target_altitude_m:.1f} m。"
                        if action == "change-altitude" and target_altitude_m is not None
                        else (
                            f"建议调整高度 {altitude_delta_m:+.1f} m。"
                            if action == "change-altitude"
                            else (f"建议执行 {action}。" if action else "未请求飞行控制。")
                        )
                    )
                )
            ),
            action=action,
            action_reason="用户明确提出白名单控制指令" if action else "",
            heading_degrees=heading_degrees,
            distance_m=distance_m,
            altitude_delta_m=altitude_delta_m,
            target_altitude_m=target_altitude_m,
            mission=mission,
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
        hostname = (urlparse(self.base_url).hostname or "").lower()
        self._is_zhipu = hostname == "bigmodel.cn" or hostname.endswith(".bigmodel.cn")
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
        self._probe_result: dict[str, Any] | None = None

    async def inspect(
        self, frame: CameraFrame, target_text: str, telemetry: Telemetry, observation_index: int
    ) -> DetectionAssessment:
        prompt = (
            "Semantic nodes with observations=1 are tentative clues, not confirmed objects. "
            "When target_clue_distance_m is present, prioritize the smallest values and "
            "do not claim a waypoint is near the target without that numeric evidence. "
            "你是无人机视觉搜索系统中的只读感知模块。"
            f"请判断图像中是否存在目标：{target_text!r}。"
            "同时列出图像中最多8个清晰且有定位价值的物体或地标。"
            "只能输出一个 JSON 对象，字段为 is_match(boolean)、confidence(0到1)、"
            "bbox_norm(null 或 {x_min,y_min,x_max,y_max}，坐标归一化到0到1)、evidence(简短中文)、"
            "observed_objects(array，每项为 {label,confidence,bbox_norm})。"
            "物体标签应简短稳定，同一物体跨帧尽量使用相同名称；忽略天空、地面和不确定纹理。"
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
            "max_tokens": 1024,
            "stream": False,
        }
        if self._is_zhipu:
            # GLM-5V-Turbo enables thinking by default. Detection is a short,
            # deterministic extraction task, so disable thinking and request
            # the provider's native JSON mode to avoid truncated/empty content.
            body["thinking"] = {"type": "disabled"}
            body["response_format"] = {"type": "json_object"}
        response = await self._post_with_rate_limit_backoff(body)
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        semantic_objects = []
        raw_objects = parsed.get("observed_objects")
        if isinstance(raw_objects, list):
            for item in raw_objects[:8]:
                try:
                    semantic_objects.append(
                        SemanticObservation.model_validate(item).model_dump(mode="json")
                    )
                except (TypeError, ValueError):
                    # Ancillary map labels must never invalidate an otherwise
                    # usable target assessment or interrupt flight guidance.
                    continue
        parsed["observed_objects"] = semantic_objects
        parsed["frame_id"] = frame.frame_id
        return DetectionAssessment.model_validate(parsed)

    async def chat(
        self,
        message: str,
        context: dict[str, Any],
        history: list[VlmChatMessage],
        frame: CameraFrame | None,
    ) -> VlmChatDecision:
        system_prompt = (
            "你是无人机任务控制台中的实时 VLM 助手。请根据当前机载画面、遥测、任务状态和地图统计，"
            "用简洁中文回答用户。你还负责把自然语言任务拆解为受限任务计划：当用户要求探索、搜索或寻找"
            "一个或多个目标时，mission.kind=target_search，并按用户表达顺序把每个独立目标写入 mission.targets；"
            "例如“探索圆锥体和橙色球体”必须拆成两个目标，不得合并成一个视觉查询。当用户要求探索整片区域、"
            "建立区域占据地图或语义拓扑图时，mission.kind=semantic_mapping、targets=[]，coverage_target 通常为 0.85。"
            "任务计划只生成确定性覆盖/搜索任务，必须等待后端安全校验和用户审核，禁止自行生成航点。"
            "普通问答、状态查询和方向讨论的 mission 必须为 null。任务计划与即时动作不能同时输出。"
            "你可以把用户明确表达的即时控制意图转换为一个白名单动作："
            "pause、resume、return-home、land、abort、explore、change-altitude。只有用户明确要求执行时才填写 action；"
            "询问状态、讨论路线或提出假设时 action 必须为 null。不得生成坐标、速度或其他直接飞控指令，"
            "不得声称可以绕过安全校验。abort 仅在用户明确说终止或取消任务时使用。"
            "当用户明确要求向某方向探索一定距离时使用 explore，并同时给出 heading_degrees 和 distance_m；"
            "NED 航向约定为北=0°、东=90°、南=180°、西=270°。对于前/后/左/右，"
            "必须基于 context.camera_yaw_degrees 换算为绝对 NED 航向。distance_m 必须在 1 到 20 米。"
            "当用户明确要求升高/降低一定米数或飞到某个高度时使用 change-altitude。相对变化填写 "
            "altitude_delta_m（升高为正、降低为负，范围 -10 到 10 米），绝对高度填写 "
            "target_altitude_m（相对起飞点向上的高度）；两者必须且只能填写一个。后端仍会按场景限高、"
            "最低高度、禁飞区和实时避障独立校验，禁止承诺指令必然执行。"
            "当前开放词汇目标以 context.target_text 为准，不得沿用历史对话中的旧目标。"
            "只输出 JSON：{\"reply\":\"中文反馈\",\"action\":null或白名单动作,"
            "\"action_reason\":\"简短原因\",\"heading_degrees\":null或数字,"
            "\"distance_m\":null或数字,\"altitude_delta_m\":null或数字,"
            "\"target_altitude_m\":null或数字,\"mission\":null或{\"kind\":\"target_search或semantic_mapping\","
            "\"summary\":\"任务摘要\",\"targets\":[\"目标1\",\"目标2\"],\"coverage_target\":0.85}}。"
            "非 explore 动作的航向和距离必须为 null；"
            "非 change-altitude 动作的两个高度字段必须为 null。"
        )
        conversation = [item.model_dump(mode="json") for item in history[-8:]]
        user_text = (
            f"conversation={json.dumps(conversation, ensure_ascii=False, separators=(',', ':'))}\n"
            f"context={json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
            f"current_user_message={message!r}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if frame is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame.scene_png_b64}"},
                }
            )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
            "stream": False,
        }
        if self._is_zhipu:
            body["thinking"] = {"type": "disabled"}
            body["response_format"] = {"type": "json_object"}
        response = await self._post_with_rate_limit_backoff(body)
        payload = response.json()
        parsed = _extract_json(payload["choices"][0]["message"]["content"])
        return VlmChatDecision.model_validate(parsed)

    async def _post_with_rate_limit_backoff(self, body: dict[str, Any]):
        """Retry transient provider throttling without creating VLM call bursts."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(4):
            response = await self._client.post(url, headers=headers, json=body)
            try:
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 429 or attempt == 3:
                    raise
                retry_after = error.response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 2**(attempt + 1)
                except ValueError:
                    delay = 2**(attempt + 1)
                await asyncio.sleep(max(0.25, min(delay, 15.0)))
        raise RuntimeError("unreachable model retry state")

    async def plan_exploration(
        self,
        target_text: str,
        topology: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> ExplorationRouteDecision:
        compact_topology = {
            "nodes": topology.get("nodes", [])[-100:],
            "edges": topology.get("edges", [])[-160:],
            "stats": topology.get("stats", {}),
        }
        system_prompt = (
            "你是无人机搜索任务的拓扑探索规划器。你的最高优先级是扩大地图覆盖率，"
            "输入的 candidates 只包含确定性连续覆盖航线上的局部前视窗口，并已按飞行顺序排列。"
            "你可以选择窗口内最有价值的局部探索目标，但不得要求跳过它之前的连接航点、倒序飞行或重排扫描带。"
            "优先探索 LiDAR 尚未覆盖的区域。candidates 中 mapping_status=unmapped_frontier "
            "或 explored=false 表示未建图前沿；partially_mapped_frontier 表示仅有少量雷达覆盖。"
            "只要存在未建图或部分建图候选，就不得把 "
            "mapping_status=mapped 或 explored=true 的已建图航点选作探索目标。"
            "已建图航点仅可作为通往未建图前沿的短距离连接点，或者在所有未建图候选均已耗尽时回访。"
            "在保持候选原顺序的前提下，优先更低 coverage_ratio，再考虑更高 novelty_m 和目标语义线索；"
            "不得为了低覆盖率在候选间来回跳转。"
            "只能从 candidates 的 id 中选择，不得创造坐标；最多返回8个。"
            "只输出 JSON：{\"waypoint_ids\":[...],\"rationale\":\"简短中文\"}。"
        )
        prompt = (
            f"搜索目标是：{target_text!r}。\n"
            "拓扑图包含已访问位置和语义物体；候选航点均已由确定性代码生成。\n"
            f"topology={json.dumps(compact_topology, ensure_ascii=False, separators=(',', ':'))}\n"
            f"candidates={json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}"
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False,
        }
        if self._is_zhipu:
            body["thinking"] = {"type": "disabled"}
            body["response_format"] = {"type": "json_object"}
        response = await self._post_with_rate_limit_backoff(body)
        payload = response.json()
        parsed = _extract_json(payload["choices"][0]["message"]["content"])
        return ExplorationRouteDecision.model_validate(parsed)

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
