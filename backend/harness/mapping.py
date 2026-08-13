from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import Vec3


@dataclass
class _OccupancyCell:
    x: float
    y: float
    z: float
    hits: int
    last_seen: float


@dataclass
class _CoverageCell:
    x: float
    y: float
    scans: int
    last_seen: float


@dataclass
class _TraversalCell:
    x: float
    y: float
    visits: int
    last_seen: float


@dataclass
class _SemanticLandmark:
    id: str
    label: str
    position: Vec3
    confidence: float
    observations: int
    frame_id: str
    last_seen: float
    aliases: set[str]


class TopologicalMapper:
    """Small in-memory 2.5D map for live visualization and search ordering.

    LiDAR remains the geometric source of truth.  VLM labels are attached as
    semantic landmarks and are never used as collision-free evidence.
    """

    def __init__(
        self,
        *,
        voxel_size_m: float = 1.0,
        coverage_cell_size_m: float = 2.0,
        place_spacing_m: float = 4.0,
        semantic_merge_m: float = 5.0,
        max_cells: int = 1800,
        max_coverage_cells: int = 5000,
        max_traversal_cells: int = 8000,
        max_places: int = 320,
        max_semantics: int = 120,
    ) -> None:
        self.voxel_size_m = voxel_size_m
        self.coverage_cell_size_m = coverage_cell_size_m
        self.place_spacing_m = place_spacing_m
        self.semantic_merge_m = semantic_merge_m
        self.max_cells = max_cells
        self.max_coverage_cells = max_coverage_cells
        self.max_traversal_cells = max_traversal_cells
        self.max_places = max_places
        self.max_semantics = max_semantics
        self.reset()

    def reset(self) -> None:
        self._occupancy: dict[tuple[int, int], _OccupancyCell] = {}
        self._coverage: dict[tuple[int, int], _CoverageCell] = {}
        self._traversal: dict[tuple[int, int], _TraversalCell] = {}
        self._places: list[dict] = []
        self._edges: list[dict] = []
        self._semantics: list[_SemanticLandmark] = []
        self._next_place_id = 1
        self._next_semantic_id = 1
        self._last_pose: Vec3 | None = None
        self._revision = 0

    @staticmethod
    def _planar_distance(left: Vec3, right: Vec3) -> float:
        return math.hypot(left.x - right.x, left.y - right.y)

    def integrate_pose(self, position: Vec3) -> bool:
        if not all(math.isfinite(value) for value in (position.x, position.y, position.z)):
            return False
        traversal_changed = self._integrate_traversal(position)
        if self._places:
            previous = Vec3.model_validate(self._places[-1]["position"])
            if self._planar_distance(previous, position) < self.place_spacing_m:
                if traversal_changed:
                    self._revision += 1
                return False
        node_id = f"place-{self._next_place_id}"
        self._next_place_id += 1
        self._places.append(
            {
                "id": node_id,
                "kind": "place",
                "position": position.model_dump(),
                "label": f"P{self._next_place_id - 1}",
            }
        )
        if len(self._places) > 1:
            self._edges.append(
                {
                    "source": self._places[-2]["id"],
                    "target": node_id,
                    "kind": "traversal",
                }
            )
        if len(self._places) > self.max_places:
            removed = self._places.pop(0)["id"]
            self._edges = [
                edge
                for edge in self._edges
                if edge["source"] != removed and edge["target"] != removed
            ]
        self._revision += 1
        return True

    def _integrate_traversal(self, position: Vec3) -> bool:
        """Rasterize the actual vehicle path independently of LiDAR coverage.

        LiDAR rays describe observed space, not where the vehicle has flown.  A
        separate traversal layer lets the planner penalize true path re-entry
        without treating every visible free-space cell as a previous visit.
        """
        now = time.monotonic()
        previous = self._last_pose
        self._last_pose = position
        positions = [position]
        if previous is not None:
            planar = self._planar_distance(previous, position)
            # Do not draw a long artificial path after a simulator reset,
            # teleport, or a fixture relocation.
            if 0.0 < planar <= 20.0:
                steps = max(1, math.ceil(planar / max(0.5, self.coverage_cell_size_m * 0.5)))
                positions = [
                    Vec3(
                        x=previous.x + (position.x - previous.x) * step / steps,
                        y=previous.y + (position.y - previous.y) * step / steps,
                        z=previous.z + (position.z - previous.z) * step / steps,
                    )
                    for step in range(1, steps + 1)
                ]
        changed = False
        for sample in positions:
            key = self._coverage_key(sample)
            cell = self._traversal.get(key)
            if cell:
                cell.visits += 1
                cell.last_seen = now
            else:
                self._traversal[key] = _TraversalCell(
                    x=(key[0] + 0.5) * self.coverage_cell_size_m,
                    y=(key[1] + 0.5) * self.coverage_cell_size_m,
                    visits=1,
                    last_seen=now,
                )
                changed = True
        if len(self._traversal) > self.max_traversal_cells:
            keep = sorted(
                self._traversal.items(),
                key=lambda item: (item[1].last_seen, item[1].visits),
                reverse=True,
            )[: self.max_traversal_cells]
            self._traversal = dict(keep)
        return changed

    def integrate_lidar(self, points: list[Vec3], vehicle_position: Vec3) -> int:
        now = time.monotonic()
        accepted = 0
        # A cell receives at most one hit per scan.  Counting every laser ray
        # made dense surfaces look certain after a single sweep and promoted
        # isolated noise into permanent obstacles.
        scan_cells: dict[tuple[int, int], list[float]] = {}
        coverage_keys: set[tuple[int, int]] = {self._coverage_key(vehicle_position)}
        for point in points:
            values = (point.x, point.y, point.z)
            if not all(math.isfinite(value) for value in values):
                continue
            planar = self._planar_distance(point, vehicle_position)
            # Ignore the vehicle body, far noise and the ground/ceiling.  The
            # resulting 2.5D layer represents obstacles relevant to this flight.
            if planar < 0.8 or planar > 45.0:
                continue
            if abs(point.z - vehicle_position.z) > 2.5:
                continue
            key = (
                int(math.floor(point.x / self.voxel_size_m)),
                int(math.floor(point.y / self.voxel_size_m)),
            )
            aggregate = scan_cells.setdefault(key, [0.0, 0.0, 0.0, 0.0])
            aggregate[0] += point.x
            aggregate[1] += point.y
            aggregate[2] += point.z
            aggregate[3] += 1
            accepted += 1
            # Mark every traversed LiDAR ray cell as observed free space.  The
            # final hit cell is excluded because it represents an obstacle.
            ray_steps = max(
                1,
                math.ceil(planar / max(0.5, self.coverage_cell_size_m * 0.6)),
            )
            for step in range(ray_steps):
                ratio = step / ray_steps
                coverage_keys.add(
                    self._coverage_key(
                        Vec3(
                            x=vehicle_position.x + (point.x - vehicle_position.x) * ratio,
                            y=vehicle_position.y + (point.y - vehicle_position.y) * ratio,
                            z=vehicle_position.z + (point.z - vehicle_position.z) * ratio,
                        )
                    )
                )
        for key in coverage_keys:
            cell = self._coverage.get(key)
            if cell:
                cell.scans += 1
                cell.last_seen = now
            else:
                self._coverage[key] = _CoverageCell(
                    x=(key[0] + 0.5) * self.coverage_cell_size_m,
                    y=(key[1] + 0.5) * self.coverage_cell_size_m,
                    scans=1,
                    last_seen=now,
                )
        for key, aggregate in scan_cells.items():
            count = aggregate[3]
            point = Vec3(
                x=aggregate[0] / count,
                y=aggregate[1] / count,
                z=aggregate[2] / count,
            )
            cell = self._occupancy.get(key)
            if cell:
                weight = min(cell.hits, 8)
                cell.x = (cell.x * weight + point.x) / (weight + 1)
                cell.y = (cell.y * weight + point.y) / (weight + 1)
                cell.z = (cell.z * weight + point.z) / (weight + 1)
                cell.hits += 1
                cell.last_seen = now
            else:
                self._occupancy[key] = _OccupancyCell(
                    x=point.x,
                    y=point.y,
                    z=point.z,
                    hits=1,
                    last_seen=now,
                )
        if len(self._occupancy) > self.max_cells:
            keep = sorted(
                self._occupancy.items(),
                key=lambda item: (item[1].hits, item[1].last_seen),
                reverse=True,
            )[: self.max_cells]
            self._occupancy = dict(keep)
        if len(self._coverage) > self.max_coverage_cells:
            keep_coverage = sorted(
                self._coverage.items(),
                key=lambda item: (item[1].scans, item[1].last_seen),
                reverse=True,
            )[: self.max_coverage_cells]
            self._coverage = dict(keep_coverage)
        if accepted:
            self._revision += 1
        return accepted

    def _coverage_key(self, position: Vec3) -> tuple[int, int]:
        return (
            int(math.floor(position.x / self.coverage_cell_size_m)),
            int(math.floor(position.y / self.coverage_cell_size_m)),
        )

    def exploration_coverage(self, position: Vec3, radius_m: float = 3.5) -> float:
        """Return the fraction of nearby 2D cells observed by LiDAR rays."""
        radius_cells = max(1, math.ceil(radius_m / self.coverage_cell_size_m))
        center_x, center_y = self._coverage_key(position)
        expected: list[tuple[int, int]] = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cell_x = (center_x + dx + 0.5) * self.coverage_cell_size_m
                cell_y = (center_y + dy + 0.5) * self.coverage_cell_size_m
                if math.hypot(cell_x - position.x, cell_y - position.y) <= radius_m:
                    expected.append((center_x + dx, center_y + dy))
        if not expected:
            return 0.0
        observed = sum(key in self._coverage for key in expected)
        return observed / len(expected)

    def segment_exploration_coverage(
        self,
        start: Vec3,
        end: Vec3,
        *,
        sample_step_m: float | None = None,
    ) -> float:
        """Return the fraction of a candidate segment already observed by LiDAR."""
        keys = self._segment_keys(start, end, sample_step_m=sample_step_m)
        if not keys:
            return 0.0
        return sum(key in self._coverage for key in keys) / len(keys)

    def trajectory_revisit_ratio(
        self,
        start: Vec3,
        end: Vec3,
        *,
        sample_step_m: float | None = None,
    ) -> float:
        """Return how much of a candidate segment re-enters the flown path.

        The first cell is excluded because every new command necessarily
        starts in the vehicle's current, already-visited cell.
        """
        keys = self._segment_keys(start, end, sample_step_m=sample_step_m)
        if len(keys) <= 1:
            return 0.0
        future_keys = keys[1:]
        return sum(key in self._traversal for key in future_keys) / len(future_keys)

    def _segment_keys(
        self,
        start: Vec3,
        end: Vec3,
        *,
        sample_step_m: float | None = None,
    ) -> list[tuple[int, int]]:
        planar = self._planar_distance(start, end)
        if planar <= 1e-9:
            return [self._coverage_key(end)]
        step_m = sample_step_m or max(0.5, self.coverage_cell_size_m * 0.5)
        steps = max(1, math.ceil(planar / step_m))
        keys: list[tuple[int, int]] = []
        for step in range(steps + 1):
            ratio = step / steps
            key = self._coverage_key(
                Vec3(
                    x=start.x + (end.x - start.x) * ratio,
                    y=start.y + (end.y - start.y) * ratio,
                    z=start.z + (end.z - start.z) * ratio,
                )
            )
            if not keys or key != keys[-1]:
                keys.append(key)
        return keys

    def coverage_ratio_in_polygon(self, points: list[tuple[float, float]]) -> float:
        """Return the LiDAR-observed fraction of cells inside a world-NED polygon."""
        if len(points) < 3:
            return 0.0

        def inside(x: float, y: float) -> bool:
            result = False
            previous = len(points) - 1
            for index, (xi, yi) in enumerate(points):
                xj, yj = points[previous]
                if (yi > y) != (yj > y):
                    crossing_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
                    if x < crossing_x:
                        result = not result
                previous = index
            return result

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_key_x = math.floor(min(xs) / self.coverage_cell_size_m)
        max_key_x = math.floor(max(xs) / self.coverage_cell_size_m)
        min_key_y = math.floor(min(ys) / self.coverage_cell_size_m)
        max_key_y = math.floor(max(ys) / self.coverage_cell_size_m)
        expected: list[tuple[int, int]] = []
        for key_x in range(min_key_x, max_key_x + 1):
            for key_y in range(min_key_y, max_key_y + 1):
                center_x = (key_x + 0.5) * self.coverage_cell_size_m
                center_y = (key_y + 0.5) * self.coverage_cell_size_m
                if inside(center_x, center_y):
                    expected.append((key_x, key_y))
        if not expected:
            return 0.0
        return sum(key in self._coverage for key in expected) / len(expected)

    @staticmethod
    def _normalized_label(label: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", label.lower())

    @classmethod
    def _ignored_label(cls, label: str) -> bool:
        normalized = cls._normalized_label(label)
        ignored = (
            "天空",
            "地面",
            "路面",
            "道路",
            "背景",
            "阴影",
            "纹理",
            "场景地标",
        )
        return normalized in {"物体", "目标", "object"} or any(
            term in normalized for term in ignored
        )

    @classmethod
    def _object_category(cls, label: str) -> str | None:
        normalized = cls._normalized_label(label)
        categories = (
            ("sphere", ("球体", "圆球", "球形", "sphere", "ball")),
            ("cone", ("圆锥", "锥体", "锥形", "cone")),
            ("cube", ("立方体", "方块", "方体", "cube", "block")),
            ("cylinder", ("圆柱", "柱体", "cylinder")),
            ("wall", ("墙壁", "墙体", "围墙", "wall")),
            ("tree", ("树木", "树", "tree")),
            ("vehicle", ("车辆", "汽车", "车", "vehicle", "car")),
            ("building", ("建筑", "房屋", "building")),
        )
        return next(
            (category for category, terms in categories if any(term in normalized for term in terms)),
            None,
        )

    @classmethod
    def _color(cls, label: str) -> str | None:
        normalized = cls._normalized_label(label)
        colors = (
            "橙色",
            "红色",
            "蓝色",
            "绿色",
            "黄色",
            "紫色",
            "白色",
            "黑色",
            "灰色",
            "深色",
            "浅色",
        )
        return next((color for color in colors if color in normalized), None)

    @classmethod
    def _labels_compatible(cls, left: str, right: str, separation_m: float) -> bool:
        left_norm = cls._normalized_label(left)
        right_norm = cls._normalized_label(right)
        left_category = cls._object_category(left)
        right_category = cls._object_category(right)
        if left_norm == right_norm:
            return separation_m <= cls._category_merge_radius(left_category)
        if left_category and left_category == right_category:
            return separation_m <= cls._category_merge_radius(left_category)
        left_color = cls._color(left)
        right_color = cls._color(right)
        if left_color and left_color == right_color and separation_m <= 3.0:
            # Handles a single object drifting between "orange object",
            # "orange sphere" and "orange cylinder" across VLM frames.
            return True
        if left_category and right_category and left_category != right_category:
            return False
        similarity = SequenceMatcher(None, left_norm, right_norm).ratio()
        return similarity >= 0.62 and separation_m <= 3.5

    @staticmethod
    def _category_merge_radius(category: str | None) -> float:
        return {
            "sphere": 5.5,
            "cone": 4.5,
            "cube": 2.8,
            "cylinder": 3.5,
            "wall": 6.0,
            "tree": 3.5,
            "vehicle": 4.0,
            "building": 6.0,
        }.get(category, 3.5)

    def integrate_semantic(
        self,
        label: str,
        position: Vec3,
        confidence: float,
        frame_id: str,
    ) -> dict | None:
        clean_label = " ".join(label.strip().split())[:80]
        if not clean_label or not all(
            math.isfinite(value) for value in (position.x, position.y, position.z, confidence)
        ):
            return None
        if confidence < 0.35 or self._ignored_label(clean_label):
            return None
        compatible = [
            (self._planar_distance(item.position, position), item)
            for item in self._semantics
            if self._labels_compatible(
                item.label, clean_label, self._planar_distance(item.position, position)
            )
            and self._planar_distance(item.position, position)
            <= max(self.semantic_merge_m, 5.5)
        ]
        match = min(compatible, key=lambda item: item[0])[1] if compatible else None
        now = time.monotonic()
        if match:
            weight = min(match.observations, 8)
            match.position = Vec3(
                x=(match.position.x * weight + position.x) / (weight + 1),
                y=(match.position.y * weight + position.y) / (weight + 1),
                z=(match.position.z * weight + position.z) / (weight + 1),
            )
            if len(clean_label) < len(match.label) or confidence > match.confidence + 0.05:
                match.label = clean_label
            match.aliases.add(clean_label)
            match.confidence = max(match.confidence, confidence)
            match.observations += 1
            match.frame_id = frame_id
            match.last_seen = now
            landmark = match
        else:
            landmark = _SemanticLandmark(
                id=f"object-{self._next_semantic_id}",
                label=clean_label,
                position=position,
                confidence=confidence,
                observations=1,
                frame_id=frame_id,
                last_seen=now,
                aliases={clean_label},
            )
            self._next_semantic_id += 1
            self._semantics.append(landmark)
            if self._places:
                nearest = min(
                    self._places,
                    key=lambda node: self._planar_distance(
                        Vec3.model_validate(node["position"]), position
                    ),
                )
                self._edges.append(
                    {
                        "source": nearest["id"],
                        "target": landmark.id,
                        "kind": "observed",
                    }
                )
            if len(self._semantics) > self.max_semantics:
                removed = self._semantics.pop(0).id
                self._edges = [
                    edge
                    for edge in self._edges
                    if edge["source"] != removed and edge["target"] != removed
                ]
        landmark = self._coalesce_semantic_landmark(landmark)
        self._revision += 1
        return self._semantic_payload(landmark)

    def _coalesce_semantic_landmark(
        self, landmark: _SemanticLandmark
    ) -> _SemanticLandmark:
        """Merge tracks that became compatible after their filtered poses converged."""
        duplicates = [
            item
            for item in self._semantics
            if item is not landmark
            and self._labels_compatible(
                landmark.label,
                item.label,
                self._planar_distance(landmark.position, item.position),
            )
        ]
        for duplicate in duplicates:
            total = landmark.observations + duplicate.observations
            landmark.position = Vec3(
                x=(landmark.position.x * landmark.observations
                   + duplicate.position.x * duplicate.observations) / total,
                y=(landmark.position.y * landmark.observations
                   + duplicate.position.y * duplicate.observations) / total,
                z=(landmark.position.z * landmark.observations
                   + duplicate.position.z * duplicate.observations) / total,
            )
            landmark.observations = total
            landmark.confidence = max(landmark.confidence, duplicate.confidence)
            landmark.aliases.update(duplicate.aliases)
            if len(duplicate.label) < len(landmark.label):
                landmark.label = duplicate.label
            if duplicate.last_seen > landmark.last_seen:
                landmark.frame_id = duplicate.frame_id
                landmark.last_seen = duplicate.last_seen
            self._semantics.remove(duplicate)
            rewritten = []
            seen_edges: set[tuple[str, str, str]] = set()
            for edge in self._edges:
                source = landmark.id if edge["source"] == duplicate.id else edge["source"]
                target = landmark.id if edge["target"] == duplicate.id else edge["target"]
                key = (source, target, edge["kind"])
                if source != target and key not in seen_edges:
                    rewritten.append({"source": source, "target": target, "kind": edge["kind"]})
                    seen_edges.add(key)
            self._edges = rewritten
        return landmark

    def is_explored(self, position: Vec3, radius_m: float = 3.5) -> bool:
        visited = any(
            self._planar_distance(Vec3.model_validate(node["position"]), position) <= radius_m
            for node in self._places
        )
        return visited or self.exploration_coverage(position, radius_m) >= 0.35

    @staticmethod
    def _semantic_payload(item: _SemanticLandmark) -> dict:
        return {
            "id": item.id,
            "kind": "object",
            "position": item.position.model_dump(),
            "label": item.label,
            "confidence": item.confidence,
            "observations": item.observations,
            "frame_id": item.frame_id,
            "aliases": sorted(item.aliases),
        }

    def snapshot(self, *, include_tentative_semantics: bool = False) -> dict:
        # Favor repeatedly observed cells, then recent cells, while keeping the
        # WebSocket payload bounded.
        confirmed_cells = [cell for cell in self._occupancy.values() if cell.hits >= 2]
        cells = sorted(
            confirmed_cells,
            key=lambda cell: (cell.hits, cell.last_seen),
            reverse=True,
        )[:1000]
        coverage_cells = sorted(
            self._coverage.values(),
            key=lambda cell: (cell.scans, cell.last_seen),
            reverse=True,
        )[:2500]
        traversal_cells = sorted(
            self._traversal.values(),
            key=lambda cell: (cell.last_seen, cell.visits),
            reverse=True,
        )[:2500]
        confirmed_semantics = [item for item in self._semantics if item.observations >= 2]
        visible_semantics = (
            list(self._semantics) if include_tentative_semantics else confirmed_semantics
        )
        semantic_nodes = []
        for item in visible_semantics:
            payload = self._semantic_payload(item)
            payload["confirmed"] = item.observations >= 2
            semantic_nodes.append(payload)
        nodes = list(self._places) + semantic_nodes
        node_ids = {node["id"] for node in nodes}
        edges = [
            edge
            for edge in self._edges
            if edge["source"] in node_ids and edge["target"] in node_ids
        ]
        return {
            "revision": self._revision,
            "obstacles": [
                {"x": cell.x, "y": cell.y, "z": cell.z, "hits": cell.hits}
                for cell in cells
            ],
            "explored": [
                {"x": cell.x, "y": cell.y, "scans": cell.scans}
                for cell in coverage_cells
            ],
            "traversed": [
                {"x": cell.x, "y": cell.y, "visits": cell.visits}
                for cell in traversal_cells
            ],
            "coverage_cell_size_m": self.coverage_cell_size_m,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "occupancy_cells": len(confirmed_cells),
                "occupancy_tracks": len(self._occupancy),
                "explored_cells": len(self._coverage),
                "traversed_cells": len(self._traversal),
                "place_nodes": len(self._places),
                "semantic_objects": len(confirmed_semantics),
                "semantic_tracks": len(self._semantics),
            },
        }
