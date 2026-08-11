from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from .models import EventEnvelope, MissionPlan, RunRecord, RunState, utc_now


class Store:
    def __init__(self, database_path: Path, runs_dir: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = runs_dir
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version(version)
                  SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
                CREATE TABLE IF NOT EXISTS plans(
                  id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  approved_at TEXT,
                  payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs(
                  id TEXT PRIMARY KEY,
                  plan_id TEXT NOT NULL,
                  state TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  ended_at TEXT,
                  error TEXT,
                  artifact_dir TEXT NOT NULL,
                  payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events(
                  sequence INTEGER PRIMARY KEY,
                  run_id TEXT,
                  topic TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  payload TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save_plan(self, plan: MissionPlan) -> None:
        payload = plan.model_dump_json()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO plans(id, created_at, approved_at, payload) VALUES(?,?,?,?)",
                (
                    plan.id,
                    plan.created_at.isoformat(),
                    plan.approved_at.isoformat() if plan.approved_at else None,
                    payload,
                ),
            )

    def get_plan(self, plan_id: str) -> MissionPlan | None:
        with self._lock:
            row = self._connection.execute("SELECT payload FROM plans WHERE id=?", (plan_id,)).fetchone()
        return MissionPlan.model_validate_json(row["payload"]) if row else None

    def create_run(self, plan_id: str) -> RunRecord:
        temporary = RunRecord(plan_id=plan_id, artifact_dir="")
        artifact_dir = self.runs_dir / temporary.id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        run = temporary.model_copy(update={"artifact_dir": str(artifact_dir)})
        self.save_run(run)
        return run

    def save_run(self, run: RunRecord) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT OR REPLACE INTO runs
                (id, plan_id, state, started_at, ended_at, error, artifact_dir, payload)
                VALUES(?,?,?,?,?,?,?,?)""",
                (
                    run.id,
                    run.plan_id,
                    run.state.value,
                    run.started_at.isoformat(),
                    run.ended_at.isoformat() if run.ended_at else None,
                    run.error,
                    run.artifact_dir,
                    run.model_dump_json(),
                ),
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row["payload"]) if row else None

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RunRecord.model_validate_json(row["payload"]) for row in rows]

    def update_run(
        self,
        run: RunRecord,
        state: RunState,
        *,
        error: str | None = None,
        ended: bool = False,
        target_position=None,
    ) -> RunRecord:
        updates = {"state": state, "error": error}
        if ended:
            updates["ended_at"] = utc_now()
        if target_position is not None:
            updates["target_position"] = target_position
        updated = run.model_copy(update=updates)
        self.save_run(updated)
        return updated

    def append_event(self, event: EventEnvelope) -> None:
        persisted = event
        if event.topic == "frame.preview" and "data_url" in event.payload:
            # The live WebSocket already received the image. Persist only its
            # frame reference: embedding every PNG again in events.jsonl made
            # active logs huge and increased Windows read/write lock windows.
            payload = dict(event.payload)
            payload.pop("data_url", None)
            payload.pop("depth_data_url", None)
            payload["artifact"] = f"frames/{payload.get('frame_id')}.png"
            persisted = event.model_copy(update={"payload": payload})
        elif event.topic == "lidar.points" and "points" in event.payload:
            # Raw point arrays are transient visualization data. Keep only
            # scan metadata in replay logs to prevent multi-megabyte event
            # files while the full occupancy result remains in the map export.
            payload = dict(event.payload)
            payload.pop("points", None)
            persisted = event.model_copy(update={"payload": payload})
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO events(sequence, run_id, topic, timestamp, payload) VALUES(?,?,?,?,?)",
                (
                    persisted.sequence,
                    persisted.run_id,
                    persisted.topic,
                    persisted.timestamp.isoformat(),
                    json.dumps(persisted.payload),
                ),
            )
        if persisted.run_id:
            run = self.get_run(persisted.run_id)
            if run:
                self.append_jsonl(
                    Path(run.artifact_dir) / "events.jsonl",
                    persisted.model_dump(mode="json"),
                )

    @staticmethod
    def append_jsonl(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        for attempt in range(101):
            try:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                return
            except PermissionError:
                if attempt == 100:
                    raise
                # Windows readers can briefly deny a concurrent append. Keep
                # the safety controller alive while a report/artifact reader
                # releases its handle instead of turning logging into a flight
                # failure.
                time.sleep(0.02)

    def list_artifacts(self, run_id: str) -> list[str]:
        run = self.get_run(run_id)
        if not run:
            return []
        root = Path(run.artifact_dir)
        return [str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()]

    def write_manifest(self, run: RunRecord, payload: dict) -> None:
        path = Path(run.artifact_dir) / "manifest.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_topology_map(self, run: RunRecord, payload: dict) -> None:
        path = Path(run.artifact_dir) / "topology_map.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_report(self, run: RunRecord, plan: MissionPlan) -> None:
        report = {
            "run": run.model_dump(mode="json"),
            "mission": plan.model_dump(mode="json"),
            "generated_at": datetime.now().astimezone().isoformat(),
        }
        root = Path(run.artifact_dir)
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown = (
            f"# AirSim 任务报告\n\n"
            f"- Run ID: `{run.id}`\n"
            f"- 状态: **{run.state.value}**\n"
            f"- 目标: {plan.request.target_text}\n"
            f"- 场景: `{plan.request.scene_id}` / `{plan.request.zone_id}`\n"
            f"- 开始: {run.started_at.isoformat()}\n"
            f"- 结束: {run.ended_at.isoformat() if run.ended_at else '-'}\n"
            f"- 错误: {run.error or '-'}\n"
        )
        (root / "report.md").write_text(markdown, encoding="utf-8")
