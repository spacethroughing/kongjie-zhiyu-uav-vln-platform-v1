import json
from pathlib import Path

from harness.models import EventEnvelope
from harness.store import Store


def test_append_jsonl_retries_a_transient_windows_share_violation(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "events.jsonl"
    real_open = Path.open
    attempts = 0

    def flaky_open(self, *args, **kwargs):
        nonlocal attempts
        if self == path and attempts < 2:
            attempts += 1
            raise PermissionError(13, "simulated sharing violation", str(path))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    Store.append_jsonl(path, {"ok": True})
    assert attempts == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_frame_preview_persists_reference_without_duplicate_base64(tmp_path: Path):
    store = Store(tmp_path / "data.sqlite3", tmp_path / "runs")
    try:
        event = EventEnvelope(
            topic="frame.preview",
            run_id=None,
            sequence=1,
            payload={
                "frame_id": "frame-1",
                "data_url": "data:image/png;base64,large-payload",
                "width": 640,
                "height": 360,
            },
        )
        store.append_event(event)
        row = store._connection.execute(
            "SELECT payload FROM events WHERE sequence=1"
        ).fetchone()
        payload = json.loads(row["payload"])
        assert "data_url" not in payload
        assert payload["artifact"] == "frames/frame-1.png"
    finally:
        store.close()


def test_lidar_event_persists_scan_metadata_without_raw_point_array(tmp_path: Path):
    store = Store(tmp_path / "data.sqlite3", tmp_path / "runs")
    try:
        event = EventEnvelope(
            topic="lidar.points",
            run_id=None,
            sequence=2,
            payload={
                "data_frame": "VehicleInertialFrame",
                "point_count": 2400,
                "sampled_point_count": 900,
                "points": [[1, 2, 3]] * 900,
            },
        )
        store.append_event(event)
        row = store._connection.execute(
            "SELECT payload FROM events WHERE sequence=2"
        ).fetchone()
        payload = json.loads(row["payload"])
        assert "points" not in payload
        assert payload["point_count"] == 2400
        assert payload["sampled_point_count"] == 900
    finally:
        store.close()
