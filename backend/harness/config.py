from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .models import SceneProfile


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    scenes_file: Path
    data_dir: Path
    runs_dir: Path
    provider: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(REPO_ROOT / ".env")
        return cls(
            host=os.getenv("HARNESS_HOST", "127.0.0.1"),
            port=int(os.getenv("HARNESS_PORT", "8000")),
            scenes_file=_resolve_repo_path(os.getenv("HARNESS_SCENES_FILE", "configs/scenes.json")),
            data_dir=_resolve_repo_path(os.getenv("HARNESS_DATA_DIR", "data")),
            runs_dir=_resolve_repo_path(os.getenv("HARNESS_RUNS_DIR", "runs")),
            provider=os.getenv("HARNESS_PROVIDER", "mock"),
            llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        )


def load_scenes(path: Path) -> dict[str, SceneProfile]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles = [SceneProfile.model_validate(item) for item in raw["scenes"]]
    result = {profile.id: profile for profile in profiles}
    if len(result) != len(profiles):
        raise ValueError("scene ids must be unique")
    return result

