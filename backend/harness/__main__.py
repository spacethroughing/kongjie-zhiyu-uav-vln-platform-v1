from __future__ import annotations

import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run("harness.app:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()

