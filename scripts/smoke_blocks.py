from __future__ import annotations

import json
import urllib.error
import urllib.request


BASE = "http://127.0.0.1:8000/api"


def request(method: str, path: str, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=payload, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=130) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8", errors="replace")) from error


def main() -> None:
    print(request("POST", "/simulator/start", {"scene_id": "blocks"}))
    try:
        print(request("POST", "/simulator/smoke"))
    finally:
        print(request("POST", "/simulator/stop"))


if __name__ == "__main__":
    main()
