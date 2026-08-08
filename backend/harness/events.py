from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .models import EventEnvelope


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[EventEnvelope]] = set()
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, payload: dict, run_id: str | None = None) -> EventEnvelope:
        async with self._lock:
            self._sequence += 1
            event = EventEnvelope(topic=topic, payload=payload, run_id=run_id, sequence=self._sequence)
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)
        return event

    async def subscribe(self) -> AsyncIterator[EventEnvelope]:
        queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

