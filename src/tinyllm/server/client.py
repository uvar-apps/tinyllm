"""`EngineClient`: the engine's message boundary, as seen from asyncio code (design 4.1, decision D5)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from tinyllm.engine.core import Engine, EngineRequest, Event, FinishEvent


class QueueFull(Exception):
    """The engine's queue holds as many requests as it may."""


class EngineClient:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def generate(self, request: EngineRequest) -> AsyncIterator[Event]:
        """The request's events, ending with its FinishEvent. Leaving early cancels the request."""
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[Event] = asyncio.Queue()

        def sink(event: Event) -> None:  # called on the engine thread
            with contextlib.suppress(RuntimeError):  # the event loop is gone (shutdown): nobody is listening
                loop.call_soon_threadsafe(events.put_nowait, event)

        if not self.engine.submit(request, sink):
            raise QueueFull
        finished = False
        try:
            while True:
                event = await events.get()
                if isinstance(event, FinishEvent):
                    finished = True
                yield event
                if finished:
                    return
        finally:
            if not finished:
                self.engine.cancel(request.id)

    def cancel(self, request_id: str) -> None:
        self.engine.cancel(request_id)
