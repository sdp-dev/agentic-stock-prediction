"""Client-side rate limiting for model calls.

Gemini's free tier allows only a few requests per minute per model, and one
analysis makes ~7 (three analysts in parallel, each calling the model twice,
plus the portfolio manager). Every agent runs `throttle` before each model
call, so all of them share one sliding-window budget instead of tripping
429 errors.
"""

import asyncio
import time
from collections import deque

from .config import REQUESTS_PER_MINUTE

WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self._sent: deque[float] = deque()
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        # Locks are bound to an event loop; each asyncio.run() gets a fresh one.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock, self._loop = asyncio.Lock(), loop
        return self._lock

    async def acquire(self) -> None:
        if self.requests_per_minute <= 0:
            return
        async with self._get_lock():
            while True:
                now = time.monotonic()
                while self._sent and now - self._sent[0] >= WINDOW_SECONDS:
                    self._sent.popleft()
                if len(self._sent) < self.requests_per_minute:
                    self._sent.append(now)
                    return
                await asyncio.sleep(WINDOW_SECONDS - (now - self._sent[0]) + 0.1)


limiter = RateLimiter(REQUESTS_PER_MINUTE)


async def throttle(callback_context, llm_request):
    """before_model_callback: wait for a free slot, then let the call proceed."""
    await limiter.acquire()
    return None
