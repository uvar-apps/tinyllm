"""Incremental detokenizing with stop strings.

Tokens can split UTF-8 characters, so bytes go through an incremental decoder. Stop strings are matched on the
decoded text with hold-back: text that could be the start of a stop string is kept until it is clear either way,
so a partial stop sequence is never emitted (design 4.11).
"""

from __future__ import annotations

import codecs
from collections.abc import Callable, Sequence


class Detokenizer:
    def __init__(self, token_bytes: Callable[[int], bytes], stop: Sequence[str] = ()) -> None:
        self._bytes = token_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.stop = tuple(s for s in stop if s)
        self._held = ""
        self.stopped: str | None = None  # the stop string that ended the text, once one did

    def push(self, token: int) -> str:
        """Text that can be emitted after `token`. After a stop string matched, nothing more is emitted."""
        if self.stopped is not None:
            return ""
        return self._feed(self._decoder.decode(self._bytes(token)), final=False)

    def finish(self) -> str:
        """Whatever was held back, at the end of generation."""
        if self.stopped is not None:
            return ""
        return self._feed(self._decoder.decode(b"", final=True), final=True)

    def _feed(self, text: str, final: bool) -> str:
        buf = self._held + text
        if not self.stop:
            self._held = ""
            return buf
        hits = [(buf.find(s), s) for s in self.stop if s in buf]
        if hits:
            at, stop = min(hits)
            self.stopped, self._held = stop, ""
            return buf[:at]
        if final:
            self._held = ""
            return buf
        # hold back the longest tail of `buf` that is the start of some stop string
        hold = 0
        for s in self.stop:
            for n in range(min(len(s) - 1, len(buf)), hold, -1):
                if s.startswith(buf[-n:]):
                    hold = n
                    break
        self._held = buf[len(buf) - hold :] if hold else ""
        return buf[: len(buf) - hold]
