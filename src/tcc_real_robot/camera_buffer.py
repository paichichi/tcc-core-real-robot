"""Background latest-frame buffering for policy camera sources."""

from __future__ import annotations

import threading
import time
from types import TracebackType
from typing import Any, Self

import numpy as np


class LatestFramePairBuffer:
    """Continuously capture pairs and deliver the newest unseen pair.

    The wrapped object remains responsible for camera synchronization and frame
    validation. This class only overlaps capture with policy inference/actuation.
    """

    def __init__(self, source_context: Any, *, timeout_s: float = 2.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._source_context = source_context
        self._source: Any | None = None
        self._timeout_s = timeout_s
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: tuple[np.ndarray, np.ndarray] | None = None
        self._latest_skew_ms: float | None = None
        self._sequence = 0
        self._delivered_sequence = 0
        self._error: BaseException | None = None
        self._captured_pairs = 0
        self._superseded_pairs = 0

    def __enter__(self) -> Self:
        self._source = self._source_context.__enter__()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="policy-camera-buffer",
            daemon=True,
        )
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        assert self._source is not None
        try:
            while not self._stop.is_set():
                pair = self._source.read_rgb_pair()
                skew_ms = self._source.last_pair_skew_ms
                if skew_ms is None:
                    raise RuntimeError("Camera source did not record pair skew")
                with self._condition:
                    self._latest = pair
                    self._latest_skew_ms = float(skew_ms)
                    self._sequence += 1
                    self._captured_pairs += 1
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001 - propagate worker failures
            with self._condition:
                if not self._stop.is_set():
                    self._error = exc
                self._condition.notify_all()

    def read_rgb_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the newest pair, waiting only when no unseen pair exists."""
        deadline = time.monotonic() + self._timeout_s
        with self._condition:
            while self._sequence <= self._delivered_sequence and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a fresh camera pair")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("Background camera capture failed") from self._error
            if self._latest is None or self._latest_skew_ms is None:
                raise RuntimeError("Camera buffer has no frame pair")
            self._superseded_pairs += max(
                0, self._sequence - self._delivered_sequence - 1
            )
            self._delivered_sequence = self._sequence
            return self._latest

    @property
    def last_pair_skew_ms(self) -> float | None:
        return self._latest_skew_ms

    @property
    def captured_pairs(self) -> int:
        return self._captured_pairs

    @property
    def dropped_pairs(self) -> int:
        return self._superseded_pairs

    @property
    def main_properties(self) -> dict[str, object]:
        if self._source is None:
            raise RuntimeError("Camera buffer is not running")
        return self._source.main_properties

    @property
    def wrist_properties(self) -> dict[str, object]:
        if self._source is None:
            raise RuntimeError("Camera buffer is not running")
        return self._source.wrist_properties

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_s)
            self._thread = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
        self._source_context.__exit__(exc_type, exc, traceback)
        self._source = None
