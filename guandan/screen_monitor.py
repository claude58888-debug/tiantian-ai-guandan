"""Real-time screen monitoring for Guandan card detection (M6).

Runs a background thread that periodically captures the game window,
detects changes via frame differencing, and fires callbacks when the
game state changes.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


# ── Events ────────────────────────────────────────────────────────────

@unique
class GameEvent(IntEnum):
    """Events detected from screen changes."""
    NEW_ROUND = 0
    CARD_PLAYED = 1
    TRIBUTE = 2
    MY_TURN = 3


@dataclass
class MonitorState:
    """Snapshot of the monitored game state between frames."""
    hand_cards: List[str] = field(default_factory=list)
    played_cards: List[str] = field(default_factory=list)
    timestamp: float = 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MonitorState):
            return NotImplemented
        return (self.hand_cards == other.hand_cards
                and self.played_cards == other.played_cards)


# ── Frame differencing ────────────────────────────────────────────────

def frames_differ(
    prev: 'np.ndarray',
    curr: 'np.ndarray',
    ssim_threshold: float = 0.95,
) -> bool:
    """Return True if *prev* and *curr* frames are meaningfully different.

    Uses SSIM (structural similarity) when available, falling back
    to normalised pixel-difference.
    """
    if not HAS_NUMPY:
        return True

    if prev.shape != curr.shape:
        return True

    if HAS_SSIM:
        # Convert to grayscale for SSIM if needed
        if prev.ndim == 3:
            prev_gray = np.mean(prev, axis=2).astype(np.uint8)
            curr_gray = np.mean(curr, axis=2).astype(np.uint8)
        else:
            prev_gray = prev
            curr_gray = curr
        score = ssim(prev_gray, curr_gray)
        return score < ssim_threshold

    # Fallback: normalised absolute difference
    diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32))
    normalised = diff.mean() / 255.0
    return normalised > 0.02


def capture_screen_mss() -> Optional['np.ndarray']:
    """Capture the full screen using mss and return as a numpy array."""
    if not HAS_MSS or not HAS_NUMPY:
        return None

    try:
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            raw = sct.grab(monitor)
            return np.array(raw)[:, :, :3]  # drop alpha channel
    except Exception:
        return None


def detect_events(
    old_state: MonitorState,
    new_state: MonitorState,
) -> List[GameEvent]:
    """Compare two monitor states and return a list of detected events."""
    events: List[GameEvent] = []

    # Cards appeared in centre → CARD_PLAYED
    if new_state.played_cards and new_state.played_cards != old_state.played_cards:
        events.append(GameEvent.CARD_PLAYED)

    # Centre cleared and hand changed → NEW_ROUND
    if not new_state.played_cards and old_state.played_cards:
        events.append(GameEvent.NEW_ROUND)

    # Hand changed (cards removed from hand) → could be MY_TURN finishing
    if (old_state.hand_cards and new_state.hand_cards
            and len(new_state.hand_cards) < len(old_state.hand_cards)):
        events.append(GameEvent.MY_TURN)

    return events


# ── Monitor class ─────────────────────────────────────────────────────

ChangeCallback = Callable[[MonitorState, MonitorState, List[GameEvent]], None]


class ScreenMonitor:
    """Background screen monitor that fires callbacks on game state changes.

    Usage::

        monitor = ScreenMonitor()
        monitor.start(callback=my_handler, fps=3)
        # ... later ...
        monitor.stop()
    """

    def __init__(self) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._prev_frame: Optional['np.ndarray'] = None
        self._prev_state = MonitorState()
        self._callback: Optional[ChangeCallback] = None
        self._fps: float = 3.0
        self._capture_func: Callable[[], Optional['np.ndarray']] = capture_screen_mss

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_state(self) -> MonitorState:
        with self._lock:
            return self._prev_state

    def set_capture_func(
        self,
        func: Callable[[], Optional['np.ndarray']],
    ) -> None:
        """Override the screen capture function (useful for testing)."""
        self._capture_func = func

    def start(
        self,
        callback: ChangeCallback,
        fps: float = 3.0,
    ) -> None:
        """Start the monitoring loop in a background daemon thread."""
        if self._running:
            return

        self._callback = callback
        self._fps = max(0.1, fps)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='screen-monitor',
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitoring loop and wait for the thread to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        """Main monitoring loop executed in the background thread."""
        interval = 1.0 / self._fps
        while self._running:
            try:
                self._tick()
            except Exception:
                pass  # swallow errors to keep the loop alive
            time.sleep(interval)

    def _tick(self) -> None:
        """Capture one frame and detect changes."""
        frame = self._capture_func()
        if frame is None:
            return

        if not HAS_NUMPY:
            return

        with self._lock:
            prev_frame = self._prev_frame

        if prev_frame is not None and not frames_differ(prev_frame, frame):
            return  # no meaningful change

        new_state = MonitorState(
            hand_cards=[],  # populated by recogniser integration
            played_cards=[],
            timestamp=time.time(),
        )

        with self._lock:
            old_state = self._prev_state
            events = detect_events(old_state, new_state)
            self._prev_state = new_state
            self._prev_frame = frame

        if self._callback is not None and events:
            self._callback(old_state, new_state, events)

    def on_change(
        self,
        old_state: MonitorState,
        new_state: MonitorState,
    ) -> List[GameEvent]:
        """Convenience method: compare two states and return events.

        Does not require the monitor to be running.
        """
        return detect_events(old_state, new_state)
