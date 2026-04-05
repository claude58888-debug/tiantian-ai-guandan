"""Real-time controller for the Guandan AI decision loop (M7).

Orchestrates:
- Screen capture at 3 fps
- GameScreenAnalyzer + DecisionEngine for play decisions
- OverlayWindow for displaying recommendations
- Keyboard hotkeys (F1 start, F2 stop, F3 recalibrate)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

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
    from pynput import keyboard as pynput_keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

from guandan.card_recognition import CardRecognizer
from guandan.decision_engine import Decision, DecisionEngine, NOT_MY_TURN
from guandan.game_screen_analyzer import ScreenRegions
from guandan.models import Rank
from guandan.overlay_display import OverlayConfig, OverlayWindow

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ControllerState(Enum):
    """State of the real-time controller."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()


@dataclass
class ControllerStats:
    """Runtime statistics for the decision loop."""
    frames_processed: int = 0
    decisions_made: int = 0
    avg_latency_ms: float = 0.0
    last_decision: Optional[Decision] = None
    state: ControllerState = ControllerState.IDLE

    def record_decision(self, decision: Decision) -> None:
        self.decisions_made += 1
        self.last_decision = decision
        # Running average
        if self.avg_latency_ms == 0.0:
            self.avg_latency_ms = decision.latency_ms
        else:
            self.avg_latency_ms = (
                self.avg_latency_ms * 0.9 + decision.latency_ms * 0.1
            )


@dataclass
class GameWindowRegion:
    """Pixel coordinates of the game window on screen."""
    x: int = 0
    y: int = 0
    width: int = 1400
    height: int = 850


# ---------------------------------------------------------------------------
# Capture helper
# ---------------------------------------------------------------------------

def capture_game_window(region: GameWindowRegion) -> Optional['Image.Image']:
    """Capture the game window region using mss.

    Returns a PIL Image or None if capture fails.
    """
    if not HAS_MSS or not HAS_PIL:
        return None
    try:
        with mss.mss() as sct:
            monitor = {
                'left': region.x,
                'top': region.y,
                'width': region.width,
                'height': region.height,
            }
            grab = sct.grab(monitor)
            return Image.frombytes('RGB', grab.size, grab.rgb)
    except Exception:
        log.debug('Screen capture failed', exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class RealtimeController:
    """Main control loop for the real-time Guandan AI.

    Parameters
    ----------
    engine : DecisionEngine | None
        Decision engine instance.
    overlay : OverlayWindow | None
        Overlay display.  Created lazily if not provided.
    fps : float
        Target frames per second (default 3).
    on_decision : Callable[[Decision], None] | None
        Optional callback fired on every decision.
    enable_hotkeys : bool
        Whether to register global hotkeys (F1/F2/F3).
    capture_fn : Callable[[GameWindowRegion], Image.Image | None] | None
        Custom capture function (for testing).
    """

    def __init__(
        self,
        engine: Optional[DecisionEngine] = None,
        overlay: Optional[OverlayWindow] = None,
        fps: float = 3.0,
        on_decision: Optional[Callable[[Decision], None]] = None,
        enable_hotkeys: bool = False,
        capture_fn: Optional[Callable[[GameWindowRegion], Optional['Image.Image']]] = None,
    ) -> None:
        self._engine = engine or DecisionEngine()
        self._overlay = overlay
        self._fps = max(0.1, fps)
        self._on_decision = on_decision
        self._enable_hotkeys = enable_hotkeys
        self._capture_fn = capture_fn or capture_game_window
        self._region = GameWindowRegion()

        self._stats = ControllerStats()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._hotkey_listener: Optional[object] = None

    # -- properties --------------------------------------------------------

    @property
    def stats(self) -> ControllerStats:
        return self._stats

    @property
    def state(self) -> ControllerState:
        return self._stats.state

    @property
    def engine(self) -> DecisionEngine:
        return self._engine

    @property
    def overlay(self) -> Optional[OverlayWindow]:
        return self._overlay

    @property
    def region(self) -> GameWindowRegion:
        return self._region

    # -- control -----------------------------------------------------------

    def start(self, region: Optional[GameWindowRegion] = None) -> None:
        """Start the real-time decision loop in a background thread.

        Parameters
        ----------
        region : GameWindowRegion | None
            Game window pixel region.  Uses default if not provided.
        """
        if self._stats.state == ControllerState.RUNNING:
            log.warning('Controller already running')
            return

        if region is not None:
            self._region = region

        self._stop_event.clear()
        self._stats = ControllerStats(state=ControllerState.RUNNING)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='guandan-ai-loop',
        )
        self._thread.start()

        if self._enable_hotkeys:
            self._register_hotkeys()

        log.info('Realtime controller started (%.1f fps)', self._fps)

    def stop(self) -> None:
        """Stop the decision loop and clean up."""
        if self._stats.state not in (
            ControllerState.RUNNING,
            ControllerState.PAUSED,
        ):
            return

        self._stats.state = ControllerState.STOPPING
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._unregister_hotkeys()

        if self._overlay is not None:
            self._overlay.hide()

        self._stats.state = ControllerState.IDLE
        log.info('Realtime controller stopped')

    def pause(self) -> None:
        """Pause the loop (stops making decisions but keeps thread alive)."""
        if self._stats.state == ControllerState.RUNNING:
            self._stats.state = ControllerState.PAUSED

    def resume(self) -> None:
        """Resume from paused state."""
        if self._stats.state == ControllerState.PAUSED:
            self._stats.state = ControllerState.RUNNING

    def recalibrate(self, region: GameWindowRegion) -> None:
        """Update the game window region without stopping the loop."""
        self._region = region
        log.info('Recalibrated to %s', region)

    # -- main loop ---------------------------------------------------------

    def _loop(self) -> None:
        """Background thread: capture → analyse → decide → display."""
        interval = 1.0 / self._fps

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            if self._stats.state == ControllerState.PAUSED:
                time.sleep(interval)
                continue

            try:
                self._tick()
            except Exception:
                log.debug('Tick error', exc_info=True)

            self._stats.frames_processed += 1

            # Sleep remainder of the frame budget
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _tick(self) -> None:
        """Single frame: capture, decide, display."""
        screenshot = self._capture_fn(self._region)
        if screenshot is None:
            return

        decision = self._engine.decide(screenshot)
        self._stats.record_decision(decision)

        # Update overlay
        if decision is not NOT_MY_TURN and self._overlay is not None:
            self._overlay.show(
                text=decision.reasoning,
                confidence=decision.confidence,
                combo_type=decision.combo_type,
            )
        elif decision is NOT_MY_TURN and self._overlay is not None:
            self._overlay.hide()

        # Fire callback
        if self._on_decision is not None:
            try:
                self._on_decision(decision)
            except Exception:
                log.debug('on_decision callback error', exc_info=True)

    # -- hotkeys -----------------------------------------------------------

    def _register_hotkeys(self) -> None:
        """Register global F1/F2/F3 hotkeys via pynput."""
        if not HAS_PYNPUT:
            log.warning('pynput not available; hotkeys disabled')
            return

        def on_press(key: 'pynput_keyboard.Key') -> None:
            try:
                if key == pynput_keyboard.Key.f1:
                    self.resume()
                elif key == pynput_keyboard.Key.f2:
                    self.stop()
                elif key == pynput_keyboard.Key.f3:
                    # F3 = recalibrate (no-op here, handled by UI)
                    pass
            except Exception:
                pass

        try:
            self._hotkey_listener = pynput_keyboard.Listener(
                on_press=on_press,
            )
            self._hotkey_listener.start()  # type: ignore[union-attr]
        except Exception:
            log.warning('Failed to register hotkeys')

    def _unregister_hotkeys(self) -> None:
        """Stop the pynput hotkey listener."""
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()  # type: ignore[union-attr]
            except Exception:
                pass
            self._hotkey_listener = None
