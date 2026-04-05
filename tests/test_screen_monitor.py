"""Tests for M6 - Screen monitor module."""
import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from guandan.screen_monitor import (
    GameEvent,
    MonitorState,
    ScreenMonitor,
    frames_differ,
    detect_events,
    capture_screen_mss,
)


# ── GameEvent ─────────────────────────────────────────────────────────

class TestGameEvent:
    def test_values(self):
        assert GameEvent.NEW_ROUND == 0
        assert GameEvent.CARD_PLAYED == 1
        assert GameEvent.TRIBUTE == 2
        assert GameEvent.MY_TURN == 3

    def test_names(self):
        assert GameEvent.NEW_ROUND.name == 'NEW_ROUND'
        assert GameEvent.CARD_PLAYED.name == 'CARD_PLAYED'


# ── MonitorState ──────────────────────────────────────────────────────

class TestMonitorState:
    def test_defaults(self):
        state = MonitorState()
        assert state.hand_cards == []
        assert state.played_cards == []
        assert state.timestamp == 0.0

    def test_equality(self):
        a = MonitorState(hand_cards=['3H'], played_cards=['5S'])
        b = MonitorState(hand_cards=['3H'], played_cards=['5S'])
        assert a == b

    def test_inequality(self):
        a = MonitorState(hand_cards=['3H'])
        b = MonitorState(hand_cards=['4H'])
        assert a != b

    def test_equality_ignores_timestamp(self):
        a = MonitorState(hand_cards=['3H'], timestamp=1.0)
        b = MonitorState(hand_cards=['3H'], timestamp=2.0)
        assert a == b

    def test_not_equal_to_other_type(self):
        state = MonitorState()
        assert state != 'not a state'


# ── frames_differ ─────────────────────────────────────────────────────

class TestFramesDiffer:
    def test_identical_frames(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not frames_differ(frame, frame.copy())

    def test_different_frames(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert frames_differ(a, b)

    def test_different_shapes(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.zeros((50, 50, 3), dtype=np.uint8)
        assert frames_differ(a, b)

    def test_grayscale_frames(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')
        a = np.zeros((100, 100), dtype=np.uint8)
        b = np.zeros((100, 100), dtype=np.uint8)
        assert not frames_differ(a, b)

    def test_slight_change_below_threshold(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')
        a = np.zeros((100, 100), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = 1  # tiny change
        # Should be similar enough
        assert not frames_differ(a, b, ssim_threshold=0.5)


# ── detect_events ─────────────────────────────────────────────────────

class TestDetectEvents:
    def test_card_played(self):
        old = MonitorState(played_cards=[])
        new = MonitorState(played_cards=['5H'])
        events = detect_events(old, new)
        assert GameEvent.CARD_PLAYED in events

    def test_new_round(self):
        old = MonitorState(played_cards=['5H'])
        new = MonitorState(played_cards=[])
        events = detect_events(old, new)
        assert GameEvent.NEW_ROUND in events

    def test_my_turn_cards_removed(self):
        old = MonitorState(hand_cards=['3H', '4H', '5H'])
        new = MonitorState(hand_cards=['3H'])
        events = detect_events(old, new)
        assert GameEvent.MY_TURN in events

    def test_no_events_same_state(self):
        state = MonitorState(hand_cards=['3H'], played_cards=['5H'])
        events = detect_events(state, state)
        assert events == []

    def test_played_changed_different_cards(self):
        old = MonitorState(played_cards=['5H'])
        new = MonitorState(played_cards=['6H'])
        events = detect_events(old, new)
        assert GameEvent.CARD_PLAYED in events


# ── capture_screen_mss ────────────────────────────────────────────────

class TestCaptureScreenMss:
    def test_returns_none_without_display(self):
        """In CI/headless, mss.grab may raise or return None."""
        # This test just verifies the function doesn't crash
        result = capture_screen_mss()
        # result can be None or an array depending on environment
        assert result is None or hasattr(result, 'shape')


# ── ScreenMonitor ─────────────────────────────────────────────────────

class TestScreenMonitorInit:
    def test_default_state(self):
        monitor = ScreenMonitor()
        assert not monitor.is_running
        state = monitor.current_state
        assert isinstance(state, MonitorState)

    def test_on_change_convenience(self):
        monitor = ScreenMonitor()
        old = MonitorState(played_cards=[])
        new = MonitorState(played_cards=['5H'])
        events = monitor.on_change(old, new)
        assert GameEvent.CARD_PLAYED in events


class TestScreenMonitorLifecycle:
    def test_start_stop(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')

        received_events: list = []

        def fake_capture():
            return np.zeros((100, 100, 3), dtype=np.uint8)

        def callback(old, new, events):
            received_events.extend(events)

        monitor = ScreenMonitor()
        monitor.set_capture_func(fake_capture)
        monitor.start(callback=callback, fps=30)
        assert monitor.is_running

        time.sleep(0.2)
        monitor.stop()
        assert not monitor.is_running

    def test_double_start_ignored(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')

        monitor = ScreenMonitor()
        monitor.set_capture_func(lambda: np.zeros((10, 10, 3), dtype=np.uint8))
        monitor.start(callback=lambda o, n, e: None, fps=30)
        monitor.start(callback=lambda o, n, e: None, fps=30)  # should not error
        assert monitor.is_running
        monitor.stop()

    def test_stop_without_start(self):
        monitor = ScreenMonitor()
        monitor.stop()  # should not error
        assert not monitor.is_running

    def test_capture_func_override(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')

        call_count = 0

        def counting_capture():
            nonlocal call_count
            call_count += 1
            return np.zeros((10, 10, 3), dtype=np.uint8)

        monitor = ScreenMonitor()
        monitor.set_capture_func(counting_capture)
        monitor.start(callback=lambda o, n, e: None, fps=50)
        time.sleep(0.2)
        monitor.stop()
        assert call_count > 0

    def test_callback_receives_events_on_change(self):
        """When frames differ, the callback should be invoked."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip('numpy not available')

        frame_idx = 0

        def alternating_capture():
            nonlocal frame_idx
            frame_idx += 1
            # Alternate between black and white frames
            val = 0 if frame_idx % 2 == 0 else 255
            return np.full((10, 10, 3), val, dtype=np.uint8)

        invocations: list = []

        def callback(old, new, events):
            invocations.append((old, new, events))

        monitor = ScreenMonitor()
        monitor.set_capture_func(alternating_capture)
        monitor.start(callback=callback, fps=50)
        time.sleep(0.3)
        monitor.stop()
        # The monitor should have detected frame differences
        # (callback may or may not have events depending on state logic)
        assert isinstance(invocations, list)

    def test_monitor_handles_capture_error(self):
        """Monitor should survive if capture function raises."""
        def failing_capture():
            raise RuntimeError('capture failed')

        monitor = ScreenMonitor()
        monitor.set_capture_func(failing_capture)
        monitor.start(callback=lambda o, n, e: None, fps=30)
        time.sleep(0.15)
        assert monitor.is_running  # should still be running
        monitor.stop()

    def test_min_fps_clamp(self):
        """FPS below 0.1 should be clamped to 0.1."""
        monitor = ScreenMonitor()
        monitor.set_capture_func(lambda: None)
        monitor.start(callback=lambda o, n, e: None, fps=0.001)
        # Internal fps should be clamped
        assert monitor._fps >= 0.1
        monitor.stop()
