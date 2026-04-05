"""Tests for guandan.realtime_controller (M7)."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, call
from typing import List, Optional

import pytest

from guandan.card_recognition import CardRecognizer
from guandan.decision_engine import Decision, DecisionEngine, NOT_MY_TURN
from guandan.models import Card, Rank, Suit
from guandan.overlay_display import OverlayWindow
from guandan.realtime_controller import (
    ControllerState,
    ControllerStats,
    GameWindowRegion,
    RealtimeController,
    capture_game_window,
)

from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screenshot(w: int = 1400, h: int = 850) -> Image.Image:
    return Image.new('RGB', (w, h), 'black')


def _fake_capture(region: GameWindowRegion) -> Image.Image:
    """Deterministic capture function for testing."""
    return _make_screenshot(region.width, region.height)


def _fake_capture_none(region: GameWindowRegion) -> None:
    """Capture function that always fails."""
    return None


# ---------------------------------------------------------------------------
# GameWindowRegion
# ---------------------------------------------------------------------------

class TestGameWindowRegion:
    def test_defaults(self) -> None:
        r = GameWindowRegion()
        assert r.x == 0
        assert r.y == 0
        assert r.width == 1400
        assert r.height == 850

    def test_custom(self) -> None:
        r = GameWindowRegion(x=100, y=200, width=800, height=600)
        assert r.x == 100
        assert r.width == 800


# ---------------------------------------------------------------------------
# ControllerStats
# ---------------------------------------------------------------------------

class TestControllerStats:
    def test_defaults(self) -> None:
        s = ControllerStats()
        assert s.frames_processed == 0
        assert s.decisions_made == 0
        assert s.avg_latency_ms == 0.0
        assert s.last_decision is None
        assert s.state == ControllerState.IDLE

    def test_record_decision_first(self) -> None:
        s = ControllerStats()
        d = Decision(latency_ms=100.0)
        s.record_decision(d)
        assert s.decisions_made == 1
        assert s.avg_latency_ms == 100.0
        assert s.last_decision is d

    def test_record_decision_running_average(self) -> None:
        s = ControllerStats()
        s.record_decision(Decision(latency_ms=100.0))
        s.record_decision(Decision(latency_ms=200.0))
        assert s.decisions_made == 2
        # Running avg: 100*0.9 + 200*0.1 = 110
        assert abs(s.avg_latency_ms - 110.0) < 0.1


# ---------------------------------------------------------------------------
# ControllerState
# ---------------------------------------------------------------------------

class TestControllerState:
    def test_all_states_exist(self) -> None:
        assert ControllerState.IDLE is not None
        assert ControllerState.RUNNING is not None
        assert ControllerState.PAUSED is not None
        assert ControllerState.STOPPING is not None


# ---------------------------------------------------------------------------
# capture_game_window (with mocked mss)
# ---------------------------------------------------------------------------

class TestCaptureGameWindow:
    def test_no_mss_returns_none(self) -> None:
        with patch('guandan.realtime_controller.HAS_MSS', False):
            result = capture_game_window(GameWindowRegion())
            assert result is None

    def test_no_pil_returns_none(self) -> None:
        with patch('guandan.realtime_controller.HAS_PIL', False):
            result = capture_game_window(GameWindowRegion())
            assert result is None

    def test_exception_returns_none(self) -> None:
        with patch('guandan.realtime_controller.HAS_MSS', True):
            with patch('guandan.realtime_controller.HAS_PIL', True):
                with patch('guandan.realtime_controller.mss') as mock_mss:
                    mock_mss.mss.side_effect = RuntimeError('no display')
                    result = capture_game_window(GameWindowRegion())
                    assert result is None


# ---------------------------------------------------------------------------
# RealtimeController — construction
# ---------------------------------------------------------------------------

class TestControllerConstruction:
    def test_defaults(self) -> None:
        ctrl = RealtimeController()
        assert ctrl.state == ControllerState.IDLE
        assert ctrl.stats.frames_processed == 0
        assert ctrl.overlay is None

    def test_custom_engine(self) -> None:
        engine = MagicMock(spec=DecisionEngine)
        ctrl = RealtimeController(engine=engine)
        assert ctrl.engine is engine

    def test_custom_overlay(self) -> None:
        overlay = MagicMock(spec=OverlayWindow)
        ctrl = RealtimeController(overlay=overlay)
        assert ctrl.overlay is overlay

    def test_custom_fps(self) -> None:
        ctrl = RealtimeController(fps=10.0)
        assert ctrl._fps == 10.0

    def test_min_fps(self) -> None:
        ctrl = RealtimeController(fps=0.01)
        assert ctrl._fps >= 0.1


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestControllerLifecycle:
    def test_start_and_stop(self) -> None:
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        ctrl = RealtimeController(
            engine=engine,
            capture_fn=_fake_capture,
            fps=30.0,
        )
        ctrl.start()
        assert ctrl.state == ControllerState.RUNNING
        time.sleep(0.15)
        ctrl.stop()
        assert ctrl.state == ControllerState.IDLE
        assert ctrl.stats.frames_processed > 0

    def test_start_with_region(self) -> None:
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        region = GameWindowRegion(x=50, y=50, width=800, height=600)
        ctrl = RealtimeController(
            engine=engine,
            capture_fn=_fake_capture,
            fps=30.0,
        )
        ctrl.start(region=region)
        assert ctrl.region.x == 50
        ctrl.stop()

    def test_double_start_ignored(self) -> None:
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        ctrl = RealtimeController(
            engine=engine,
            capture_fn=_fake_capture,
            fps=30.0,
        )
        ctrl.start()
        ctrl.start()  # should be ignored
        ctrl.stop()

    def test_stop_when_idle(self) -> None:
        ctrl = RealtimeController()
        ctrl.stop()  # should not raise
        assert ctrl.state == ControllerState.IDLE


# ---------------------------------------------------------------------------
# pause / resume
# ---------------------------------------------------------------------------

class TestPauseResume:
    def test_pause_and_resume(self) -> None:
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        ctrl = RealtimeController(
            engine=engine,
            capture_fn=_fake_capture,
            fps=30.0,
        )
        ctrl.start()
        ctrl.pause()
        assert ctrl.state == ControllerState.PAUSED
        ctrl.resume()
        assert ctrl.state == ControllerState.RUNNING
        ctrl.stop()

    def test_pause_when_idle_no_effect(self) -> None:
        ctrl = RealtimeController()
        ctrl.pause()
        assert ctrl.state == ControllerState.IDLE

    def test_resume_when_idle_no_effect(self) -> None:
        ctrl = RealtimeController()
        ctrl.resume()
        assert ctrl.state == ControllerState.IDLE


# ---------------------------------------------------------------------------
# recalibrate
# ---------------------------------------------------------------------------

class TestRecalibrate:
    def test_recalibrate_updates_region(self) -> None:
        ctrl = RealtimeController()
        new_region = GameWindowRegion(x=200, y=100, width=1000, height=700)
        ctrl.recalibrate(new_region)
        assert ctrl.region.x == 200
        assert ctrl.region.width == 1000


# ---------------------------------------------------------------------------
# Decision callback
# ---------------------------------------------------------------------------

class TestOnDecisionCallback:
    def test_callback_fires(self) -> None:
        decisions: List[Decision] = []
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        ctrl = RealtimeController(
            engine=engine,
            capture_fn=_fake_capture,
            fps=30.0,
            on_decision=lambda d: decisions.append(d),
        )
        ctrl.start()
        time.sleep(0.15)
        ctrl.stop()
        assert len(decisions) > 0
        assert all(d is NOT_MY_TURN for d in decisions)

    def test_callback_exception_does_not_crash(self) -> None:
        def bad_callback(d: Decision) -> None:
            raise RuntimeError('boom')

        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        ctrl = RealtimeController(
            engine=engine,
            capture_fn=_fake_capture,
            fps=30.0,
            on_decision=bad_callback,
        )
        ctrl.start()
        time.sleep(0.15)
        ctrl.stop()
        # Should still have processed frames without crashing
        assert ctrl.stats.frames_processed > 0


# ---------------------------------------------------------------------------
# Overlay integration
# ---------------------------------------------------------------------------

class TestOverlayIntegration:
    def test_overlay_show_on_decision(self) -> None:
        overlay = MagicMock(spec=OverlayWindow)
        decision = Decision(
            cards_to_play=(Card(rank=Rank.ACE, suit=Suit.HEARTS),),
            combo_type='single',
            confidence=0.9,
            reasoning='Play ace',
            latency_ms=10.0,
        )
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = decision
        ctrl = RealtimeController(
            engine=engine,
            overlay=overlay,
            capture_fn=_fake_capture,
            fps=30.0,
        )
        ctrl.start()
        time.sleep(0.15)
        ctrl.stop()
        overlay.show.assert_called()

    def test_overlay_hide_when_not_my_turn(self) -> None:
        overlay = MagicMock(spec=OverlayWindow)
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        ctrl = RealtimeController(
            engine=engine,
            overlay=overlay,
            capture_fn=_fake_capture,
            fps=30.0,
        )
        ctrl.start()
        time.sleep(0.15)
        ctrl.stop()
        overlay.hide.assert_called()

    def test_overlay_hidden_on_stop(self) -> None:
        overlay = MagicMock(spec=OverlayWindow)
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        ctrl = RealtimeController(
            engine=engine,
            overlay=overlay,
            capture_fn=_fake_capture,
            fps=30.0,
        )
        ctrl.start()
        time.sleep(0.1)
        ctrl.stop()
        overlay.hide.assert_called()


# ---------------------------------------------------------------------------
# Capture failure handling
# ---------------------------------------------------------------------------

class TestCaptureFailure:
    def test_none_capture_no_crash(self) -> None:
        engine = MagicMock(spec=DecisionEngine)
        ctrl = RealtimeController(
            engine=engine,
            capture_fn=_fake_capture_none,
            fps=30.0,
        )
        ctrl.start()
        time.sleep(0.15)
        ctrl.stop()
        # Engine should never be called if capture returns None
        engine.decide.assert_not_called()
        assert ctrl.stats.frames_processed > 0


# ---------------------------------------------------------------------------
# Hotkeys
# ---------------------------------------------------------------------------

class TestHotkeys:
    def test_hotkeys_disabled_by_default(self) -> None:
        ctrl = RealtimeController()
        assert ctrl._enable_hotkeys is False
        assert ctrl._hotkey_listener is None

    def test_hotkeys_no_pynput(self) -> None:
        engine = MagicMock(spec=DecisionEngine)
        engine.decide.return_value = NOT_MY_TURN
        ctrl = RealtimeController(
            engine=engine,
            capture_fn=_fake_capture,
            fps=30.0,
            enable_hotkeys=True,
        )
        with patch('guandan.realtime_controller.HAS_PYNPUT', False):
            ctrl.start()
            time.sleep(0.1)
            ctrl.stop()
            assert ctrl._hotkey_listener is None


# ---------------------------------------------------------------------------
# Main.py integration
# ---------------------------------------------------------------------------

class TestMainModule:
    def test_build_parser(self) -> None:
        from guandan.main import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        assert args.realtime is False
        assert args.level == Rank.TWO
        assert args.fps == 3.0

    def test_parse_realtime_flag(self) -> None:
        from guandan.main import build_parser
        parser = build_parser()
        args = parser.parse_args(['--realtime'])
        assert args.realtime is True

    def test_parse_level(self) -> None:
        from guandan.main import build_parser
        parser = build_parser()
        args = parser.parse_args(['--level', 'A'])
        assert args.level == Rank.ACE

    def test_parse_fps(self) -> None:
        from guandan.main import build_parser
        parser = build_parser()
        args = parser.parse_args(['--fps', '5'])
        assert args.fps == 5.0

    def test_parse_hotkeys(self) -> None:
        from guandan.main import build_parser
        parser = build_parser()
        args = parser.parse_args(['--hotkeys'])
        assert args.hotkeys is True

    def test_parse_verbose(self) -> None:
        from guandan.main import build_parser
        parser = build_parser()
        args = parser.parse_args(['-v'])
        assert args.verbose is True

    def test_invalid_level_exits(self) -> None:
        from guandan.main import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--level', 'Z'])

    def test_main_cli_mode(self) -> None:
        from guandan.main import main
        with patch('guandan.main.run_realtime') as mock_rt:
            with patch('guandan.cli.main') as mock_cli:
                main(['--verbose'])
                mock_cli.assert_called_once()
                mock_rt.assert_not_called()

    def test_main_realtime_mode(self) -> None:
        from guandan.main import main
        with patch('guandan.main.run_realtime') as mock_rt:
            main(['--realtime', '--level', '3', '--fps', '5'])
            mock_rt.assert_called_once_with(
                level=Rank.THREE,
                fps=5.0,
                enable_hotkeys=False,
            )
