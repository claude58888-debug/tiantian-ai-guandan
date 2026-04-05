"""Tests for guandan.overlay_display (M7)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from guandan.overlay_display import (
    DEFAULT_CONFIG,
    OverlayConfig,
    OverlayWindow,
    confidence_colour,
)


# ---------------------------------------------------------------------------
# OverlayConfig
# ---------------------------------------------------------------------------

class TestOverlayConfig:
    def test_defaults(self) -> None:
        cfg = DEFAULT_CONFIG
        assert cfg.width == 380
        assert cfg.height == 120
        assert cfg.alpha == 0.85
        assert cfg.auto_hide_ms == 5000

    def test_custom_config(self) -> None:
        cfg = OverlayConfig(width=500, height=200, alpha=0.5)
        assert cfg.width == 500
        assert cfg.height == 200
        assert cfg.alpha == 0.5

    def test_frozen(self) -> None:
        cfg = DEFAULT_CONFIG
        with pytest.raises(AttributeError):
            cfg.width = 999  # type: ignore[misc]

    def test_colours_are_hex(self) -> None:
        cfg = DEFAULT_CONFIG
        assert cfg.high_colour.startswith('#')
        assert cfg.mid_colour.startswith('#')
        assert cfg.low_colour.startswith('#')


# ---------------------------------------------------------------------------
# confidence_colour
# ---------------------------------------------------------------------------

class TestConfidenceColour:
    def test_high_confidence(self) -> None:
        assert confidence_colour(0.9) == DEFAULT_CONFIG.high_colour

    def test_high_boundary(self) -> None:
        assert confidence_colour(0.8) == DEFAULT_CONFIG.high_colour

    def test_mid_confidence(self) -> None:
        assert confidence_colour(0.6) == DEFAULT_CONFIG.mid_colour

    def test_mid_boundary(self) -> None:
        assert confidence_colour(0.5) == DEFAULT_CONFIG.mid_colour

    def test_low_confidence(self) -> None:
        assert confidence_colour(0.3) == DEFAULT_CONFIG.low_colour

    def test_zero_confidence(self) -> None:
        assert confidence_colour(0.0) == DEFAULT_CONFIG.low_colour

    def test_custom_config(self) -> None:
        cfg = OverlayConfig(high_colour='#aaa', mid_colour='#bbb', low_colour='#ccc')
        assert confidence_colour(0.9, cfg) == '#aaa'
        assert confidence_colour(0.6, cfg) == '#bbb'
        assert confidence_colour(0.2, cfg) == '#ccc'


# ---------------------------------------------------------------------------
# OverlayWindow — no-display tests (tkinter mocked)
# ---------------------------------------------------------------------------

class TestOverlayWindowNoDisplay:
    """Tests that work without a real display by mocking tkinter."""

    def test_initial_state(self) -> None:
        overlay = OverlayWindow()
        assert overlay.is_visible is False
        assert overlay.config is DEFAULT_CONFIG

    def test_custom_config(self) -> None:
        cfg = OverlayConfig(width=600)
        overlay = OverlayWindow(config=cfg)
        assert overlay.config.width == 600

    def test_custom_position(self) -> None:
        overlay = OverlayWindow(position=(100, 200))
        assert overlay._position == (100, 200)

    def test_destroy_idempotent(self) -> None:
        overlay = OverlayWindow()
        overlay.destroy()
        overlay.destroy()  # should not raise
        assert overlay.is_visible is False

    def test_hide_when_not_created(self) -> None:
        overlay = OverlayWindow()
        overlay.hide()  # should not raise
        assert overlay.is_visible is False

    def test_update_when_not_created(self) -> None:
        overlay = OverlayWindow()
        overlay.update()  # should not raise

    def test_show_without_tk_does_nothing(self) -> None:
        overlay = OverlayWindow()
        with patch('guandan.overlay_display.HAS_TK', False):
            overlay.show('test', confidence=0.9)
            assert overlay.is_visible is False

    def test_show_after_destroy_does_nothing(self) -> None:
        overlay = OverlayWindow()
        overlay.destroy()
        overlay.show('test', confidence=0.9)
        assert overlay.is_visible is False

    def test_set_position_before_window(self) -> None:
        overlay = OverlayWindow()
        overlay.set_position(300, 400)
        assert overlay._position == (300, 400)


# ---------------------------------------------------------------------------
# OverlayWindow — with mocked tkinter window
# ---------------------------------------------------------------------------

@patch('guandan.overlay_display.HAS_TK', True)
class TestOverlayWindowMocked:
    """Tests with a mocked tkinter backend to verify logic flow.

    The class-level ``@patch`` ensures ``HAS_TK`` is ``True`` for
    every test so that ``_ensure_window`` finds the pre-set mock
    objects and returns ``True``.
    """

    @staticmethod
    def _make_overlay_with_mock_tk() -> OverlayWindow:
        overlay = OverlayWindow()
        # Pre-create mocked window objects so _ensure_window returns True
        mock_root = MagicMock()
        mock_window = MagicMock()
        mock_label = MagicMock()
        mock_window.after.return_value = 'job_id'
        overlay._root = mock_root
        overlay._window = mock_window
        overlay._label = mock_label
        return overlay

    def test_show_sets_visible(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('Play pair', confidence=0.9)
        assert overlay.is_visible is True

    def test_show_updates_label(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('Play pair', confidence=0.9, combo_type='pair')
        overlay._label.configure.assert_called()
        call_kwargs = overlay._label.configure.call_args
        assert 'pair' in call_kwargs[1]['text']

    def test_show_with_combo_type_prefix(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('Strong play', confidence=0.5, combo_type='bomb')
        call_kwargs = overlay._label.configure.call_args
        assert '[bomb]' in call_kwargs[1]['text']

    def test_hide_sets_invisible(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('test')
        overlay.hide()
        assert overlay.is_visible is False
        overlay._window.withdraw.assert_called()

    def test_set_position_updates_geometry(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.set_position(500, 600)
        overlay._window.geometry.assert_called()

    def test_auto_hide_scheduled(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('test', confidence=0.9)
        overlay._window.after.assert_called()

    def test_auto_hide_cancelled_on_new_show(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('first')
        overlay._window.after.return_value = 'job2'
        overlay.show('second')
        overlay._window.after_cancel.assert_called()

    def test_destroy_cleans_up(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('test')
        overlay.destroy()
        assert overlay.is_visible is False
        assert overlay._window is None
        assert overlay._root is None

    def test_update_processes_events(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.update()
        overlay._root.update_idletasks.assert_called()
        overlay._root.update.assert_called()

    def test_high_confidence_green(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('test', confidence=0.95)
        call_kwargs = overlay._label.configure.call_args
        assert call_kwargs[1]['fg'] == DEFAULT_CONFIG.high_colour

    def test_mid_confidence_yellow(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('test', confidence=0.6)
        call_kwargs = overlay._label.configure.call_args
        assert call_kwargs[1]['fg'] == DEFAULT_CONFIG.mid_colour

    def test_low_confidence_red(self) -> None:
        overlay = self._make_overlay_with_mock_tk()
        overlay.show('test', confidence=0.2)
        call_kwargs = overlay._label.configure.call_args
        assert call_kwargs[1]['fg'] == DEFAULT_CONFIG.low_colour

    def test_no_auto_hide_when_disabled(self) -> None:
        cfg = OverlayConfig(auto_hide_ms=0)
        overlay = OverlayWindow(config=cfg)
        mock_window = MagicMock()
        mock_label = MagicMock()
        overlay._root = MagicMock()
        overlay._window = mock_window
        overlay._label = mock_label
        overlay.show('test')
        mock_window.after.assert_not_called()
