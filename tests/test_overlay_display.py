"""Tests for guandan.overlay_display (M7 - console mode)."""
from __future__ import annotations

import pytest

from guandan.overlay_display import (
    OverlayConfig,
    OverlayWindow,
)


# ---------------------------------------------------------------------------
# OverlayConfig
# ---------------------------------------------------------------------------

class TestOverlayConfig:
    def test_defaults(self) -> None:
        cfg = OverlayConfig()
        assert cfg.width == 400
        assert cfg.height == 120
        assert cfg.bg_alpha == 0.85
        assert cfg.font_size == 16

    def test_custom_config(self) -> None:
        cfg = OverlayConfig(width=500, height=200, bg_alpha=0.5)
        assert cfg.width == 500
        assert cfg.height == 200
        assert cfg.bg_alpha == 0.5

    def test_frozen(self) -> None:
        cfg = OverlayConfig()
        with pytest.raises(AttributeError):
            cfg.width = 999  # type: ignore[misc]

    def test_colours_are_hex(self) -> None:
        cfg = OverlayConfig()
        assert cfg.high_color.startswith('#')
        assert cfg.med_color.startswith('#')
        assert cfg.low_color.startswith('#')

    def test_default_colors(self) -> None:
        cfg = OverlayConfig()
        assert cfg.high_color == "#00FF00"
        assert cfg.med_color == "#FFFF00"
        assert cfg.low_color == "#FF0000"


# ---------------------------------------------------------------------------
# OverlayWindow
# ---------------------------------------------------------------------------

class TestOverlayWindow:
    def test_initial_state(self) -> None:
        overlay = OverlayWindow()
        assert overlay.visible is False

    def test_default_config(self) -> None:
        overlay = OverlayWindow()
        assert isinstance(overlay.config, OverlayConfig)
        assert overlay.config.width == 400

    def test_custom_config(self) -> None:
        cfg = OverlayConfig(width=600)
        overlay = OverlayWindow(config=cfg)
        assert overlay.config.width == 600

    def test_custom_position(self) -> None:
        overlay = OverlayWindow(position=(200, 300))
        assert overlay.position == (200, 300)

    def test_show_sets_visible(self, capsys) -> None:
        overlay = OverlayWindow()
        overlay.show('Play pair', confidence=0.9)
        assert overlay.visible is True
        captured = capsys.readouterr()
        assert 'Play pair' in captured.out

    def test_show_includes_combo_type(self, capsys) -> None:
        overlay = OverlayWindow()
        overlay.show('Strong play', confidence=0.8, combo_type='bomb')
        captured = capsys.readouterr()
        assert 'bomb' in captured.out

    def test_show_includes_reasoning(self, capsys) -> None:
        overlay = OverlayWindow()
        overlay.show('Play', confidence=0.7, reasoning='best option')
        captured = capsys.readouterr()
        assert 'best option' in captured.out

    def test_show_includes_confidence_pct(self, capsys) -> None:
        overlay = OverlayWindow()
        overlay.show('Play', confidence=0.85)
        captured = capsys.readouterr()
        assert '85%' in captured.out

    def test_hide(self) -> None:
        overlay = OverlayWindow()
        overlay.show('test')
        overlay.hide()
        assert overlay.visible is False

    def test_update_position(self) -> None:
        overlay = OverlayWindow()
        overlay.update_position(500, 600)
        assert overlay.position == (500, 600)

    def test_destroy(self) -> None:
        overlay = OverlayWindow()
        overlay.show('test')
        overlay.destroy()
        assert overlay.visible is False

    def test_destroy_idempotent(self) -> None:
        overlay = OverlayWindow()
        overlay.destroy()
        overlay.destroy()  # should not raise
        assert overlay.visible is False

    def test_hide_when_not_shown(self) -> None:
        overlay = OverlayWindow()
        overlay.hide()  # should not raise
        assert overlay.visible is False


# ---------------------------------------------------------------------------
# _confidence_color
# ---------------------------------------------------------------------------

class TestConfidenceColor:
    def test_high_confidence(self) -> None:
        overlay = OverlayWindow()
        color = overlay._confidence_color(0.9)
        assert '92m' in color  # bright green

    def test_high_boundary(self) -> None:
        overlay = OverlayWindow()
        color = overlay._confidence_color(0.8)
        assert '92m' in color

    def test_mid_confidence(self) -> None:
        overlay = OverlayWindow()
        color = overlay._confidence_color(0.6)
        assert '93m' in color  # bright yellow

    def test_mid_boundary(self) -> None:
        overlay = OverlayWindow()
        color = overlay._confidence_color(0.5)
        assert '93m' in color

    def test_low_confidence(self) -> None:
        overlay = OverlayWindow()
        color = overlay._confidence_color(0.3)
        assert '91m' in color  # bright red

    def test_zero_confidence(self) -> None:
        overlay = OverlayWindow()
        color = overlay._confidence_color(0.0)
        assert '91m' in color
