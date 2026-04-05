"""Overlay display for AI recommendations (M7).

Console-based output mode - no tkinter dependency.
Prints recommendations directly to terminal to
avoid GUI freeze issues.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverlayConfig:
    """Visual configuration for the overlay."""
    width: int = 400
    height: int = 120
    bg_alpha: float = 0.85
    font_size: int = 16
    high_color: str = "#00FF00"
    med_color: str = "#FFFF00"
    low_color: str = "#FF0000"


@dataclass
class OverlayWindow:
    """Console-based overlay that prints AI recommendations.

    This replaces the tkinter-based overlay to avoid GUI thread freezing.
    Recommendations are printed to stdout with color coding via ANSI codes.
    """

    config: OverlayConfig = field(default_factory=OverlayConfig)
    position: Tuple[int, int] = (100, 100)
    _visible: bool = False

    def show(self, text: str, confidence: float = 0.8,
             combo_type: str = "", reasoning: str = "") -> None:
        """Display a recommendation in the console."""
        self._visible = True
        color = self._confidence_color(confidence)
        conf_pct = f"{confidence * 100:.0f}%"

        # Clear line and print
        print(f"\r{' ' * 80}", end="\r")
        print(f"{color}[AI] {text} | {combo_type} | confidence: {conf_pct} | {reasoning}\033[0m")
        sys.stdout.flush()

    def hide(self) -> None:
        """Hide the overlay (no-op for console mode)."""
        self._visible = False

    def update(self) -> None:
        """Process pending events (no-op for console mode)."""
        pass

    def update_position(self, x: int, y: int) -> None:
        """Update overlay position (no-op for console mode)."""
        self.position = (x, y)

    def destroy(self) -> None:
        """Clean up (no-op for console mode)."""
        self._visible = False

    @property
    def visible(self) -> bool:
        return self._visible

    @staticmethod
    def _confidence_color(confidence: float) -> str:
        """Return ANSI color code based on confidence level."""
        if confidence >= 0.8:
            return "\033[92m"  # bright green
        elif confidence >= 0.5:
            return "\033[93m"  # bright yellow
        else:
            return "\033[91m"  # bright red

    def _ensure_window(self) -> None:
        """No-op for console mode."""
        pass
