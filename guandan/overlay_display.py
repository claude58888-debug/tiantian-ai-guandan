"""Transparent overlay window for displaying AI recommendations (M7).

Shows an always-on-top, semi-transparent window that floats over the
game client and highlights the recommended play with colour coding.

Requires tkinter (standard library) for the overlay window.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    import tkinter as tk
    HAS_TK = True
except ImportError:
    HAS_TK = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverlayConfig:
    """Visual configuration for the overlay window.

    Attributes
    ----------
    width : int
        Overlay width in pixels.
    height : int
        Overlay height in pixels.
    bg_colour : str
        Background colour (hex).
    alpha : float
        Window transparency 0.0 (invisible) to 1.0 (opaque).
    font_family : str
        Font family for text.
    font_size : int
        Font size for the main text.
    high_colour : str
        Colour for high-confidence (>= 0.8) text.
    mid_colour : str
        Colour for medium-confidence (>= 0.5) text.
    low_colour : str
        Colour for low-confidence text.
    auto_hide_ms : int
        Auto-hide the overlay after this many milliseconds (0 = no auto-hide).
    """
    width: int = 380
    height: int = 120
    bg_colour: str = '#1a1a2e'
    alpha: float = 0.85
    font_family: str = 'Helvetica'
    font_size: int = 14
    high_colour: str = '#00ff88'
    mid_colour: str = '#ffcc00'
    low_colour: str = '#ff4444'
    auto_hide_ms: int = 5000


DEFAULT_CONFIG = OverlayConfig()


def confidence_colour(confidence: float, cfg: OverlayConfig = DEFAULT_CONFIG) -> str:
    """Return a hex colour string based on confidence level."""
    if confidence >= 0.8:
        return cfg.high_colour
    if confidence >= 0.5:
        return cfg.mid_colour
    return cfg.low_colour


# ---------------------------------------------------------------------------
# Overlay window
# ---------------------------------------------------------------------------

class OverlayWindow:
    """Transparent always-on-top overlay for AI recommendations.

    Usage::

        overlay = OverlayWindow()
        overlay.show('Play 3H 3D (pair)', confidence=0.92)
        # later ...
        overlay.hide()
        overlay.destroy()

    The overlay is created lazily on first :meth:`show` call so that
    importing this module does not require a display.
    """

    def __init__(
        self,
        config: Optional[OverlayConfig] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._config = config or DEFAULT_CONFIG
        self._position = position or (50, 50)
        self._root: Optional['tk.Tk'] = None
        self._window: Optional['tk.Toplevel'] = None
        self._label: Optional['tk.Label'] = None
        self._hide_job: Optional[str] = None
        self._visible = False
        self._destroyed = False

    # -- properties --------------------------------------------------------

    @property
    def is_visible(self) -> bool:
        return self._visible

    @property
    def config(self) -> OverlayConfig:
        return self._config

    # -- lifecycle ---------------------------------------------------------

    def _ensure_window(self) -> bool:
        """Create the tkinter window if not already created.

        Returns True if the window is ready, False if tkinter is
        unavailable or display cannot be opened.
        """
        if self._destroyed:
            return False
        if not HAS_TK:
            return False
        if self._window is not None:
            return True

        try:
            self._root = tk.Tk()
            self._root.withdraw()

            self._window = tk.Toplevel(self._root)
            self._window.title('Guandan AI')
            self._window.overrideredirect(True)
            self._window.attributes('-topmost', True)

            # Transparency (platform-dependent)
            try:
                self._window.attributes('-alpha', self._config.alpha)
            except tk.TclError:
                pass

            cfg = self._config
            x, y = self._position
            self._window.geometry(f'{cfg.width}x{cfg.height}+{x}+{y}')
            self._window.configure(bg=cfg.bg_colour)

            self._label = tk.Label(
                self._window,
                text='',
                font=(cfg.font_family, cfg.font_size),
                fg=cfg.high_colour,
                bg=cfg.bg_colour,
                wraplength=cfg.width - 20,
                justify='left',
                anchor='nw',
                padx=10,
                pady=10,
            )
            self._label.pack(fill='both', expand=True)
            self._window.withdraw()
            return True
        except Exception:
            log.warning('Failed to create overlay window')
            self._window = None
            return False

    def show(
        self,
        text: str,
        confidence: float = 0.8,
        combo_type: str = '',
    ) -> None:
        """Display a recommendation on the overlay.

        Parameters
        ----------
        text : str
            Main recommendation text.
        confidence : float
            Confidence score for colour coding.
        combo_type : str
            Optional combo type prefix.
        """
        if not self._ensure_window():
            return

        cfg = self._config
        colour = confidence_colour(confidence, cfg)

        display = f'[{combo_type}] {text}' if combo_type else text
        if self._label is not None:
            self._label.configure(text=display, fg=colour)

        if self._window is not None:
            self._window.deiconify()
            self._window.lift()
        self._visible = True

        # Schedule auto-hide
        if cfg.auto_hide_ms > 0 and self._window is not None:
            if self._hide_job is not None:
                self._window.after_cancel(self._hide_job)
            self._hide_job = self._window.after(cfg.auto_hide_ms, self.hide)

    def hide(self) -> None:
        """Hide the overlay."""
        if self._window is not None:
            self._window.withdraw()
        self._visible = False

    def set_position(self, x: int, y: int) -> None:
        """Move the overlay to screen position (x, y)."""
        self._position = (x, y)
        if self._window is not None:
            self._window.geometry(
                f'{self._config.width}x{self._config.height}+{x}+{y}'
            )

    def destroy(self) -> None:
        """Destroy the overlay window and release resources."""
        self._destroyed = True
        self._visible = False
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None

    def update(self) -> None:
        """Process pending tkinter events.

        Call this from a main loop if you need non-blocking updates.
        """
        if self._root is not None:
            try:
                self._root.update_idletasks()
                self._root.update()
            except Exception:
                pass
