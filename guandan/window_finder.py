"""Auto-detect game window by title (Windows via win32gui, fallback to full screen)."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# Known window title keywords for the game
GAME_TITLE_KEYWORDS = [
    "天天爱掼蛋",
    "掼蛋",
    "ttigd",
]


@dataclass
class WindowRect:
    """Rectangle describing a window's position and size on screen."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    title: str = ""
    hwnd: int = 0

    def is_valid(self) -> bool:
        return self.width > 50 and self.height > 50


def _find_windows_win32() -> List[WindowRect]:
    """Use win32gui to enumerate all visible windows (Windows only)."""
    try:
        import win32gui
    except ImportError:
        return []

    results: List[WindowRect] = []

    def enum_callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            w = x2 - x
            h = y2 - y
            if w > 50 and h > 50:
                results.append(WindowRect(
                    x=x, y=y, width=w, height=h,
                    title=title, hwnd=hwnd,
                ))
        except Exception:
            pass

    try:
        win32gui.EnumWindows(enum_callback, None)
    except Exception:
        pass
    return results


def find_game_window() -> Optional[WindowRect]:
    """Find the game window automatically.

    Searches visible windows for titles containing known game keywords.
    Returns the best match or None if not found.
    """
    windows = _find_windows_win32()
    if not windows:
        log.warning("win32gui not available or no windows found")
        return None

    for win in windows:
        title_lower = win.title.lower()
        for keyword in GAME_TITLE_KEYWORDS:
            if keyword.lower() in title_lower:
                log.info("Found game window: '%s' at (%d,%d) %dx%d",
                         win.title, win.x, win.y, win.width, win.height)
                return win

    log.warning("Game window not found. Visible windows: %s",
                [w.title for w in windows[:10]])
    return None


def find_game_window_blocking(timeout: float = 60.0, interval: float = 1.0) -> Optional[WindowRect]:
    """Wait for the game window to appear, polling every *interval* seconds.

    Returns the WindowRect once found, or None after *timeout* seconds.
    """
    import time
    print("Searching for game window (天天爱掼蛋)...")
    print("Please make sure the game is open.")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        win = find_game_window()
        if win is not None:
            print(f"Found: '{win.title}' at ({win.x},{win.y}) {win.width}x{win.height}")
            return win
        remaining = int(deadline - time.monotonic())
        print(f"\r  Waiting... ({remaining}s remaining)", end="", flush=True)
        time.sleep(interval)
    print("\nGame window not found within timeout.")
    return None


def refresh_window_rect(hwnd: int) -> Optional[WindowRect]:
    """Get the current position/size of a window by its handle.

    Used to track window movement in real-time.
    """
    try:
        import win32gui
        if not win32gui.IsWindow(hwnd):
            return None
        title = win32gui.GetWindowText(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        return WindowRect(
            x=x, y=y, width=x2 - x, height=y2 - y,
            title=title, hwnd=hwnd,
        )
    except Exception:
        return None
