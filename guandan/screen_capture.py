"""Screen capture module for Guandan game (Atom 2.1).

Captures screenshots of the game window for card recognition.
Supports Windows desktop capture via win32gui/PIL or platform-agnostic
fallback using PIL.ImageGrab.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

try:
    from PIL import Image, ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import win32gui
    import win32ui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


@dataclass
class WindowInfo:
    """Information about a detected game window."""
    hwnd: int = 0
    title: str = ''
    rect: Tuple[int, int, int, int] = (0, 0, 0, 0)  # left, top, right, bottom

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    @property
    def is_valid(self) -> bool:
        return self.hwnd != 0 and self.width > 0 and self.height > 0


@dataclass
class CaptureRegion:
    """Defines a rectangular region relative to the game window."""
    name: str
    x: int  # left offset from window
    y: int  # top offset from window
    w: int  # width
    h: int  # height

    def to_bbox(self, window_left: int = 0, window_top: int = 0) -> Tuple[int, int, int, int]:
        """Convert to absolute screen coordinates (left, top, right, bottom)."""
        return (
            window_left + self.x,
            window_top + self.y,
            window_left + self.x + self.w,
            window_top + self.y + self.h,
        )

    @property
    def is_valid(self) -> bool:
        """Check if the region has positive dimensions."""
        return self.w > 0 and self.h > 0

    def scaled(self, factor: float) -> 'CaptureRegion':
        """Return a new region with coordinates scaled by *factor*.

        Useful for adapting regions designed at 1x to HiDPI displays.
        """
        if factor <= 0:
            raise ValueError(f'Scale factor must be positive, got {factor}')
        return CaptureRegion(
            name=self.name,
            x=round(self.x * factor),
            y=round(self.y * factor),
            w=round(self.w * factor),
            h=round(self.h * factor),
        )

    def clamped(self, max_width: int, max_height: int) -> 'CaptureRegion':
        """Return a new region clamped to fit within *max_width* x *max_height*.

        Handles partially off-screen regions by shrinking the region so
        that it stays within bounds.  If the region starts beyond the
        bounds the result will have zero width/height.
        """
        x = max(0, min(self.x, max_width))
        y = max(0, min(self.y, max_height))
        w = max(0, min(self.w, max_width - x))
        h = max(0, min(self.h, max_height - y))
        return CaptureRegion(name=self.name, x=x, y=y, w=w, h=h)

    def is_within_bounds(self, max_width: int, max_height: int) -> bool:
        """Check if the region is fully within the given bounds."""
        return (
            self.x >= 0 and self.y >= 0
            and self.x + self.w <= max_width
            and self.y + self.h <= max_height
        )


# Default capture regions for the Guandan game UI
# These are approximate and may need calibration per resolution
DEFAULT_REGIONS = {
    'my_hand': CaptureRegion('my_hand', 180, 520, 680, 120),
    'played_center': CaptureRegion('played_center', 300, 250, 400, 150),
    'player_left': CaptureRegion('player_left', 20, 200, 150, 200),
    'player_right': CaptureRegion('player_right', 830, 200, 150, 200),
    'player_top': CaptureRegion('player_top', 300, 30, 400, 100),
    'info_bar': CaptureRegion('info_bar', 0, 0, 1024, 30),
}


GAME_WINDOW_TITLES = [
    '\u5929\u5929\u7231\u639c\u86cb',  # "天天爱掼蛋" in unicode
    'Guandan',
    'guandan',
]


def find_game_window(custom_title: Optional[str] = None) -> Optional[WindowInfo]:
    """Find the Guandan game window by title.

    Args:
        custom_title: Override window title to search for.

    Returns:
        WindowInfo if found, None otherwise.
    """
    if not HAS_WIN32:
        return None

    titles_to_search = [custom_title] if custom_title else GAME_WINDOW_TITLES
    result = None

    def _enum_callback(hwnd: int, _extra) -> None:
        nonlocal result
        if result is not None:
            return
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        for search_title in titles_to_search:
            if search_title in title:
                rect = win32gui.GetWindowRect(hwnd)
                result = WindowInfo(hwnd=hwnd, title=title, rect=rect)
                return

    win32gui.EnumWindows(_enum_callback, None)
    return result


def capture_full_screen() -> Optional['Image.Image']:
    """Capture the entire screen."""
    if not HAS_PIL:
        return None
    return ImageGrab.grab()


def capture_window(window: WindowInfo) -> Optional['Image.Image']:
    """Capture a specific window.

    Uses win32 API for accurate window capture if available,
    falls back to screen region crop.
    """
    if not HAS_PIL:
        return None

    if not window.is_valid:
        return None

    # Try win32 capture first
    if HAS_WIN32:
        try:
            hwnd = window.hwnd
            left, top, right, bottom = window.rect
            w = right - left
            h = bottom - top

            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
            saveDC.SelectObject(saveBitMap)
            saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)

            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)

            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1
            )

            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

            return img
        except Exception:
            pass

    # Fallback: crop from full screen
    screen = capture_full_screen()
    if screen is None:
        return None
    return screen.crop(window.rect)


def capture_region(
    window: WindowInfo,
    region: CaptureRegion,
    dpi_scale: float = 1.0,
) -> Optional['Image.Image']:
    """Capture a specific region of the game window.

    If *dpi_scale* is not 1.0 the region coordinates are scaled before
    cropping, allowing regions designed at 1x to work on HiDPI displays.
    The region is always clamped to the captured image bounds so that
    partially off-screen windows don't cause errors.
    """
    full = capture_window(window)
    if full is None:
        return None

    effective = region.scaled(dpi_scale) if dpi_scale != 1.0 else region
    safe = effective.clamped(full.width, full.height)
    if not safe.is_valid:
        return None

    bbox = (safe.x, safe.y, safe.x + safe.w, safe.y + safe.h)
    return full.crop(bbox)


def save_screenshot(img: 'Image.Image', path: Path, prefix: str = 'capture') -> Path:
    """Save a screenshot with timestamp."""
    path.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    filepath = path / f'{prefix}_{ts}.png'
    img.save(filepath)
    return filepath


def scale_regions(
    regions: dict[str, CaptureRegion],
    factor: float,
) -> dict[str, CaptureRegion]:
    """Return a copy of *regions* with all coordinates scaled by *factor*."""
    return {name: r.scaled(factor) for name, r in regions.items()}


class GameCapture:
    """High-level game capture manager."""

    def __init__(
        self,
        custom_title: Optional[str] = None,
        regions: Optional[dict] = None,
        save_dir: Optional[Path] = None,
        dpi_scale: float = 1.0,
    ):
        self.custom_title = custom_title
        self.regions = regions or DEFAULT_REGIONS
        self.save_dir = save_dir or Path('screenshots')
        self.dpi_scale = dpi_scale
        self._window: Optional[WindowInfo] = None

    def find_window(self) -> Optional[WindowInfo]:
        """Find and cache game window."""
        self._window = find_game_window(self.custom_title)
        return self._window

    def refresh_window(self) -> Optional[WindowInfo]:
        """Re-detect the game window after resolution or position changes."""
        return self.find_window()

    @property
    def window(self) -> Optional[WindowInfo]:
        return self._window

    def capture(self) -> Optional['Image.Image']:
        """Capture the full game window."""
        if self._window is None:
            self.find_window()
        if self._window is None:
            return None
        return capture_window(self._window)

    def capture_region_by_name(self, name: str) -> Optional['Image.Image']:
        """Capture a named region, applying DPI scaling and boundary clamping."""
        if name not in self.regions:
            return None
        if self._window is None:
            self.find_window()
        if self._window is None:
            return None
        return capture_region(self._window, self.regions[name], self.dpi_scale)

    def get_scaled_region(self, name: str) -> Optional[CaptureRegion]:
        """Return the named region with DPI scaling applied, or None."""
        if name not in self.regions:
            return None
        region = self.regions[name]
        if self.dpi_scale != 1.0:
            return region.scaled(self.dpi_scale)
        return region

    def capture_my_hand(self) -> Optional['Image.Image']:
        return self.capture_region_by_name('my_hand')

    def capture_played_center(self) -> Optional['Image.Image']:
        return self.capture_region_by_name('played_center')

    def save_capture(self, img: 'Image.Image', prefix: str = 'capture') -> Path:
        return save_screenshot(img, self.save_dir, prefix)
