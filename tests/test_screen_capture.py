"""Tests for Atom 2.1 - Screen capture module."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from guandan.screen_capture import (
    WindowInfo, CaptureRegion, DEFAULT_REGIONS,
    GAME_WINDOW_TITLES, GameCapture,
    find_game_window, capture_full_screen,
    capture_window, capture_region, save_screenshot,
)


class TestWindowInfo:
    def test_defaults(self):
        w = WindowInfo()
        assert w.hwnd == 0
        assert w.width == 0
        assert w.height == 0
        assert not w.is_valid

    def test_valid_window(self):
        w = WindowInfo(hwnd=123, title='test', rect=(0, 0, 800, 600))
        assert w.width == 800
        assert w.height == 600
        assert w.is_valid

    def test_invalid_zero_size(self):
        w = WindowInfo(hwnd=1, rect=(100, 100, 100, 100))
        assert not w.is_valid


class TestCaptureRegion:
    def test_to_bbox_no_offset(self):
        r = CaptureRegion('test', 10, 20, 100, 50)
        assert r.to_bbox() == (10, 20, 110, 70)

    def test_to_bbox_with_offset(self):
        r = CaptureRegion('test', 10, 20, 100, 50)
        assert r.to_bbox(50, 30) == (60, 50, 160, 100)


class TestDefaultRegions:
    def test_regions_exist(self):
        assert 'my_hand' in DEFAULT_REGIONS
        assert 'played_center' in DEFAULT_REGIONS
        assert 'player_left' in DEFAULT_REGIONS
        assert 'player_right' in DEFAULT_REGIONS
        assert 'player_top' in DEFAULT_REGIONS
        assert 'info_bar' in DEFAULT_REGIONS

    def test_region_types(self):
        for name, region in DEFAULT_REGIONS.items():
            assert isinstance(region, CaptureRegion)
            assert region.name == name
            assert region.w > 0
            assert region.h > 0


class TestGameWindowTitles:
    def test_titles_not_empty(self):
        assert len(GAME_WINDOW_TITLES) > 0


class TestFindGameWindow:
    def test_no_win32_returns_none(self):
        with patch('guandan.screen_capture.HAS_WIN32', False):
            result = find_game_window()
            assert result is None

    def test_custom_title(self):
        with patch('guandan.screen_capture.HAS_WIN32', False):
            result = find_game_window('Custom Title')
            assert result is None


class TestCaptureFullScreen:
    def test_no_pil_returns_none(self):
        with patch('guandan.screen_capture.HAS_PIL', False):
            result = capture_full_screen()
            assert result is None


class TestCaptureWindow:
    def test_no_pil_returns_none(self):
        with patch('guandan.screen_capture.HAS_PIL', False):
            w = WindowInfo(hwnd=1, rect=(0, 0, 800, 600))
            result = capture_window(w)
            assert result is None

    def test_invalid_window_returns_none(self):
        w = WindowInfo()  # invalid
        result = capture_window(w)
        assert result is None


class TestCaptureRegionFunc:
    def test_returns_none_when_window_capture_fails(self):
        w = WindowInfo()  # invalid
        r = CaptureRegion('test', 0, 0, 100, 100)
        result = capture_region(w, r)
        assert result is None


class TestSaveScreenshot:
    def test_save_creates_file(self, tmp_path):
        try:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            result = save_screenshot(img, tmp_path, 'test')
            assert result.exists()
            assert result.suffix == '.png'
            assert 'test_' in result.name
        except ImportError:
            pytest.skip('PIL not available')


class TestGameCapture:
    def test_init_defaults(self):
        gc = GameCapture()
        assert gc.custom_title is None
        assert gc.regions == DEFAULT_REGIONS
        assert gc.window is None

    def test_init_custom(self):
        gc = GameCapture(custom_title='MyGame', save_dir=Path('/tmp/test'))
        assert gc.custom_title == 'MyGame'
        assert gc.save_dir == Path('/tmp/test')

    def test_find_window_no_win32(self):
        with patch('guandan.screen_capture.HAS_WIN32', False):
            gc = GameCapture()
            result = gc.find_window()
            assert result is None

    def test_capture_no_window(self):
        with patch('guandan.screen_capture.HAS_WIN32', False):
            gc = GameCapture()
            result = gc.capture()
            assert result is None

    def test_capture_region_invalid_name(self):
        gc = GameCapture()
        result = gc.capture_region_by_name('nonexistent')
        assert result is None

    def test_capture_my_hand_no_window(self):
        with patch('guandan.screen_capture.HAS_WIN32', False):
            gc = GameCapture()
            result = gc.capture_my_hand()
            assert result is None

    def test_capture_played_center_no_window(self):
        with patch('guandan.screen_capture.HAS_WIN32', False):
            gc = GameCapture()
            result = gc.capture_played_center()
            assert result is None

    def test_save_capture(self, tmp_path):
        try:
            from PIL import Image
            gc = GameCapture(save_dir=tmp_path)
            img = Image.new('RGB', (100, 100))
            result = gc.save_capture(img, 'test')
            assert result.exists()
        except ImportError:
            pytest.skip('PIL not available')
