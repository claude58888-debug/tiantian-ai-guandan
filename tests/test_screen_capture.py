"""Tests for Atom 2.1 - Screen capture module."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from guandan.screen_capture import (
    WindowInfo, CaptureRegion, DEFAULT_REGIONS,
    GAME_WINDOW_TITLES, GameCapture,
    find_game_window, capture_full_screen,
    capture_window, capture_region, save_screenshot,
    scale_regions,
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


# ── V0.3 H-5: Screen capture resilience tests ────────────────────────


class TestCaptureRegionIsValid:
    def test_positive_dimensions(self):
        r = CaptureRegion('test', 0, 0, 100, 50)
        assert r.is_valid

    def test_zero_width(self):
        r = CaptureRegion('test', 0, 0, 0, 50)
        assert not r.is_valid

    def test_zero_height(self):
        r = CaptureRegion('test', 0, 0, 100, 0)
        assert not r.is_valid


class TestCaptureRegionScaled:
    def test_scale_2x(self):
        r = CaptureRegion('test', 10, 20, 100, 50)
        s = r.scaled(2.0)
        assert s.x == 20
        assert s.y == 40
        assert s.w == 200
        assert s.h == 100
        assert s.name == 'test'

    def test_scale_1_5x(self):
        r = CaptureRegion('test', 10, 10, 100, 100)
        s = r.scaled(1.5)
        assert s.x == 15
        assert s.y == 15
        assert s.w == 150
        assert s.h == 150

    def test_scale_down(self):
        r = CaptureRegion('test', 100, 200, 400, 300)
        s = r.scaled(0.5)
        assert s.x == 50
        assert s.y == 100
        assert s.w == 200
        assert s.h == 150

    def test_scale_negative_raises(self):
        r = CaptureRegion('test', 10, 20, 100, 50)
        with pytest.raises(ValueError, match='positive'):
            r.scaled(-1.0)

    def test_scale_zero_raises(self):
        r = CaptureRegion('test', 10, 20, 100, 50)
        with pytest.raises(ValueError, match='positive'):
            r.scaled(0)


class TestCaptureRegionClamped:
    def test_fully_within_bounds(self):
        r = CaptureRegion('test', 10, 20, 100, 50)
        c = r.clamped(800, 600)
        assert c.x == 10 and c.y == 20 and c.w == 100 and c.h == 50

    def test_partially_off_right(self):
        r = CaptureRegion('test', 750, 0, 200, 50)
        c = r.clamped(800, 600)
        assert c.x == 750
        assert c.w == 50  # clamped from 200 to 50

    def test_partially_off_bottom(self):
        r = CaptureRegion('test', 0, 550, 100, 200)
        c = r.clamped(800, 600)
        assert c.y == 550
        assert c.h == 50

    def test_fully_off_screen(self):
        r = CaptureRegion('test', 900, 700, 100, 100)
        c = r.clamped(800, 600)
        assert c.w == 0 or c.h == 0
        assert not c.is_valid

    def test_negative_offset_clamped(self):
        r = CaptureRegion('test', -10, -20, 100, 50)
        c = r.clamped(800, 600)
        assert c.x == 0
        assert c.y == 0


class TestCaptureRegionIsWithinBounds:
    def test_within(self):
        r = CaptureRegion('test', 0, 0, 800, 600)
        assert r.is_within_bounds(800, 600)

    def test_exceeds_width(self):
        r = CaptureRegion('test', 700, 0, 200, 50)
        assert not r.is_within_bounds(800, 600)

    def test_exceeds_height(self):
        r = CaptureRegion('test', 0, 500, 100, 200)
        assert not r.is_within_bounds(800, 600)

    def test_negative_x(self):
        r = CaptureRegion('test', -1, 0, 100, 50)
        assert not r.is_within_bounds(800, 600)


class TestScaleRegions:
    def test_scales_all(self):
        regions = {
            'a': CaptureRegion('a', 10, 20, 100, 50),
            'b': CaptureRegion('b', 30, 40, 200, 100),
        }
        scaled = scale_regions(regions, 2.0)
        assert scaled['a'].w == 200
        assert scaled['b'].x == 60

    def test_preserves_names(self):
        regions = {'foo': CaptureRegion('foo', 0, 0, 100, 100)}
        scaled = scale_regions(regions, 1.5)
        assert 'foo' in scaled
        assert scaled['foo'].name == 'foo'


class TestGameCaptureDPI:
    def test_init_dpi_scale(self):
        gc = GameCapture(dpi_scale=2.0)
        assert gc.dpi_scale == 2.0

    def test_default_dpi_scale(self):
        gc = GameCapture()
        assert gc.dpi_scale == 1.0

    def test_refresh_window_no_win32(self):
        with patch('guandan.screen_capture.HAS_WIN32', False):
            gc = GameCapture()
            result = gc.refresh_window()
            assert result is None

    def test_get_scaled_region_1x(self):
        gc = GameCapture(dpi_scale=1.0)
        region = gc.get_scaled_region('my_hand')
        assert region is not None
        assert region is DEFAULT_REGIONS['my_hand']

    def test_get_scaled_region_2x(self):
        gc = GameCapture(dpi_scale=2.0)
        region = gc.get_scaled_region('my_hand')
        original = DEFAULT_REGIONS['my_hand']
        assert region is not None
        assert region.w == round(original.w * 2.0)
        assert region.h == round(original.h * 2.0)

    def test_get_scaled_region_invalid_name(self):
        gc = GameCapture(dpi_scale=2.0)
        assert gc.get_scaled_region('nonexistent') is None


class TestCaptureRegionWithDPI:
    def test_capture_region_returns_none_for_zero_region(self):
        # Region that clamps to zero after scaling
        w = WindowInfo(hwnd=1, rect=(0, 0, 100, 100))
        r = CaptureRegion('test', 200, 200, 100, 100)
        # Would be off-screen, should return None after clamping
        with patch('guandan.screen_capture.capture_window') as mock_cap:
            try:
                from PIL import Image
                mock_cap.return_value = Image.new('RGB', (100, 100))
                result = capture_region(w, r, dpi_scale=1.0)
                assert result is None
            except ImportError:
                pytest.skip('PIL not available')
