"""Tests for guandan.calibration (M8)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from guandan.calibration import (
    DEFAULT_BLACK_RANGE,
    DEFAULT_RED_RANGE,
    CalibrationData,
    CalibrationManager,
    HSVRange,
)
from guandan.game_screen_analyzer import ScreenRegions
from guandan.models import Card, Rank, Suit


# ---------------------------------------------------------------------------
# HSVRange
# ---------------------------------------------------------------------------

class TestHSVRange:
    def test_defaults(self) -> None:
        r = HSVRange()
        assert r.h_low == 0
        assert r.h_high == 180
        assert r.s_high == 255
        assert r.v_high == 255

    def test_lower_upper(self) -> None:
        r = HSVRange(h_low=10, s_low=20, v_low=30, h_high=40, s_high=50, v_high=60)
        assert r.lower() == (10, 20, 30)
        assert r.upper() == (40, 50, 60)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        r = HSVRange(h_low=5, s_low=10, v_low=15, h_high=170, s_high=200, v_high=220)
        d = r.to_dict()
        restored = HSVRange.from_dict(d)
        assert restored == r

    def test_frozen(self) -> None:
        r = HSVRange()
        with pytest.raises(AttributeError):
            r.h_low = 99  # type: ignore[misc]


class TestDefaultRanges:
    def test_default_red_range(self) -> None:
        assert DEFAULT_RED_RANGE.h_low == 0
        assert DEFAULT_RED_RANGE.s_low == 80

    def test_default_black_range(self) -> None:
        assert DEFAULT_BLACK_RANGE.v_high == 80


# ---------------------------------------------------------------------------
# CalibrationData
# ---------------------------------------------------------------------------

class TestCalibrationData:
    def test_defaults(self) -> None:
        cd = CalibrationData()
        assert cd.template_scale == 1.0
        assert cd.match_threshold == 0.8
        assert cd.source_resolution == (1400, 850)
        assert cd.screen_regions == {}
        assert cd.color_ranges == {}

    def test_to_screen_regions_empty(self) -> None:
        cd = CalibrationData()
        sr = cd.to_screen_regions()
        assert isinstance(sr, ScreenRegions)
        assert sr.hand_top == 0.60  # default

    def test_to_screen_regions_custom(self) -> None:
        cd = CalibrationData(screen_regions={'hand_top': 0.55, 'hand_bottom': 0.95})
        sr = cd.to_screen_regions()
        assert sr.hand_top == 0.55
        assert sr.hand_bottom == 0.95

    def test_get_red_range_default(self) -> None:
        cd = CalibrationData()
        assert cd.get_red_range() == DEFAULT_RED_RANGE

    def test_get_red_range_custom(self) -> None:
        custom = HSVRange(h_low=1, s_low=2, v_low=3, h_high=4, s_high=5, v_high=6)
        cd = CalibrationData(color_ranges={'red': custom.to_dict()})
        assert cd.get_red_range() == custom

    def test_get_black_range_default(self) -> None:
        cd = CalibrationData()
        assert cd.get_black_range() == DEFAULT_BLACK_RANGE

    def test_get_black_range_custom(self) -> None:
        custom = HSVRange(h_low=10, s_low=20, v_low=30, h_high=40, s_high=50, v_high=60)
        cd = CalibrationData(color_ranges={'black': custom.to_dict()})
        assert cd.get_black_range() == custom

    def test_to_dict(self) -> None:
        cd = CalibrationData(
            template_scale=0.9,
            match_threshold=0.75,
            source_resolution=(1920, 1080),
        )
        d = cd.to_dict()
        assert d['template_scale'] == 0.9
        assert d['match_threshold'] == 0.75
        assert d['source_resolution'] == [1920, 1080]

    def test_from_dict(self) -> None:
        raw = {
            'screen_regions': {'hand_top': 0.5},
            'template_scale': 1.1,
            'match_threshold': 0.7,
            'color_ranges': {},
            'source_resolution': [1400, 850],
        }
        cd = CalibrationData.from_dict(raw)
        assert cd.template_scale == 1.1
        assert cd.match_threshold == 0.7
        assert cd.screen_regions == {'hand_top': 0.5}
        assert cd.source_resolution == (1400, 850)

    def test_roundtrip(self) -> None:
        cd = CalibrationData(
            screen_regions={'hand_top': 0.55},
            template_scale=0.85,
            match_threshold=0.72,
            color_ranges={'red': DEFAULT_RED_RANGE.to_dict()},
            source_resolution=(1600, 900),
        )
        restored = CalibrationData.from_dict(cd.to_dict())
        assert restored.template_scale == cd.template_scale
        assert restored.match_threshold == cd.match_threshold
        assert restored.source_resolution == cd.source_resolution
        assert restored.screen_regions == cd.screen_regions

    def test_from_dict_missing_keys(self) -> None:
        cd = CalibrationData.from_dict({})
        assert cd.template_scale == 1.0
        assert cd.match_threshold == 0.8


# ---------------------------------------------------------------------------
# CalibrationManager
# ---------------------------------------------------------------------------

class TestCalibrationManager:
    def test_construction(self) -> None:
        cm = CalibrationManager()
        assert isinstance(cm.data, CalibrationData)

    def test_construction_with_recognizer(self) -> None:
        mock_rec = MagicMock()
        cm = CalibrationManager(recognizer=mock_rec)
        assert cm.data.template_scale == 1.0


class TestCalibrateFromScreenshot:
    @patch('guandan.calibration.HAS_CV2', False)
    def test_no_cv2_returns_default(self) -> None:
        cm = CalibrationManager()
        img = MagicMock()
        result = cm.calibrate_from_screenshot(img, [])
        assert result.match_threshold == 0.8

    @patch('guandan.calibration.HAS_PIL', False)
    def test_no_pil_returns_default(self) -> None:
        cm = CalibrationManager()
        img = MagicMock()
        result = cm.calibrate_from_screenshot(img, [])
        assert result.match_threshold == 0.8

    @patch('guandan.calibration.HAS_CV2', True)
    @patch('guandan.calibration.HAS_PIL', True)
    def test_sweeps_parameters(self) -> None:
        mock_rec = MagicMock()
        # Return a detection for the ace of hearts at all thresholds
        ace_h = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        mock_det = MagicMock()
        mock_det.card.display.return_value = 'A\u2665'
        mock_rec.find_cards_in_region.return_value = [mock_det]

        cm = CalibrationManager(recognizer=mock_rec)
        mock_img = MagicMock()
        mock_img.size = (1400, 850)

        result = cm.calibrate_from_screenshot(mock_img, [ace_h])
        assert result.source_resolution == (1400, 850)
        assert result.match_threshold in {0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9}
        assert 'red' in result.color_ranges
        assert 'black' in result.color_ranges

    @patch('guandan.calibration.HAS_CV2', True)
    @patch('guandan.calibration.HAS_PIL', True)
    def test_empty_hand_label(self) -> None:
        mock_rec = MagicMock()
        mock_rec.find_cards_in_region.return_value = []
        cm = CalibrationManager(recognizer=mock_rec)
        mock_img = MagicMock()
        mock_img.size = (1400, 850)
        result = cm.calibrate_from_screenshot(mock_img, [])
        assert result.match_threshold in {0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9}


class TestAutoDetectRegions:
    @patch('guandan.calibration.HAS_CV2', False)
    def test_no_cv2_returns_default(self) -> None:
        cm = CalibrationManager()
        regions = cm.auto_detect_regions(MagicMock())
        assert isinstance(regions, ScreenRegions)
        assert regions.hand_top == 0.60

    @patch('guandan.calibration.HAS_CV2', True)
    @patch('guandan.calibration.HAS_PIL', True)
    def test_detects_regions(self) -> None:
        cm = CalibrationManager()
        # Create a simple synthetic image
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip('PIL/numpy required')

        img = Image.new('RGB', (1400, 850), (40, 40, 40))
        # Draw bright pixels in bottom 40% to simulate hand cards
        arr = np.array(img)
        arr[510:850, 100:1300] = (255, 255, 255)
        img = Image.fromarray(arr)

        regions = cm.auto_detect_regions(img)
        assert isinstance(regions, ScreenRegions)
        # hand_top should be clamped between 0.50 and 0.75
        assert 0.50 <= regions.hand_top <= 0.75


class TestGenerateGameTemplates:
    @patch('guandan.calibration.HAS_CV2', False)
    def test_no_cv2_returns_empty(self) -> None:
        cm = CalibrationManager()
        result = cm.generate_game_templates(MagicMock())
        assert result == {}

    @patch('guandan.calibration.HAS_CV2', True)
    @patch('guandan.calibration.HAS_PIL', True)
    def test_extracts_from_image(self) -> None:
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip('PIL/numpy required')

        # Create image with some vertical edges in bottom 40%
        img = Image.new('RGB', (400, 200), (200, 200, 200))
        arr = np.array(img)
        # Simulate card edges at x=50, x=90, x=130
        for x in [50, 90, 130]:
            arr[120:200, x, :] = 0  # dark vertical line
        img = Image.fromarray(arr)

        cm = CalibrationManager()
        templates = cm.generate_game_templates(img)
        assert isinstance(templates, dict)
        # May or may not find templates depending on edge detection
        # but should not crash


class TestPersistence:
    def test_save_and_load(self) -> None:
        cm = CalibrationManager()
        cm._data = CalibrationData(
            template_scale=0.9,
            match_threshold=0.72,
            source_resolution=(1920, 1080),
            screen_regions={'hand_top': 0.55},
            color_ranges={'red': DEFAULT_RED_RANGE.to_dict()},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'cal.json'
            cm.save_calibration(path)
            assert path.exists()

            cm2 = CalibrationManager()
            loaded = cm2.load_calibration(path)
            assert loaded.template_scale == 0.9
            assert loaded.match_threshold == 0.72
            assert loaded.source_resolution == (1920, 1080)
            assert loaded.screen_regions == {'hand_top': 0.55}

    def test_save_creates_parent_dirs(self) -> None:
        cm = CalibrationManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'subdir' / 'deep' / 'cal.json'
            cm.save_calibration(path)
            assert path.exists()

    def test_load_invalid_json_raises(self) -> None:
        cm = CalibrationManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'bad.json'
            path.write_text('not json')
            with pytest.raises(json.JSONDecodeError):
                cm.load_calibration(path)

    def test_load_nonexistent_raises(self) -> None:
        cm = CalibrationManager()
        with pytest.raises(FileNotFoundError):
            cm.load_calibration(Path('/nonexistent/path.json'))
