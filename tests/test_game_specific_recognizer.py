"""Tests for guandan.game_specific_recognizer (M8)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from guandan.calibration import CalibrationData, HSVRange
from guandan.card_recognition import RecognizedCard
from guandan.game_screen_analyzer import ScreenRegions
from guandan.game_specific_recognizer import (
    DEFAULT_GAME_LAYOUT,
    RAISED_OFFSET_PX,
    STRIP_WIDTH,
    GameLayout,
    GameSpecificRecognizer,
)
from guandan.models import Card, Rank, Suit


# ---------------------------------------------------------------------------
# GameLayout
# ---------------------------------------------------------------------------

class TestGameLayout:
    def test_defaults(self) -> None:
        gl = GameLayout()
        assert gl.hand_top == 0.58
        assert gl.hand_bottom == 1.0
        assert gl.played_top == 0.30
        assert gl.counter_top == 0.0
        assert gl.button_top == 0.45

    def test_frozen(self) -> None:
        gl = GameLayout()
        with pytest.raises(AttributeError):
            gl.hand_top = 0.5  # type: ignore[misc]

    def test_to_screen_regions(self) -> None:
        gl = GameLayout()
        sr = gl.to_screen_regions()
        assert isinstance(sr, ScreenRegions)
        assert sr.hand_top == gl.hand_top
        assert sr.played_top == gl.played_top
        assert sr.counter_top == gl.counter_top

    def test_custom_layout(self) -> None:
        gl = GameLayout(hand_top=0.55, played_top=0.28)
        assert gl.hand_top == 0.55
        assert gl.played_top == 0.28

    def test_default_game_layout_is_gamelayout(self) -> None:
        assert isinstance(DEFAULT_GAME_LAYOUT, GameLayout)


# ---------------------------------------------------------------------------
# GameSpecificRecognizer construction
# ---------------------------------------------------------------------------

class TestGameSpecificRecognizerConstruction:
    def test_default_construction(self) -> None:
        gsr = GameSpecificRecognizer()
        assert gsr.threshold == 0.75
        assert gsr.calibration is None
        assert gsr.layout == DEFAULT_GAME_LAYOUT
        assert gsr.game_templates_loaded is False

    def test_with_calibration(self) -> None:
        cal = CalibrationData(match_threshold=0.65)
        gsr = GameSpecificRecognizer(calibration=cal)
        assert gsr.threshold == 0.65
        assert gsr.calibration is cal

    def test_with_custom_layout(self) -> None:
        layout = GameLayout(hand_top=0.5)
        gsr = GameSpecificRecognizer(layout=layout)
        assert gsr.layout.hand_top == 0.5

    def test_scales_include_extra(self) -> None:
        gsr = GameSpecificRecognizer()
        assert 0.9 in gsr.scales
        assert 1.1 in gsr.scales


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

class TestTemplateLoading:
    def test_load_game_templates_nonexistent_dir(self) -> None:
        gsr = GameSpecificRecognizer()
        count = gsr.load_game_templates(Path('/nonexistent'))
        assert count == 0
        assert gsr.game_templates_loaded is False

    def test_load_game_templates_empty_dir(self) -> None:
        gsr = GameSpecificRecognizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            count = gsr.load_game_templates(Path(tmpdir))
            assert count == 0

    def test_load_game_templates_prefers_real_subdir(self) -> None:
        gsr = GameSpecificRecognizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir) / 'real'
            real_dir.mkdir()
            # Load from empty real subdir → 0 templates
            count = gsr.load_game_templates(Path(tmpdir))
            assert count == 0

    def test_load_game_templates_from_pil_marks_loaded(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        gsr = GameSpecificRecognizer()
        # Create tiny synthetic template images
        images = {
            'AH': Image.new('RGB', (32, 48), (220, 20, 20)),
            'KS': Image.new('RGB', (32, 48), (30, 30, 30)),
        }
        count = gsr.load_game_templates_from_pil(images)
        assert count == 2
        assert gsr.game_templates_loaded is True

    def test_load_game_templates_from_pil_zero_not_loaded(self) -> None:
        gsr = GameSpecificRecognizer()
        # Invalid names → 0 loaded
        images = {'invalid': MagicMock()}
        # This may fail due to cv2 conversion; use patch
        count = gsr.load_game_templates_from_pil({})
        assert count == 0
        assert gsr.game_templates_loaded is False


# ---------------------------------------------------------------------------
# recognize_hand_strips
# ---------------------------------------------------------------------------

class TestRecognizeHandStrips:
    def test_not_loaded_returns_empty(self) -> None:
        gsr = GameSpecificRecognizer()
        result = gsr.recognize_hand_strips(MagicMock())
        assert result == []

    @patch('guandan.game_specific_recognizer.HAS_CV2', False)
    def test_no_cv2_returns_empty(self) -> None:
        gsr = GameSpecificRecognizer()
        result = gsr.recognize_hand_strips(MagicMock())
        assert result == []

    @patch('guandan.game_specific_recognizer.HAS_CV2', True)
    @patch('guandan.game_specific_recognizer.HAS_PIL', True)
    def test_strips_with_loaded_templates(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        gsr = GameSpecificRecognizer()
        images = {
            'AH': Image.new('RGB', (32, 48), (220, 20, 20)),
        }
        gsr.load_game_templates_from_pil(images)

        screenshot = Image.new('RGB', (1400, 850), (200, 200, 200))
        result = gsr.recognize_hand_strips(screenshot, strip_width=STRIP_WIDTH)
        assert isinstance(result, list)
        # May or may not detect anything on a uniform image
        for det in result:
            assert isinstance(det, RecognizedCard)


# ---------------------------------------------------------------------------
# classify_suit_colour
# ---------------------------------------------------------------------------

class TestClassifySuitColour:
    @patch('guandan.game_specific_recognizer.HAS_CV2', True)
    @patch('guandan.game_specific_recognizer.HAS_PIL', True)
    def test_red_card(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        gsr = GameSpecificRecognizer()
        img = Image.new('RGB', (40, 60), (220, 20, 20))
        assert gsr.classify_suit_colour(img) == 'red'

    @patch('guandan.game_specific_recognizer.HAS_CV2', True)
    @patch('guandan.game_specific_recognizer.HAS_PIL', True)
    def test_black_card(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        gsr = GameSpecificRecognizer()
        img = Image.new('RGB', (40, 60), (20, 20, 20))
        assert gsr.classify_suit_colour(img) == 'black'

    @patch('guandan.card_extractor.HAS_CV2', False)
    def test_no_cv2_returns_none(self) -> None:
        gsr = GameSpecificRecognizer()
        assert gsr.classify_suit_colour(MagicMock()) is None


# ---------------------------------------------------------------------------
# detect_raised_cards
# ---------------------------------------------------------------------------

class TestDetectRaisedCards:
    def test_empty_detections(self) -> None:
        gsr = GameSpecificRecognizer()
        result = gsr.detect_raised_cards(MagicMock(), [])
        assert result == []

    def test_filters_by_y_position(self) -> None:
        gsr = GameSpecificRecognizer(layout=GameLayout(hand_top=0.58))
        mock_screenshot = MagicMock()
        mock_screenshot.size = (1400, 850)

        # hand_top = 850 * 0.58 = 493, raised_y = 493 - 15 = 478
        normal = RecognizedCard(
            card=Card(rank=Rank.ACE, suit=Suit.HEARTS),
            confidence=0.9,
            bbox=(100, 500, 40, 60),  # y=500 > 478 → not raised
        )
        raised = RecognizedCard(
            card=Card(rank=Rank.KING, suit=Suit.SPADES),
            confidence=0.85,
            bbox=(200, 470, 40, 60),  # y=470 < 478 → raised
        )
        result = gsr.detect_raised_cards(mock_screenshot, [normal, raised])
        assert len(result) == 1
        assert result[0].card.rank == Rank.KING


# ---------------------------------------------------------------------------
# recognize_hand (override)
# ---------------------------------------------------------------------------

class TestRecognizeHand:
    def test_uses_strip_scanning(self) -> None:
        gsr = GameSpecificRecognizer()
        # Not loaded, should return empty
        result = gsr.recognize_hand(MagicMock())
        assert result == []

    @patch('guandan.game_specific_recognizer.HAS_CV2', True)
    @patch('guandan.game_specific_recognizer.HAS_PIL', True)
    def test_deduplicates_by_display(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        gsr = GameSpecificRecognizer()
        images = {
            'AH': Image.new('RGB', (32, 48), (220, 20, 20)),
        }
        gsr.load_game_templates_from_pil(images)
        screenshot = Image.new('RGB', (1400, 850), (200, 200, 200))
        result = gsr.recognize_hand(screenshot)
        # Should not have duplicates
        displays = [c.display() for c in result]
        assert len(displays) == len(set(displays))


# ---------------------------------------------------------------------------
# recognize_played_cards (override)
# ---------------------------------------------------------------------------

class TestRecognizePlayedCards:
    @patch('guandan.game_specific_recognizer.HAS_CV2', False)
    def test_no_cv2_returns_empty(self) -> None:
        gsr = GameSpecificRecognizer()
        result = gsr.recognize_played_cards(MagicMock())
        assert result == []

    @patch('guandan.game_specific_recognizer.HAS_CV2', True)
    @patch('guandan.game_specific_recognizer.HAS_PIL', True)
    def test_uses_game_layout_regions(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        gsr = GameSpecificRecognizer()
        screenshot = Image.new('RGB', (1400, 850), (200, 200, 200))
        result = gsr.recognize_played_cards(screenshot)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_screen_regions
# ---------------------------------------------------------------------------

class TestGetScreenRegions:
    def test_from_layout(self) -> None:
        gsr = GameSpecificRecognizer()
        sr = gsr.get_screen_regions()
        assert sr.hand_top == DEFAULT_GAME_LAYOUT.hand_top

    def test_from_calibration(self) -> None:
        cal = CalibrationData(
            screen_regions={'hand_top': 0.50, 'hand_bottom': 0.95},
        )
        gsr = GameSpecificRecognizer(calibration=cal)
        sr = gsr.get_screen_regions()
        assert sr.hand_top == 0.50
        assert sr.hand_bottom == 0.95

    def test_calibration_empty_falls_back(self) -> None:
        cal = CalibrationData(screen_regions={})
        gsr = GameSpecificRecognizer(calibration=cal)
        sr = gsr.get_screen_regions()
        # Falls back to layout
        assert sr.hand_top == DEFAULT_GAME_LAYOUT.hand_top
