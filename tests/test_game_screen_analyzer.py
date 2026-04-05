"""Tests for guandan.game_screen_analyzer (M7)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional

import pytest

from guandan.models import Card, Rank, Suit, JokerType
from guandan.card_recognition import CardRecognizer, RecognizedCard
from guandan.game_screen_analyzer import (
    DEFAULT_REGIONS,
    GameScreenAnalyzer,
    ScreenRegions,
    _crop_region,
    _LABEL_TO_RANK,
)

# We always have PIL available in the test environment
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screenshot(width: int = 1400, height: int = 850, color: str = 'black') -> Image.Image:
    """Create a synthetic screenshot of the given size."""
    return Image.new('RGB', (width, height), color)


def _make_recognizer_stub(cards: Optional[List[Card]] = None) -> CardRecognizer:
    """Return a CardRecognizer whose find_cards_in_region returns *cards*."""
    rec = MagicMock(spec=CardRecognizer)
    if cards is None:
        cards = []
    detections = [
        RecognizedCard(card=c, confidence=0.9, bbox=(0, 0, 30, 40))
        for c in cards
    ]
    rec.find_cards_in_region.return_value = detections
    return rec


# ---------------------------------------------------------------------------
# ScreenRegions tests
# ---------------------------------------------------------------------------

class TestScreenRegions:
    def test_default_regions_are_valid(self) -> None:
        r = DEFAULT_REGIONS
        assert 0.0 <= r.hand_top < r.hand_bottom <= 1.0
        assert 0.0 <= r.played_top < r.played_bottom <= 1.0
        assert 0.0 <= r.button_top < r.button_bottom <= 1.0
        assert 0.0 <= r.counter_top < r.counter_bottom <= 1.0
        assert 0.0 <= r.level_top < r.level_bottom <= 1.0

    def test_custom_regions(self) -> None:
        r = ScreenRegions(hand_top=0.5, hand_bottom=0.9)
        assert r.hand_top == 0.5
        assert r.hand_bottom == 0.9

    def test_frozen(self) -> None:
        r = DEFAULT_REGIONS
        with pytest.raises(AttributeError):
            r.hand_top = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _crop_region tests
# ---------------------------------------------------------------------------

class TestCropRegion:
    def test_full_crop(self) -> None:
        img = _make_screenshot(100, 100)
        cropped = _crop_region(img, 0.0, 1.0, 0.0, 1.0)
        assert cropped.size == (100, 100)

    def test_partial_crop(self) -> None:
        img = _make_screenshot(200, 100)
        cropped = _crop_region(img, 0.5, 1.0, 0.25, 0.75)
        assert cropped.size == (100, 50)

    def test_zero_size_crop(self) -> None:
        img = _make_screenshot(100, 100)
        cropped = _crop_region(img, 0.5, 0.5, 0.0, 0.0)
        assert cropped.size[0] == 0 or cropped.size[1] == 0


# ---------------------------------------------------------------------------
# Label-to-rank mapping
# ---------------------------------------------------------------------------

class TestLabelToRank:
    def test_all_ranks_mapped(self) -> None:
        assert _LABEL_TO_RANK['A'] == Rank.ACE
        assert _LABEL_TO_RANK['K'] == Rank.KING
        assert _LABEL_TO_RANK['2'] == Rank.TWO
        assert _LABEL_TO_RANK['10'] == Rank.TEN

    def test_count(self) -> None:
        assert len(_LABEL_TO_RANK) == 13


# ---------------------------------------------------------------------------
# GameScreenAnalyzer construction
# ---------------------------------------------------------------------------

class TestAnalyzerConstruction:
    def test_default_construction(self) -> None:
        analyzer = GameScreenAnalyzer()
        assert analyzer.regions is DEFAULT_REGIONS
        assert analyzer.recognizer is not None

    def test_custom_recognizer(self) -> None:
        rec = _make_recognizer_stub()
        analyzer = GameScreenAnalyzer(recognizer=rec)
        assert analyzer.recognizer is rec

    def test_custom_regions(self) -> None:
        r = ScreenRegions(hand_top=0.7)
        analyzer = GameScreenAnalyzer(regions=r)
        assert analyzer.regions.hand_top == 0.7

    def test_set_regions(self) -> None:
        analyzer = GameScreenAnalyzer()
        new_r = ScreenRegions(hand_top=0.8)
        analyzer.set_regions(new_r)
        assert analyzer.regions.hand_top == 0.8


# ---------------------------------------------------------------------------
# detect_my_turn
# ---------------------------------------------------------------------------

class TestDetectMyTurn:
    def test_no_cv2_returns_false(self) -> None:
        analyzer = GameScreenAnalyzer()
        screenshot = _make_screenshot()
        with patch('guandan.game_screen_analyzer.HAS_CV2', False):
            assert analyzer.detect_my_turn(screenshot) is False

    def test_black_screenshot_not_my_turn(self) -> None:
        """A fully black screenshot has no golden buttons."""
        analyzer = GameScreenAnalyzer()
        screenshot = _make_screenshot(color='black')
        result = analyzer.detect_my_turn(screenshot)
        assert result is False

    def test_golden_region_detected(self) -> None:
        """An image with a golden region in the button area should trigger."""
        img = _make_screenshot(1400, 850, color='black')
        # Paint golden pixels in the button region
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Button region: 30-70% x, 55-75% y
        x1 = int(1400 * 0.35)
        y1 = int(850 * 0.58)
        x2 = int(1400 * 0.65)
        y2 = int(850 * 0.72)
        draw.rectangle([x1, y1, x2, y2], fill=(220, 180, 50))
        analyzer = GameScreenAnalyzer()
        result = analyzer.detect_my_turn(img)
        assert result is True


# ---------------------------------------------------------------------------
# detect_hand_cards
# ---------------------------------------------------------------------------

class TestDetectHandCards:
    def test_returns_cards_from_recognizer(self) -> None:
        cards = [
            Card(rank=Rank.THREE, suit=Suit.HEARTS),
            Card(rank=Rank.FIVE, suit=Suit.SPADES),
        ]
        rec = _make_recognizer_stub(cards)
        analyzer = GameScreenAnalyzer(recognizer=rec)
        screenshot = _make_screenshot()
        result = analyzer.detect_hand_cards(screenshot)
        assert len(result) == 2
        assert result[0].rank == Rank.THREE
        assert result[1].rank == Rank.FIVE

    def test_deduplicates_cards(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.CLUBS)
        rec = MagicMock(spec=CardRecognizer)
        rec.find_cards_in_region.return_value = [
            RecognizedCard(card=card, confidence=0.9, bbox=(0, 0, 30, 40)),
            RecognizedCard(card=card, confidence=0.85, bbox=(10, 0, 30, 40)),
        ]
        analyzer = GameScreenAnalyzer(recognizer=rec)
        result = analyzer.detect_hand_cards(_make_screenshot())
        assert len(result) == 1

    def test_empty_hand(self) -> None:
        rec = _make_recognizer_stub([])
        analyzer = GameScreenAnalyzer(recognizer=rec)
        result = analyzer.detect_hand_cards(_make_screenshot())
        assert result == []


# ---------------------------------------------------------------------------
# detect_played_cards
# ---------------------------------------------------------------------------

class TestDetectPlayedCards:
    def test_returns_none_when_empty(self) -> None:
        rec = _make_recognizer_stub([])
        analyzer = GameScreenAnalyzer(recognizer=rec)
        result = analyzer.detect_played_cards(_make_screenshot())
        assert result is None

    def test_returns_cards(self) -> None:
        cards = [Card(rank=Rank.KING, suit=Suit.DIAMONDS)]
        rec = _make_recognizer_stub(cards)
        analyzer = GameScreenAnalyzer(recognizer=rec)
        result = analyzer.detect_played_cards(_make_screenshot())
        assert result is not None
        assert len(result) == 1
        assert result[0].rank == Rank.KING

    def test_deduplicates(self) -> None:
        card = Card(rank=Rank.TEN, suit=Suit.HEARTS)
        rec = MagicMock(spec=CardRecognizer)
        rec.find_cards_in_region.return_value = [
            RecognizedCard(card=card, confidence=0.9, bbox=(0, 0, 30, 40)),
            RecognizedCard(card=card, confidence=0.8, bbox=(5, 0, 30, 40)),
        ]
        analyzer = GameScreenAnalyzer(recognizer=rec)
        result = analyzer.detect_played_cards(_make_screenshot())
        assert result is not None
        assert len(result) == 1


# ---------------------------------------------------------------------------
# detect_card_counter
# ---------------------------------------------------------------------------

class TestDetectCardCounter:
    def test_no_easyocr_returns_empty(self) -> None:
        analyzer = GameScreenAnalyzer()
        with patch('guandan.game_screen_analyzer.HAS_EASYOCR', False):
            result = analyzer.detect_card_counter(_make_screenshot())
            assert result == {}

    def test_no_cv2_returns_empty(self) -> None:
        analyzer = GameScreenAnalyzer()
        with patch('guandan.game_screen_analyzer.HAS_CV2', False):
            result = analyzer.detect_card_counter(_make_screenshot())
            assert result == {}

    def test_ocr_init_failure_returns_empty(self) -> None:
        analyzer = GameScreenAnalyzer()
        with patch('guandan.game_screen_analyzer.HAS_EASYOCR', True):
            with patch.object(analyzer, '_get_ocr', return_value=None):
                result = analyzer.detect_card_counter(_make_screenshot())
                assert result == {}

    def test_ocr_returns_rank_labels(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [20, 0], [20, 20], [0, 20]], 'A', 0.9),
            ([[30, 0], [50, 0], [50, 20], [30, 20]], 'K', 0.85),
        ]
        analyzer = GameScreenAnalyzer()
        with patch.object(analyzer, '_get_ocr', return_value=mock_reader):
            result = analyzer.detect_card_counter(_make_screenshot())
            assert Rank.ACE in result
            assert Rank.KING in result

    def test_ocr_exception_returns_empty(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError('OCR crash')
        analyzer = GameScreenAnalyzer()
        with patch.object(analyzer, '_get_ocr', return_value=mock_reader):
            result = analyzer.detect_card_counter(_make_screenshot())
            assert result == {}


# ---------------------------------------------------------------------------
# detect_level_card
# ---------------------------------------------------------------------------

class TestDetectLevelCard:
    def test_no_easyocr_returns_none(self) -> None:
        analyzer = GameScreenAnalyzer()
        with patch('guandan.game_screen_analyzer.HAS_EASYOCR', False):
            assert analyzer.detect_level_card(_make_screenshot()) is None

    def test_no_cv2_returns_none(self) -> None:
        analyzer = GameScreenAnalyzer()
        with patch('guandan.game_screen_analyzer.HAS_CV2', False):
            assert analyzer.detect_level_card(_make_screenshot()) is None

    def test_ocr_finds_rank(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ['本局打 2']
        analyzer = GameScreenAnalyzer()
        with patch.object(analyzer, '_get_ocr', return_value=mock_reader):
            result = analyzer.detect_level_card(_make_screenshot())
            assert result == Rank.TWO

    def test_ocr_finds_high_rank(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ['Level A']
        analyzer = GameScreenAnalyzer()
        with patch.object(analyzer, '_get_ocr', return_value=mock_reader):
            result = analyzer.detect_level_card(_make_screenshot())
            assert result == Rank.ACE

    def test_ocr_no_match_returns_none(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ['nothing useful']
        analyzer = GameScreenAnalyzer()
        with patch.object(analyzer, '_get_ocr', return_value=mock_reader):
            result = analyzer.detect_level_card(_make_screenshot())
            assert result is None

    def test_ocr_exception_returns_none(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError('OCR crash')
        analyzer = GameScreenAnalyzer()
        with patch.object(analyzer, '_get_ocr', return_value=mock_reader):
            assert analyzer.detect_level_card(_make_screenshot()) is None


# ---------------------------------------------------------------------------
# detect_opponent_card_count
# ---------------------------------------------------------------------------

class TestDetectOpponentCardCount:
    def test_no_cv2_returns_empty(self) -> None:
        analyzer = GameScreenAnalyzer()
        with patch('guandan.game_screen_analyzer.HAS_CV2', False):
            result = analyzer.detect_opponent_card_count(_make_screenshot())
            assert result == {}

    def test_returns_three_positions(self) -> None:
        analyzer = GameScreenAnalyzer()
        result = analyzer.detect_opponent_card_count(_make_screenshot())
        assert 'left' in result
        assert 'right' in result
        assert 'top' in result

    def test_black_image_zero_counts(self) -> None:
        analyzer = GameScreenAnalyzer()
        result = analyzer.detect_opponent_card_count(
            _make_screenshot(color='black')
        )
        for pos in ('left', 'right', 'top'):
            assert result[pos] >= 0


# ---------------------------------------------------------------------------
# analyze (full pipeline)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_returns_all_keys(self) -> None:
        rec = _make_recognizer_stub([])
        analyzer = GameScreenAnalyzer(recognizer=rec)
        state = analyzer.analyze(_make_screenshot())
        assert 'my_turn' in state
        assert 'hand' in state
        assert 'played' in state
        assert 'counter' in state
        assert 'level' in state
        assert 'opponent_counts' in state

    def test_analyze_hand_populated(self) -> None:
        cards = [Card(rank=Rank.SEVEN, suit=Suit.CLUBS)]
        rec = _make_recognizer_stub(cards)
        analyzer = GameScreenAnalyzer(recognizer=rec)
        state = analyzer.analyze(_make_screenshot())
        assert len(state['hand']) == 1

    def test_analyze_played_none_when_empty(self) -> None:
        rec = _make_recognizer_stub([])
        analyzer = GameScreenAnalyzer(recognizer=rec)
        state = analyzer.analyze(_make_screenshot())
        assert state['played'] is None
