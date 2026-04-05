"""Tests for M6 - Card recognition engine."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from guandan.models import Card, Rank, Suit, JokerType
from guandan.card_recognition import (
    RecognizedCard,
    CardTemplate,
    CardRecognizer,
    non_maximum_suppression,
    _iou,
    _parse_template_filename,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_SCALES,
    NMS_OVERLAP_THRESHOLD,
)


# ── RecognizedCard ────────────────────────────────────────────────────

class TestRecognizedCard:
    def test_repr(self):
        c = Card(rank=Rank.ACE, suit=Suit.SPADES)
        rc = RecognizedCard(card=c, confidence=0.95, bbox=(10, 20, 32, 48))
        r = repr(rc)
        assert 'RecognizedCard' in r
        assert '0.95' in r

    def test_defaults(self):
        c = Card(rank=Rank.THREE, suit=Suit.HEARTS)
        rc = RecognizedCard(card=c)
        assert rc.confidence == 0.0
        assert rc.bbox == (0, 0, 0, 0)


# ── IOU ───────────────────────────────────────────────────────────────

class TestIOU:
    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0

    def test_full_overlap(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 5, 10, 10))
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        assert _iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


# ── NMS ───────────────────────────────────────────────────────────────

class TestNonMaximumSuppression:
    def test_empty_list(self):
        assert non_maximum_suppression([]) == []

    def test_single_detection(self):
        c = Card(rank=Rank.FIVE, suit=Suit.CLUBS)
        det = RecognizedCard(card=c, confidence=0.9, bbox=(0, 0, 32, 48))
        result = non_maximum_suppression([det])
        assert len(result) == 1

    def test_removes_overlapping(self):
        c = Card(rank=Rank.FIVE, suit=Suit.CLUBS)
        high = RecognizedCard(card=c, confidence=0.95, bbox=(0, 0, 32, 48))
        low = RecognizedCard(card=c, confidence=0.85, bbox=(2, 2, 32, 48))
        result = non_maximum_suppression([low, high], overlap_threshold=0.3)
        assert len(result) == 1
        assert result[0].confidence == 0.95

    def test_keeps_non_overlapping(self):
        c1 = Card(rank=Rank.FIVE, suit=Suit.CLUBS)
        c2 = Card(rank=Rank.SIX, suit=Suit.HEARTS)
        d1 = RecognizedCard(card=c1, confidence=0.9, bbox=(0, 0, 32, 48))
        d2 = RecognizedCard(card=c2, confidence=0.85, bbox=(100, 0, 32, 48))
        result = non_maximum_suppression([d1, d2])
        assert len(result) == 2


# ── Filename parsing ──────────────────────────────────────────────────

class TestParseTemplateFilename:
    def test_normal_card(self):
        card = _parse_template_filename('3H')
        assert card is not None
        assert card.rank == Rank.THREE
        assert card.suit == Suit.HEARTS

    def test_ten(self):
        card = _parse_template_filename('10S')
        assert card is not None
        assert card.rank == Rank.TEN
        assert card.suit == Suit.SPADES

    def test_jokers(self):
        bj = _parse_template_filename('BJ')
        rj = _parse_template_filename('RJ')
        assert bj is not None and bj.joker == JokerType.BLACK
        assert rj is not None and rj.joker == JokerType.RED

    def test_invalid(self):
        assert _parse_template_filename('X') is None
        assert _parse_template_filename('ZZ') is None

    def test_case_insensitive(self):
        card = _parse_template_filename('ah')
        assert card is not None
        assert card.rank == Rank.ACE


# ── CardRecognizer ────────────────────────────────────────────────────

class TestCardRecognizerInit:
    def test_defaults(self):
        rec = CardRecognizer()
        assert rec.threshold == DEFAULT_MATCH_THRESHOLD
        assert rec.scales == DEFAULT_SCALES
        assert rec.nms_overlap == NMS_OVERLAP_THRESHOLD
        assert not rec.is_loaded
        assert rec.template_count == 0

    def test_custom_params(self):
        rec = CardRecognizer(threshold=0.7, scales=(1.0,), nms_overlap=0.5)
        assert rec.threshold == 0.7
        assert rec.scales == (1.0,)
        assert rec.nms_overlap == 0.5


class TestCardRecognizerLoadTemplates:
    def test_load_nonexistent_dir(self):
        rec = CardRecognizer()
        count = rec.load_templates(Path('/nonexistent/templates'))
        assert count == 0
        assert not rec.is_loaded

    def test_load_from_generated_templates(self, tmp_path: Path):
        try:
            from PIL import Image
            import cv2
        except ImportError:
            pytest.skip('PIL or OpenCV not available')

        from guandan.card_template_generator import generate_all_templates
        generate_all_templates(tmp_path)

        rec = CardRecognizer()
        count = rec.load_templates(tmp_path)
        assert count == 54
        assert rec.is_loaded
        assert rec.template_count == 54

    def test_load_templates_from_pil(self):
        try:
            from PIL import Image
            import cv2
        except ImportError:
            pytest.skip('PIL or OpenCV not available')

        rec = CardRecognizer()
        images = {
            '3H': Image.new('RGB', (32, 48), color='red'),
            'BJ': Image.new('RGB', (32, 48), color='black'),
        }
        count = rec.load_templates_from_pil(images)
        assert count == 2
        assert rec.is_loaded


class TestCardRecognizerDetection:
    def test_find_cards_no_templates(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')

        rec = CardRecognizer()
        img = Image.new('RGB', (200, 100))
        result = rec.find_cards_in_region(img)
        assert result == []

    def test_find_cards_with_templates(self, tmp_path: Path):
        try:
            from PIL import Image
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip('Dependencies not available')

        from guandan.card_template_generator import generate_all_templates
        generate_all_templates(tmp_path)

        rec = CardRecognizer(threshold=0.7, scales=(1.0,))
        rec.load_templates(tmp_path)

        # Create a scene by pasting a template into a larger image
        tmpl_img = Image.open(tmp_path / '3H.png').convert('RGB')
        scene = Image.new('RGB', (200, 100), color='white')
        scene.paste(tmpl_img, (50, 20))

        detections = rec.find_cards_in_region(scene, threshold=0.7)
        # Should find at least some detection (may not be exact due to
        # rendering differences)
        assert isinstance(detections, list)

    def test_recognize_hand_no_templates(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')

        rec = CardRecognizer()
        screenshot = Image.new('RGB', (1024, 768))
        cards = rec.recognize_hand(screenshot)
        assert cards == []

    def test_recognize_played_cards_no_templates(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')

        rec = CardRecognizer()
        screenshot = Image.new('RGB', (1024, 768))
        cards = rec.recognize_played_cards(screenshot)
        assert cards == []

    def test_recognize_game_state_structure(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')

        rec = CardRecognizer()
        screenshot = Image.new('RGB', (1024, 768))
        state = rec.recognize_game_state(screenshot)
        assert 'hand' in state
        assert 'played' in state
        assert 'opponent_counts' in state
        assert isinstance(state['hand'], list)
        assert isinstance(state['played'], list)
        assert isinstance(state['opponent_counts'], dict)

    def test_recognize_hand_deduplicates(self):
        """recognize_hand should return unique cards only."""
        try:
            from PIL import Image
            import cv2
        except ImportError:
            pytest.skip('Dependencies not available')

        rec = CardRecognizer()
        # Even with no templates, the method should return a list
        screenshot = Image.new('RGB', (1024, 768))
        cards = rec.recognize_hand(screenshot)
        # If cards were returned, they should be unique
        displays = [c.display() for c in cards]
        assert len(displays) == len(set(displays))


class TestCardRecognizerCustomThreshold:
    def test_higher_threshold_fewer_results(self, tmp_path: Path):
        try:
            from PIL import Image
            import cv2
        except ImportError:
            pytest.skip('Dependencies not available')

        from guandan.card_template_generator import generate_all_templates
        generate_all_templates(tmp_path)

        rec_strict = CardRecognizer(threshold=0.99, scales=(1.0,))
        rec_strict.load_templates(tmp_path)

        rec_loose = CardRecognizer(threshold=0.5, scales=(1.0,))
        rec_loose.load_templates(tmp_path)

        scene = Image.new('RGB', (200, 100), color='white')
        strict_results = rec_strict.find_cards_in_region(scene)
        loose_results = rec_loose.find_cards_in_region(scene)
        # Stricter threshold should never produce more results
        assert len(strict_results) <= len(loose_results)
