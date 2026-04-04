"""Tests for Atom 2.2 - Card detector module."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from guandan.models import Card, Rank, Suit, JokerType
from guandan.card_detector import (
    DetectedCard, CardTemplate, TemplateStore, CardDetector,
    hamming_distance, segment_hand_cards, detect_card_color,
)


class TestDetectedCard:
    def test_repr(self):
        c = Card(rank=Rank.ACE, suit=Suit.SPADES)
        dc = DetectedCard(card=c, confidence=0.95)
        r = repr(dc)
        assert 'DetectedCard' in r
        assert '0.95' in r

    def test_defaults(self):
        c = Card(rank=Rank.THREE, suit=Suit.HEARTS)
        dc = DetectedCard(card=c)
        assert dc.confidence == 0.0
        assert dc.bbox == (0, 0, 0, 0)
        assert dc.center == (0, 0)


class TestCardTemplate:
    def test_compute_phash(self):
        try:
            from PIL import Image
            img = Image.new('RGB', (50, 80), color='red')
            h = CardTemplate.compute_phash(img)
            assert isinstance(h, str)
            assert len(h) == 16
        except ImportError:
            pytest.skip('PIL not available')

    def test_phash_different_images(self):
        try:
            from PIL import Image
            img1 = Image.new('RGB', (50, 80), color='red')
            img2 = Image.new('RGB', (50, 80), color='blue')
            h1 = CardTemplate.compute_phash(img1)
            h2 = CardTemplate.compute_phash(img2)
            # Different images may have different hashes
            assert isinstance(h1, str)
            assert isinstance(h2, str)
        except ImportError:
            pytest.skip('PIL not available')


class TestHammingDistance:
    def test_same_hash(self):
        assert hamming_distance('abcd1234abcd1234', 'abcd1234abcd1234') == 0

    def test_different_hash(self):
        d = hamming_distance('0000000000000000', 'ffffffffffffffff')
        assert d == 64  # all bits different

    def test_one_bit_diff(self):
        d = hamming_distance('0000000000000000', '0000000000000001')
        assert d == 1


class TestTemplateStore:
    def test_init_default(self):
        store = TemplateStore()
        assert store.template_dir == Path('templates')
        assert store.templates == {}

    def test_load_no_dir(self):
        store = TemplateStore(Path('/nonexistent/dir'))
        count = store.load_templates()
        assert count == 0

    def test_parse_filename_normal(self):
        store = TemplateStore()
        card = store._parse_filename('3H')
        assert card is not None
        assert card.rank == Rank.THREE
        assert card.suit == Suit.HEARTS

    def test_parse_filename_ten(self):
        store = TemplateStore()
        card = store._parse_filename('10S')
        assert card is not None
        assert card.rank == Rank.TEN

    def test_parse_filename_jokers(self):
        store = TemplateStore()
        bj = store._parse_filename('BJ')
        rj = store._parse_filename('RJ')
        assert bj.joker == JokerType.BLACK
        assert rj.joker == JokerType.RED

    def test_parse_filename_invalid(self):
        store = TemplateStore()
        assert store._parse_filename('X') is None
        assert store._parse_filename('ZZ') is None

    def test_find_best_match_empty(self):
        store = TemplateStore()
        try:
            from PIL import Image
            img = Image.new('RGB', (50, 80))
            result = store.find_best_match(img)
            assert result is None
        except ImportError:
            pytest.skip('PIL not available')


class TestSegmentHandCards:
    def test_segment_basic(self):
        try:
            from PIL import Image
            # 200px wide, 100px tall hand image
            img = Image.new('RGB', (200, 100))
            segments = segment_hand_cards(img, card_width=40, overlap=15)
            assert len(segments) > 0
            for seg in segments:
                assert seg.size[0] == 40
                assert seg.size[1] == 100
        except ImportError:
            pytest.skip('PIL not available')

    def test_segment_small_image(self):
        try:
            from PIL import Image
            img = Image.new('RGB', (30, 100))  # smaller than card_width
            segments = segment_hand_cards(img, card_width=40)
            assert len(segments) == 0
        except ImportError:
            pytest.skip('PIL not available')


class TestDetectCardColor:
    def test_red_card(self):
        try:
            from PIL import Image
            img = Image.new('RGB', (50, 80), color=(200, 50, 50))
            assert detect_card_color(img) == 'red'
        except ImportError:
            pytest.skip('PIL not available')

    def test_black_card(self):
        try:
            from PIL import Image
            img = Image.new('RGB', (50, 80), color=(30, 30, 30))
            assert detect_card_color(img) == 'black'
        except ImportError:
            pytest.skip('PIL not available')


class TestCardDetector:
    def test_init(self):
        det = CardDetector()
        assert not det.is_loaded

    def test_load_no_templates(self):
        det = CardDetector(Path('/nonexistent'))
        count = det.load()
        assert count == 0
        assert not det.is_loaded

    def test_detect_hand_empty(self):
        try:
            from PIL import Image
            det = CardDetector()
            img = Image.new('RGB', (200, 100))
            results = det.detect_hand(img)
            assert results == []
        except ImportError:
            pytest.skip('PIL not available')

    def test_detect_single_no_templates(self):
        try:
            from PIL import Image
            det = CardDetector()
            img = Image.new('RGB', (50, 80))
            result = det.detect_single(img)
            assert result is None
        except ImportError:
            pytest.skip('PIL not available')
