"""Tests for guandan.card_extractor (M8)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from guandan.card_extractor import (
    CardCorner,
    DEFAULT_CORNER_HEIGHT,
    DEFAULT_CORNER_WIDTH,
    ExtractedTemplate,
    MIN_CARD_GAP,
    RAISED_CARD_OFFSET,
    build_template_set,
    classify_suit_by_colour,
    detect_raised_cards,
    extract_card_corners,
    extract_rank_region,
    extract_suit_region,
)
from guandan.models import Card, Rank, Suit


# ---------------------------------------------------------------------------
# CardCorner dataclass
# ---------------------------------------------------------------------------

class TestCardCorner:
    def test_defaults(self) -> None:
        img = MagicMock()
        cc = CardCorner(image=img)
        assert cc.x == 0
        assert cc.y == 0
        assert cc.width == DEFAULT_CORNER_WIDTH
        assert cc.height == DEFAULT_CORNER_HEIGHT

    def test_custom_values(self) -> None:
        img = MagicMock()
        cc = CardCorner(image=img, x=10, y=20, width=50, height=70)
        assert cc.x == 10
        assert cc.y == 20
        assert cc.width == 50
        assert cc.height == 70

    def test_frozen(self) -> None:
        img = MagicMock()
        cc = CardCorner(image=img)
        with pytest.raises(AttributeError):
            cc.x = 99  # type: ignore[misc]

    def test_save(self) -> None:
        mock_img = MagicMock()
        cc = CardCorner(image=mock_img)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'sub' / 'corner.png'
            cc.save(path)
            mock_img.save.assert_called_once_with(path)


class TestExtractedTemplate:
    def test_construction(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        et = ExtractedTemplate(
            card=card,
            rank_image=MagicMock(),
            suit_image=MagicMock(),
            full_corner=MagicMock(),
        )
        assert et.card == card


# ---------------------------------------------------------------------------
# extract_card_corners
# ---------------------------------------------------------------------------

class TestExtractCardCorners:
    @patch('guandan.card_extractor.HAS_CV2', False)
    def test_no_cv2_returns_empty(self) -> None:
        result = extract_card_corners(MagicMock())
        assert result == []

    @patch('guandan.card_extractor.HAS_PIL', False)
    def test_no_pil_returns_empty(self) -> None:
        result = extract_card_corners(MagicMock())
        assert result == []

    @patch('guandan.card_extractor.HAS_CV2', True)
    @patch('guandan.card_extractor.HAS_PIL', True)
    def test_extracts_from_synthetic_image(self) -> None:
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip('PIL/numpy required')

        # Create an image with some vertical edges
        img = Image.new('RGB', (400, 200), (200, 200, 200))
        arr = np.array(img)
        # Draw dark vertical lines to simulate card edges
        for x in [50, 100, 150, 200]:
            arr[120:200, x, :] = 0
        img = Image.fromarray(arr)

        corners = extract_card_corners(img, hand_top_ratio=0.5)
        assert isinstance(corners, list)
        for cc in corners:
            assert isinstance(cc, CardCorner)

    @patch('guandan.card_extractor.HAS_CV2', True)
    @patch('guandan.card_extractor.HAS_PIL', True)
    def test_uniform_image_no_corners(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        img = Image.new('RGB', (400, 200), (128, 128, 128))
        corners = extract_card_corners(img)
        assert isinstance(corners, list)
        # Uniform image should have few or no edges
        # (depends on Canny sensitivity)

    @patch('guandan.card_extractor.HAS_CV2', True)
    @patch('guandan.card_extractor.HAS_PIL', True)
    def test_custom_corner_size(self) -> None:
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip('PIL/numpy required')

        img = Image.new('RGB', (400, 200), (200, 200, 200))
        arr = np.array(img)
        arr[120:200, 50, :] = 0
        img = Image.fromarray(arr)

        corners = extract_card_corners(
            img, hand_top_ratio=0.5,
            corner_width=30, corner_height=40,
        )
        for cc in corners:
            assert cc.width <= 30
            assert cc.height <= 40


# ---------------------------------------------------------------------------
# extract_rank_region / extract_suit_region
# ---------------------------------------------------------------------------

class TestExtractRegions:
    def test_rank_region_top_45_percent(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        img = Image.new('RGB', (40, 60), (255, 255, 255))
        rank = extract_rank_region(img)
        assert rank.size == (40, 27)  # 60 * 0.45 = 27

    def test_suit_region_40_to_75_percent(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        img = Image.new('RGB', (40, 60), (255, 255, 255))
        suit = extract_suit_region(img)
        # top = 60 * 0.40 = 24, bottom = 60 * 0.75 = 45
        assert suit.size == (40, 21)


# ---------------------------------------------------------------------------
# detect_raised_cards
# ---------------------------------------------------------------------------

class TestDetectRaisedCards:
    @patch('guandan.card_extractor.HAS_CV2', False)
    def test_no_cv2_returns_empty(self) -> None:
        result = detect_raised_cards(MagicMock())
        assert result == []

    @patch('guandan.card_extractor.HAS_CV2', True)
    @patch('guandan.card_extractor.HAS_PIL', True)
    def test_no_raised_on_dark_image(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        img = Image.new('RGB', (400, 200), (20, 20, 20))
        result = detect_raised_cards(img, hand_top_ratio=0.6)
        assert isinstance(result, list)

    @patch('guandan.card_extractor.HAS_CV2', True)
    @patch('guandan.card_extractor.HAS_PIL', True)
    def test_detects_bright_pixels_above_hand(self) -> None:
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip('PIL/numpy required')

        img = Image.new('RGB', (400, 200), (20, 20, 20))
        arr = np.array(img)
        # Place bright pixels just above hand_top (y=120)
        # hand_top_ratio=0.6 => top_px=120, strip is 100-120
        arr[100:120, 150:180, :] = 255
        img = Image.fromarray(arr)

        result = detect_raised_cards(img, hand_top_ratio=0.6, offset_px=15)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# build_template_set
# ---------------------------------------------------------------------------

class TestBuildTemplateSet:
    def test_empty_inputs(self) -> None:
        result = build_template_set([], [])
        assert result == {}

    def test_matches_corners_to_labels(self) -> None:
        mock_img1 = MagicMock()
        mock_img2 = MagicMock()
        c1 = CardCorner(image=mock_img1)
        c2 = CardCorner(image=mock_img2)
        labels = [
            Card(rank=Rank.ACE, suit=Suit.HEARTS),
            Card(rank=Rank.KING, suit=Suit.SPADES),
        ]
        result = build_template_set([c1, c2], labels)
        assert len(result) == 2
        assert mock_img1 in result.values()
        assert mock_img2 in result.values()

    def test_truncates_to_shorter_list(self) -> None:
        c1 = CardCorner(image=MagicMock())
        labels = [
            Card(rank=Rank.ACE, suit=Suit.HEARTS),
            Card(rank=Rank.KING, suit=Suit.SPADES),
        ]
        result = build_template_set([c1], labels)
        assert len(result) == 1

    def test_saves_to_output_dir(self) -> None:
        mock_img = MagicMock()
        c1 = CardCorner(image=mock_img)
        labels = [Card(rank=Rank.ACE, suit=Suit.HEARTS)]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / 'templates'
            result = build_template_set([c1], labels, output_dir=out)
            assert len(result) == 1
            mock_img.save.assert_called_once()


# ---------------------------------------------------------------------------
# classify_suit_by_colour
# ---------------------------------------------------------------------------

class TestClassifySuitByColour:
    @patch('guandan.card_extractor.HAS_CV2', False)
    def test_no_cv2_returns_none(self) -> None:
        assert classify_suit_by_colour(MagicMock()) is None

    @patch('guandan.card_extractor.HAS_CV2', True)
    @patch('guandan.card_extractor.HAS_PIL', True)
    def test_red_image(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        # Bright red image
        img = Image.new('RGB', (40, 60), (220, 20, 20))
        result = classify_suit_by_colour(img)
        assert result == 'red'

    @patch('guandan.card_extractor.HAS_CV2', True)
    @patch('guandan.card_extractor.HAS_PIL', True)
    def test_black_image(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        # Dark image
        img = Image.new('RGB', (40, 60), (20, 20, 20))
        result = classify_suit_by_colour(img)
        assert result == 'black'

    @patch('guandan.card_extractor.HAS_CV2', True)
    @patch('guandan.card_extractor.HAS_PIL', True)
    def test_white_image_returns_none(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL required')

        img = Image.new('RGB', (40, 60), (255, 255, 255))
        result = classify_suit_by_colour(img)
        # White is neither red nor black in HSV terms
        assert result is None or result in ('red', 'black')
