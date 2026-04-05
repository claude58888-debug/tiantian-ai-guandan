"""Tests for M6 - Card template generator module."""
import pytest
from pathlib import Path

from guandan.models import Rank, Suit
from guandan.card_template_generator import (
    TEMPLATE_WIDTH,
    TEMPLATE_HEIGHT,
    SUIT_SYMBOLS,
    SUIT_COLORS,
    RANK_LABELS,
    TemplateSpec,
    render_template,
    generate_all_templates,
    get_all_specs,
    _build_specs,
    _get_font,
)


class TestConstants:
    def test_template_dimensions_positive(self):
        assert TEMPLATE_WIDTH > 0
        assert TEMPLATE_HEIGHT > 0

    def test_suit_symbols_complete(self):
        for suit in Suit:
            assert suit in SUIT_SYMBOLS
            assert len(SUIT_SYMBOLS[suit]) == 1

    def test_suit_colors_complete(self):
        for suit in Suit:
            assert suit in SUIT_COLORS
            r, g, b = SUIT_COLORS[suit]
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_rank_labels_complete(self):
        for rank in Rank:
            assert rank in RANK_LABELS

    def test_red_suits_are_red(self):
        for suit in (Suit.HEARTS, Suit.DIAMONDS):
            r, g, b = SUIT_COLORS[suit]
            assert r > g and r > b

    def test_black_suits_are_dark(self):
        for suit in (Suit.CLUBS, Suit.SPADES):
            r, g, b = SUIT_COLORS[suit]
            assert r < 100 and g < 100 and b < 100


class TestTemplateSpec:
    def test_normal_card_spec(self):
        spec = TemplateSpec(
            filename='3H',
            rank_label='3',
            suit_symbol='\u2665',
            color=(220, 20, 20),
        )
        assert spec.filename == '3H'
        assert not spec.is_joker
        assert spec.joker_label == ''

    def test_joker_spec(self):
        spec = TemplateSpec(
            filename='BJ',
            rank_label='',
            suit_symbol='',
            color=(30, 30, 30),
            is_joker=True,
            joker_label='BJ',
        )
        assert spec.is_joker
        assert spec.joker_label == 'BJ'

    def test_frozen(self):
        spec = TemplateSpec(filename='AH', rank_label='A',
                            suit_symbol='\u2665', color=(220, 20, 20))
        with pytest.raises(AttributeError):
            spec.filename = 'XX'  # type: ignore[misc]


class TestBuildSpecs:
    def test_total_count(self):
        specs = _build_specs()
        # 13 ranks x 4 suits + 2 jokers = 54
        assert len(specs) == 54

    def test_unique_filenames(self):
        specs = _build_specs()
        filenames = [s.filename for s in specs]
        assert len(set(filenames)) == 54

    def test_jokers_present(self):
        specs = _build_specs()
        joker_fnames = [s.filename for s in specs if s.is_joker]
        assert 'BJ' in joker_fnames
        assert 'RJ' in joker_fnames

    def test_all_suits_represented(self):
        specs = _build_specs()
        suit_initials = {s.filename[-1] for s in specs if not s.is_joker}
        assert suit_initials == {'H', 'D', 'C', 'S'}

    def test_get_all_specs_matches(self):
        assert get_all_specs() == _build_specs()


class TestGetFont:
    def test_returns_font(self):
        try:
            from PIL import ImageFont
        except ImportError:
            pytest.skip('PIL not available')
        font = _get_font(14)
        assert font is not None


class TestRenderTemplate:
    def test_normal_card(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')
        spec = TemplateSpec(
            filename='KS',
            rank_label='K',
            suit_symbol='\u2660',
            color=(30, 30, 30),
        )
        img = render_template(spec)
        assert img.mode == 'RGBA'
        assert img.size == (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)

    def test_joker(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')
        spec = TemplateSpec(
            filename='RJ', rank_label='', suit_symbol='',
            color=(220, 20, 20), is_joker=True, joker_label='RJ',
        )
        img = render_template(spec)
        assert img.mode == 'RGBA'
        assert img.size == (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)

    def test_custom_dimensions(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')
        spec = TemplateSpec(
            filename='5D', rank_label='5',
            suit_symbol='\u2666', color=(220, 20, 20),
        )
        img = render_template(spec, width=64, height=96)
        assert img.size == (64, 96)

    def test_ten_renders(self):
        """10 is a two-character rank label; should not crash."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')
        spec = TemplateSpec(
            filename='10H', rank_label='10',
            suit_symbol='\u2665', color=(220, 20, 20),
        )
        img = render_template(spec)
        assert img.size == (TEMPLATE_WIDTH, TEMPLATE_HEIGHT)


class TestGenerateAllTemplates:
    def test_generates_54_files(self, tmp_path: Path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')
        count = generate_all_templates(tmp_path)
        assert count == 54

        png_files = list(tmp_path.glob('*.png'))
        assert len(png_files) == 54

    def test_creates_output_dir(self, tmp_path: Path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')
        out = tmp_path / 'sub' / 'templates'
        count = generate_all_templates(out)
        assert count == 54
        assert out.is_dir()

    def test_template_files_are_rgba(self, tmp_path: Path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')
        generate_all_templates(tmp_path)
        sample = Image.open(tmp_path / '3H.png')
        assert sample.mode == 'RGBA'

    def test_joker_files_exist(self, tmp_path: Path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('PIL not available')
        generate_all_templates(tmp_path)
        assert (tmp_path / 'BJ.png').exists()
        assert (tmp_path / 'RJ.png').exists()
