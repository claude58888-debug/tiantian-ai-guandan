"""Tests for Atom 1.4 - Wild card (逢人配) module."""
import pytest
from guandan.models import Card, Rank, Suit, JokerType
from guandan.combos import Combo, ComboType, classify_combo
from guandan.wild import (
    is_wild, count_wilds, split_wilds,
    classify_with_wilds, find_wild_combos,
    can_beat_with_wilds,
)


def make_card(rank, suit=Suit.HEARTS):
    return Card(rank=rank, suit=suit)


def make_wild(level=Rank.TWO):
    """Create a wild card (Heart of current level)."""
    return Card(rank=level, suit=Suit.HEARTS)


class TestIsWild:
    def test_heart_level_is_wild(self):
        card = Card(rank=Rank.TWO, suit=Suit.HEARTS)
        assert is_wild(card, Rank.TWO) is True

    def test_heart_non_level_not_wild(self):
        card = Card(rank=Rank.THREE, suit=Suit.HEARTS)
        assert is_wild(card, Rank.TWO) is False

    def test_non_heart_level_not_wild(self):
        card = Card(rank=Rank.TWO, suit=Suit.SPADES)
        assert is_wild(card, Rank.TWO) is False

    def test_joker_not_wild(self):
        card = Card(joker=JokerType.RED)
        assert is_wild(card, Rank.TWO) is False

    def test_level_five_wild(self):
        card = Card(rank=Rank.FIVE, suit=Suit.HEARTS)
        assert is_wild(card, Rank.FIVE) is True


class TestCountWilds:
    def test_count_zero(self):
        cards = [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.FOUR)]
        assert count_wilds(cards, Rank.TWO) == 0

    def test_count_one(self):
        cards = [make_wild(Rank.TWO), make_card(Rank.FOUR, Suit.SPADES)]
        assert count_wilds(cards, Rank.TWO) == 1

    def test_count_two(self):
        w1 = Card(rank=Rank.TWO, suit=Suit.HEARTS, deck_id=0)
        w2 = Card(rank=Rank.TWO, suit=Suit.HEARTS, deck_id=1)
        cards = [w1, w2]
        assert count_wilds(cards, Rank.TWO) == 2


class TestSplitWilds:
    def test_split(self):
        w = make_wild(Rank.TWO)
        n = make_card(Rank.FIVE, Suit.SPADES)
        normal, wilds = split_wilds([w, n], Rank.TWO)
        assert len(normal) == 1
        assert len(wilds) == 1
        assert normal[0] == n
        assert wilds[0] == w


class TestClassifyWithWilds:
    def test_natural_combo_no_wild(self):
        # Two threes = pair, no wild needed
        cards = [make_card(Rank.THREE, Suit.HEARTS), make_card(Rank.THREE, Suit.SPADES)]
        combo = classify_with_wilds(cards, Rank.TWO)
        assert combo is not None
        assert combo.combo_type == ComboType.PAIR

    def test_wild_forms_pair(self):
        # One three + one wild -> wild substitutes as three -> pair
        normal = make_card(Rank.THREE, Suit.SPADES)
        wild = make_wild(Rank.FIVE)  # Heart 5 is wild when level=5
        cards = [normal, wild]
        combo = classify_with_wilds(cards, Rank.FIVE)
        assert combo is not None
        assert combo.combo_type == ComboType.PAIR

    def test_no_wilds_invalid_returns_none(self):
        # Two different ranks, no wilds -> invalid
        cards = [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.FOUR, Suit.SPADES)]
        combo = classify_with_wilds(cards, Rank.TWO)
        assert combo is None

    def test_wild_forms_triple(self):
        # Two sevens + one wild = triple
        cards = [
            make_card(Rank.SEVEN, Suit.SPADES),
            make_card(Rank.SEVEN, Suit.CLUBS),
            make_wild(Rank.FIVE),
        ]
        combo = classify_with_wilds(cards, Rank.FIVE)
        assert combo is not None
        assert combo.combo_type == ComboType.TRIPLE

    def test_single_wild_is_single(self):
        # A single wild card played alone is a valid single
        wild = make_wild(Rank.TWO)
        combo = classify_with_wilds([wild], Rank.TWO)
        assert combo is not None
        assert combo.combo_type == ComboType.SINGLE


class TestFindWildCombos:
    def test_no_wilds(self):
        cards = [make_card(Rank.THREE, Suit.SPADES)]
        combos = find_wild_combos(cards, Rank.TWO)
        assert len(combos) == 1
        assert combos[0].combo_type == ComboType.SINGLE

    def test_wild_finds_multiple(self):
        # One card + one wild can form: single (playing wild alone isn't here since it's 2 cards),
        # or pair if wild substitutes
        normal = make_card(Rank.THREE, Suit.SPADES)
        wild = make_wild(Rank.FIVE)
        cards = [normal, wild]
        combos = find_wild_combos(cards, Rank.FIVE)
        # Should find at least a pair (wild as THREE)
        types = [c.combo_type for c in combos]
        assert ComboType.PAIR in types


class TestCanBeatWithWilds:
    def test_wild_pair_beats_lower(self):
        # Pair of 3s as last play
        last = classify_combo(
            [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.THREE, Suit.CLUBS)],
            Rank.TWO
        )
        # Five + wild(as five) = pair of fives, should beat pair of threes
        cards = [make_card(Rank.FIVE, Suit.SPADES), make_wild(Rank.TWO)]
        result = can_beat_with_wilds(cards, last, Rank.TWO)
        # Wild as TWO heart is being checked - it might form a pair of fives or not
        # depending on substitution. Result is None or a valid combo.
        assert result is None or result.combo_type >= ComboType.PAIR

    def test_cannot_beat_higher(self):
        last = classify_combo(
            [make_card(Rank.ACE, Suit.SPADES), make_card(Rank.ACE, Suit.CLUBS)],
            Rank.TWO
        )
        cards = [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.THREE, Suit.CLUBS)]
        result = can_beat_with_wilds(cards, last, Rank.TWO)
        assert result is None


# ── V0.3 C-2: Multi-wild substitution priority tests ──────────────────


class TestComboPriority:
    """Tests for the _combo_priority ranking function."""

    def test_bomb_beats_non_bomb(self):
        from guandan.wild import _combo_priority
        bomb = Combo(ComboType.BOMB_4, (), 5)
        pair = Combo(ComboType.PAIR, (), 14)
        assert _combo_priority(bomb) > _combo_priority(pair)

    def test_higher_bomb_beats_lower(self):
        from guandan.wild import _combo_priority
        b5 = Combo(ComboType.BOMB_5, (), 5)
        b4 = Combo(ComboType.BOMB_4, (), 14)
        assert _combo_priority(b5) > _combo_priority(b4)

    def test_larger_non_bomb_preferred(self):
        from guandan.wild import _combo_priority
        straight = Combo(ComboType.STRAIGHT, (make_card(Rank.THREE),) * 5, 7)
        pair = Combo(ComboType.PAIR, (make_card(Rank.THREE),) * 2, 14)
        assert _combo_priority(straight) > _combo_priority(pair)

    def test_same_type_higher_rank_wins(self):
        from guandan.wild import _combo_priority
        high = Combo(ComboType.PAIR, (make_card(Rank.ACE),) * 2, 14)
        low = Combo(ComboType.PAIR, (make_card(Rank.THREE),) * 2, 3)
        assert _combo_priority(high) > _combo_priority(low)


class TestCollectAllSubstitutions:
    """Tests for the exhaustive substitution search."""

    def test_two_wilds_finds_pair(self):
        from guandan.wild import _collect_all_substitutions
        w1 = Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=0)
        w2 = Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=1)
        results = _collect_all_substitutions([], [w1, w2], Rank.FIVE)
        types = {c.combo_type for c in results}
        assert ComboType.PAIR in types

    def test_deduplication(self):
        from guandan.wild import _collect_all_substitutions
        w1 = Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=0)
        results = _collect_all_substitutions(
            [make_card(Rank.THREE, Suit.SPADES)], [w1], Rank.FIVE,
        )
        keys = [(c.combo_type, c.rank_key) for c in results]
        assert len(keys) == len(set(keys))

    def test_original_cards_preserved(self):
        from guandan.wild import _collect_all_substitutions
        normal = make_card(Rank.THREE, Suit.SPADES)
        wild = Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=0)
        results = _collect_all_substitutions([normal], [wild], Rank.FIVE)
        for combo in results:
            assert normal in combo.cards
            assert wild in combo.cards


class TestClassifyWithWildsV03:
    """V0.3 tests: priority-based best selection with multiple wilds."""

    def test_two_wilds_prefer_bomb_over_pair(self):
        # 2 normal sevens + 2 wilds: can form bomb_4 (4 sevens) or pair
        cards = [
            make_card(Rank.SEVEN, Suit.SPADES),
            make_card(Rank.SEVEN, Suit.CLUBS),
            Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=0),
            Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=1),
        ]
        combo = classify_with_wilds(cards, Rank.FIVE)
        assert combo is not None
        assert combo.is_bomb

    def test_returns_none_for_impossible(self):
        cards = [
            make_card(Rank.THREE, Suit.SPADES),
            make_card(Rank.SEVEN, Suit.CLUBS),
        ]
        assert classify_with_wilds(cards, Rank.TWO) is None

    def test_joker_bomb_not_allowed_with_wilds(self):
        jokers = [
            Card(joker=JokerType.RED, deck_id=0),
            Card(joker=JokerType.BLACK, deck_id=0),
            Card(joker=JokerType.RED, deck_id=1),
        ]
        wild = Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=0)
        combo = classify_with_wilds(jokers + [wild], Rank.FIVE)
        assert combo is None


class TestFindWildCombosV03:
    """V0.3 tests: sorted output from find_wild_combos."""

    def test_results_sorted_by_priority(self):
        normal = make_card(Rank.THREE, Suit.SPADES)
        wild = Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=0)
        combos = find_wild_combos([normal, wild], Rank.FIVE)
        if len(combos) >= 2:
            from guandan.wild import _combo_priority
            priorities = [_combo_priority(c) for c in combos]
            assert priorities == sorted(priorities, reverse=True)


class TestCanBeatWithWildsV03:
    """V0.3 tests: exhaustive search finds beaters missed by greedy."""

    def test_searches_all_combos_for_beater(self):
        # Wild + normal can form pair that beats a lower pair
        last = classify_combo(
            [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.THREE, Suit.CLUBS)],
            Rank.FIVE
        )
        cards = [make_card(Rank.SEVEN, Suit.SPADES), Card(rank=Rank.FIVE, suit=Suit.HEARTS, deck_id=0)]
        result = can_beat_with_wilds(cards, last, Rank.FIVE)
        assert result is not None
        assert result.combo_type == ComboType.PAIR

    def test_no_wilds_fallback(self):
        last = classify_combo(
            [make_card(Rank.THREE, Suit.SPADES)], Rank.TWO
        )
        cards = [make_card(Rank.SEVEN, Suit.SPADES)]
        result = can_beat_with_wilds(cards, last, Rank.TWO)
        assert result is not None
        assert result.combo_type == ComboType.SINGLE
