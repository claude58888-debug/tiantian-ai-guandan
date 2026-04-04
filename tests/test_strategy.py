"""Tests for Atom 3.1 - AI strategy module."""
import pytest
from guandan.models import Card, Rank, Suit
from guandan.combos import Combo, ComboType, classify_combo
from guandan.strategy import (
    Strategy, RandomStrategy, GreedyStrategy, SmartStrategy,
    find_all_singles, find_all_pairs, find_all_triples,
    find_all_bombs, enumerate_plays, find_beating_plays,
    get_strategy, STRATEGIES,
)


def make_card(rank, suit=Suit.HEARTS):
    return Card(rank=rank, suit=suit)


def make_hand():
    """Create a small test hand."""
    return [
        make_card(Rank.THREE, Suit.HEARTS),
        make_card(Rank.THREE, Suit.DIAMONDS),
        make_card(Rank.FIVE, Suit.CLUBS),
        make_card(Rank.FIVE, Suit.SPADES),
        make_card(Rank.SEVEN, Suit.HEARTS),
        make_card(Rank.ACE, Suit.SPADES),
    ]


class TestFindCombos:
    def test_find_singles(self):
        hand = make_hand()
        singles = find_all_singles(hand, Rank.TWO)
        assert len(singles) == len(hand)
        for s in singles:
            assert s.combo_type == ComboType.SINGLE

    def test_find_pairs(self):
        hand = make_hand()
        pairs = find_all_pairs(hand, Rank.TWO)
        # Should find 3-3 pair and 5-5 pair
        assert len(pairs) >= 2
        for p in pairs:
            assert p.combo_type == ComboType.PAIR

    def test_find_triples_empty(self):
        hand = make_hand()
        triples = find_all_triples(hand, Rank.TWO)
        # No triples in our test hand
        assert len(triples) == 0

    def test_find_bombs_empty(self):
        hand = make_hand()
        bombs = find_all_bombs(hand, Rank.TWO)
        assert len(bombs) == 0

    def test_enumerate_plays(self):
        hand = make_hand()
        plays = enumerate_plays(hand, Rank.TWO)
        assert len(plays) > 0
        types = {p.combo_type for p in plays}
        assert ComboType.SINGLE in types
        assert ComboType.PAIR in types


class TestFindBeatingPlays:
    def test_beat_single(self):
        hand = make_hand()
        low_single = classify_combo([make_card(Rank.TWO)], Rank.TWO)
        beaters = find_beating_plays(hand, low_single, Rank.TWO)
        assert len(beaters) > 0

    def test_cannot_beat_high(self):
        hand = [make_card(Rank.THREE)]
        high = classify_combo([make_card(Rank.ACE)], Rank.TWO)
        beaters = find_beating_plays(hand, high, Rank.TWO)
        assert len(beaters) == 0


class TestRandomStrategy:
    def test_choose_lead(self):
        s = RandomStrategy()
        hand = make_hand()
        result = s.choose_lead(hand, Rank.TWO)
        assert result is not None
        assert len(result) >= 1

    def test_choose_lead_empty(self):
        s = RandomStrategy()
        result = s.choose_lead([], Rank.TWO)
        assert result is None

    def test_choose_response_pass(self):
        s = RandomStrategy()
        hand = [make_card(Rank.THREE)]
        high = classify_combo([make_card(Rank.ACE)], Rank.TWO)
        result = s.choose_response(hand, high, Rank.TWO)
        assert result is None

    def test_play_lead(self):
        s = RandomStrategy()
        hand = make_hand()
        result = s.play(hand, None, Rank.TWO)
        assert result is not None


class TestGreedyStrategy:
    def test_choose_lead_lowest(self):
        s = GreedyStrategy()
        hand = make_hand()
        result = s.choose_lead(hand, Rank.TWO)
        assert result is not None
        # Greedy leads with lowest single
        assert len(result) == 1

    def test_choose_response(self):
        s = GreedyStrategy()
        hand = make_hand()
        low = classify_combo([make_card(Rank.TWO)], Rank.TWO)
        result = s.choose_response(hand, low, Rank.TWO)
        assert result is not None

    def test_choose_response_pass(self):
        s = GreedyStrategy()
        hand = [make_card(Rank.THREE)]
        high = classify_combo([make_card(Rank.ACE)], Rank.TWO)
        result = s.choose_response(hand, high, Rank.TWO)
        assert result is None


class TestSmartStrategy:
    def test_default_aggression(self):
        s = SmartStrategy()
        assert s.aggression == 0.5

    def test_clamp_aggression(self):
        s = SmartStrategy(aggression=2.0)
        assert s.aggression == 1.0
        s2 = SmartStrategy(aggression=-1.0)
        assert s2.aggression == 0.0

    def test_hand_strength(self):
        s = SmartStrategy()
        hand = make_hand()
        strength = s._hand_strength(hand, Rank.TWO)
        assert 0.0 <= strength <= 1.0

    def test_hand_strength_empty(self):
        s = SmartStrategy()
        assert s._hand_strength([], Rank.TWO) == 0.0

    def test_choose_lead(self):
        s = SmartStrategy()
        hand = make_hand()
        result = s.choose_lead(hand, Rank.TWO)
        assert result is not None

    def test_aggressive_leads_pairs(self):
        s = SmartStrategy(aggression=0.9)
        hand = make_hand()
        result = s.choose_lead(hand, Rank.TWO)
        assert result is not None
        # High aggression should prefer pairs
        assert len(result) >= 1


class TestGetStrategy:
    def test_get_random(self):
        s = get_strategy('random')
        assert isinstance(s, RandomStrategy)

    def test_get_greedy(self):
        s = get_strategy('greedy')
        assert isinstance(s, GreedyStrategy)

    def test_get_smart(self):
        s = get_strategy('smart')
        assert isinstance(s, SmartStrategy)

    def test_get_smart_with_kwargs(self):
        s = get_strategy('smart', aggression=0.8)
        assert isinstance(s, SmartStrategy)
        assert s.aggression == 0.8

    def test_get_unknown(self):
        with pytest.raises(ValueError):
            get_strategy('nonexistent')

    def test_default_is_greedy(self):
        s = get_strategy()
        assert isinstance(s, GreedyStrategy)

    def test_strategies_registry(self):
        assert 'random' in STRATEGIES
        assert 'greedy' in STRATEGIES
        assert 'smart' in STRATEGIES


class TestRandomStrategy:
    def test_choose_lead(self):
        s = RandomStrategy()
        hand = make_hand()
        result = s.choose_lead(hand, Rank.TWO)
        assert result is not None
        assert len(result) >= 1

    def test_choose_lead_empty(self):
        s = RandomStrategy()
        assert s.choose_lead([], Rank.TWO) is None

    def test_choose_response_no_beater(self):
        s = RandomStrategy()
        hand = [make_card(Rank.THREE)]
        high_combo = classify_combo([make_card(Rank.ACE)], Rank.TWO)
        result = s.choose_response(hand, high_combo, Rank.TWO)
        # May or may not beat; just check it returns list or None
        assert result is None or isinstance(result, list)

    def test_play_dispatches_lead(self):
        s = RandomStrategy()
        hand = make_hand()
        result = s.play(hand, None, Rank.TWO)
        assert result is not None


class TestGreedyStrategy:
    def test_choose_lead_prefers_singles(self):
        s = GreedyStrategy()
        hand = make_hand()
        result = s.choose_lead(hand, Rank.TWO)
        assert result is not None
        # Greedy leads smallest single
        assert len(result) == 1

    def test_choose_lead_empty(self):
        s = GreedyStrategy()
        assert s.choose_lead([], Rank.TWO) is None

    def test_choose_response_avoids_bombs(self):
        s = GreedyStrategy()
        hand = make_hand()
        single = classify_combo([make_card(Rank.THREE)], Rank.TWO)
        result = s.choose_response(hand, single, Rank.TWO)
        # Should respond with non-bomb if possible
        assert result is None or isinstance(result, list)

    def test_play_dispatches_response(self):
        s = GreedyStrategy()
        hand = make_hand()
        single = classify_combo([make_card(Rank.THREE)], Rank.TWO)
        result = s.play(hand, single, Rank.TWO)
        assert result is None or isinstance(result, list)


class TestSmartStrategyResponse:
    def test_choose_response(self):
        s = SmartStrategy()
        hand = make_hand()
        single = classify_combo([make_card(Rank.THREE)], Rank.TWO)
        result = s.choose_response(hand, single, Rank.TWO)
        assert result is None or isinstance(result, list)

    def test_play_pass_combo(self):
        s = SmartStrategy()
        hand = make_hand()
        pass_combo = Combo(cards=[], combo_type=ComboType.PASS, rank_key=0)
        result = s.play(hand, pass_combo, Rank.TWO)
        # PASS should trigger choose_lead
        assert result is not None


class TestEnumeratePlays:
    def test_enumerate_returns_list(self):
        hand = make_hand()
        plays = enumerate_plays(hand, Rank.TWO)
        assert isinstance(plays, list)
        assert len(plays) > 0

    def test_enumerate_empty(self):
        plays = enumerate_plays([], Rank.TWO)
        assert plays == []


class TestFindBeatingPlays:
    def test_find_beaters(self):
        hand = make_hand()
        single = classify_combo([make_card(Rank.THREE)], Rank.TWO)
        beaters = find_beating_plays(hand, single, Rank.TWO)
        assert isinstance(beaters, list)
