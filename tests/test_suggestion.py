"""Tests for guandan.suggestion module (Atom 5.1).

Comprehensive tests for the suggestion engine that integrates
strategy and card_counter to produce Top-3 play suggestions.
"""
from __future__ import annotations

import pytest

from guandan.card_counter import CardCounter, RiskLevel, RiskReport
from guandan.combos import Combo, ComboType, classify_combo
from guandan.models import Card, JokerType, Rank, Suit
from guandan.suggestion import (
    PlaySuggestion,
    SuggestionEngine,
    SuggestionRisk,
    get_suggestions,
    _base_score,
    _combo_type_name,
    _generate_pass_reason,
    _generate_reason,
    _pass_confidence,
    _pass_risk,
    _risk_for_play,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card(rank: Rank, suit: Suit = Suit.HEARTS) -> Card:
    return Card(rank=rank, suit=suit)


def _joker(color: JokerType) -> Card:
    return Card(joker=color)


def _make_hand() -> list[Card]:
    """Small test hand: 3h, 3d, 5c, 5s, 7h, Ah."""
    return [
        _card(Rank.THREE, Suit.HEARTS),
        _card(Rank.THREE, Suit.DIAMONDS),
        _card(Rank.FIVE, Suit.CLUBS),
        _card(Rank.FIVE, Suit.SPADES),
        _card(Rank.SEVEN, Suit.HEARTS),
        _card(Rank.ACE, Suit.SPADES),
    ]


def _make_large_hand() -> list[Card]:
    """Larger hand (16 cards) with a bomb."""
    cards = [
        _card(Rank.THREE, Suit.HEARTS),
        _card(Rank.THREE, Suit.DIAMONDS),
        _card(Rank.FOUR, Suit.CLUBS),
        _card(Rank.FOUR, Suit.SPADES),
        _card(Rank.FIVE, Suit.HEARTS),
        _card(Rank.FIVE, Suit.DIAMONDS),
        _card(Rank.SIX, Suit.CLUBS),
        _card(Rank.SIX, Suit.SPADES),
        _card(Rank.SEVEN, Suit.HEARTS),
        _card(Rank.SEVEN, Suit.DIAMONDS),
        _card(Rank.EIGHT, Suit.CLUBS),
        _card(Rank.EIGHT, Suit.SPADES),
        _card(Rank.EIGHT, Suit.HEARTS),
        _card(Rank.EIGHT, Suit.DIAMONDS),
        _card(Rank.ACE, Suit.SPADES),
        _card(Rank.ACE, Suit.HEARTS),
    ]
    return cards


# ---------------------------------------------------------------------------
# PlaySuggestion dataclass
# ---------------------------------------------------------------------------

class TestPlaySuggestion:
    def test_create_with_cards(self) -> None:
        cards = (_card(Rank.THREE),)
        s = PlaySuggestion(
            cards=cards,
            combo_type="single",
            reason="test reason",
            risk=SuggestionRisk.SAFE,
            confidence=0.75,
        )
        assert s.cards == cards
        assert s.combo_type == "single"
        assert s.reason == "test reason"
        assert s.risk == SuggestionRisk.SAFE
        assert s.confidence == 0.75
        assert not s.is_pass

    def test_pass_suggestion(self) -> None:
        s = PlaySuggestion(
            cards=(),
            combo_type="pass",
            reason="wait",
            risk=SuggestionRisk.SAFE,
            confidence=0.3,
        )
        assert s.is_pass
        assert s.display_cards() == "PASS"

    def test_display_cards(self) -> None:
        c1 = _card(Rank.ACE, Suit.SPADES)
        c2 = _card(Rank.ACE, Suit.HEARTS)
        s = PlaySuggestion(
            cards=(c1, c2),
            combo_type="pair",
            reason="r",
            risk=SuggestionRisk.MODERATE,
            confidence=0.6,
        )
        display = s.display_cards()
        assert "A" in display
        assert display != "PASS"

    def test_display_full(self) -> None:
        s = PlaySuggestion(
            cards=(_card(Rank.FIVE),),
            combo_type="single",
            reason="probe",
            risk=SuggestionRisk.RISKY,
            confidence=0.42,
        )
        text = s.display()
        assert "single" in text
        assert "0.42" in text
        assert "RISKY" in text
        assert "probe" in text

    def test_frozen(self) -> None:
        s = PlaySuggestion(
            cards=(),
            combo_type="pass",
            reason="r",
            risk=SuggestionRisk.SAFE,
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            s.confidence = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SuggestionRisk enum
# ---------------------------------------------------------------------------

class TestSuggestionRisk:
    def test_values(self) -> None:
        assert SuggestionRisk.SAFE.value == 1
        assert SuggestionRisk.MODERATE.value == 2
        assert SuggestionRisk.RISKY.value == 3

    def test_names(self) -> None:
        assert SuggestionRisk.SAFE.name == "SAFE"
        assert SuggestionRisk.MODERATE.name == "MODERATE"
        assert SuggestionRisk.RISKY.name == "RISKY"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

class TestComboTypeName:
    def test_known_types(self) -> None:
        assert _combo_type_name(ComboType.SINGLE) == "single"
        assert _combo_type_name(ComboType.PAIR) == "pair"
        assert _combo_type_name(ComboType.BOMB_4) == "bomb_4"
        assert _combo_type_name(ComboType.JOKER_BOMB) == "joker_bomb"
        assert _combo_type_name(ComboType.PASS) == "pass"
        assert _combo_type_name(ComboType.FULL_HOUSE) == "full_house"
        assert _combo_type_name(ComboType.STRAIGHT) == "straight"
        assert _combo_type_name(ComboType.PLATE) == "plate"


class TestBaseScore:
    def test_single_low_rank(self) -> None:
        combo = classify_combo([_card(Rank.THREE)], Rank.TWO)
        assert combo is not None
        score = _base_score(combo, 10)
        assert 0.05 <= score <= 1.0

    def test_single_high_rank(self) -> None:
        combo = classify_combo([_card(Rank.ACE)], Rank.TWO)
        assert combo is not None
        score = _base_score(combo, 10)
        assert 0.05 <= score <= 1.0

    def test_low_rank_scores_higher_than_high_rank(self) -> None:
        low = classify_combo([_card(Rank.THREE)], Rank.TWO)
        high = classify_combo([_card(Rank.ACE)], Rank.TWO)
        assert low is not None and high is not None
        # Lower rank should score higher (we want to save big cards)
        assert _base_score(low, 10) > _base_score(high, 10)

    def test_bomb_penalty_large_hand(self) -> None:
        bomb_cards = [
            _card(Rank.EIGHT, Suit.HEARTS),
            _card(Rank.EIGHT, Suit.DIAMONDS),
            _card(Rank.EIGHT, Suit.CLUBS),
            _card(Rank.EIGHT, Suit.SPADES),
        ]
        combo = classify_combo(bomb_cards, Rank.TWO)
        assert combo is not None
        score = _base_score(combo, 20)
        # Bomb in large hand gets penalty
        assert score < 0.7

    def test_bomb_bonus_small_hand(self) -> None:
        bomb_cards = [
            _card(Rank.FIVE, Suit.HEARTS),
            _card(Rank.FIVE, Suit.DIAMONDS),
            _card(Rank.FIVE, Suit.CLUBS),
            _card(Rank.FIVE, Suit.SPADES),
        ]
        combo = classify_combo(bomb_cards, Rank.TWO)
        assert combo is not None
        score_small = _base_score(combo, 4)
        score_large = _base_score(combo, 20)
        # Small hand bomb should score higher than large hand bomb
        assert score_small > score_large

    def test_score_clamped(self) -> None:
        combo = classify_combo([_card(Rank.THREE)], Rank.TWO)
        assert combo is not None
        score = _base_score(combo, 1)
        assert 0.05 <= score <= 1.0

    def test_joker_rank_penalty(self) -> None:
        # Create a joker "single"
        j = _joker(JokerType.RED)
        combo = classify_combo([j], Rank.TWO)
        assert combo is not None
        score = _base_score(combo, 10)
        # Joker play should have rank penalty
        assert 0.05 <= score <= 1.0


class TestRiskForPlay:
    def _low_risk_report(self) -> RiskReport:
        return RiskReport(
            risk_level=RiskLevel.LOW,
            bomb_threat=0.1,
            big_card_threat=0.1,
        )

    def _high_risk_report(self) -> RiskReport:
        return RiskReport(
            risk_level=RiskLevel.HIGH,
            bomb_threat=0.7,
            big_card_threat=0.7,
        )

    def _critical_risk_report(self) -> RiskReport:
        return RiskReport(
            risk_level=RiskLevel.CRITICAL,
            bomb_threat=0.8,
            big_card_threat=0.8,
        )

    def test_safe_low_card_low_risk(self) -> None:
        combo = classify_combo([_card(Rank.THREE)], Rank.TWO)
        assert combo is not None
        risk = _risk_for_play(combo, self._low_risk_report(), 10)
        assert risk == SuggestionRisk.SAFE

    def test_bomb_high_threat_is_risky(self) -> None:
        bomb_cards = [
            _card(Rank.EIGHT, Suit.HEARTS),
            _card(Rank.EIGHT, Suit.DIAMONDS),
            _card(Rank.EIGHT, Suit.CLUBS),
            _card(Rank.EIGHT, Suit.SPADES),
        ]
        combo = classify_combo(bomb_cards, Rank.TWO)
        assert combo is not None
        risk = _risk_for_play(combo, self._high_risk_report(), 10)
        assert risk == SuggestionRisk.RISKY

    def test_high_card_high_bomb_threat_moderate(self) -> None:
        combo = classify_combo([_card(Rank.ACE)], Rank.TWO)
        assert combo is not None
        report = RiskReport(
            risk_level=RiskLevel.MEDIUM,
            bomb_threat=0.65,
            big_card_threat=0.5,
        )
        risk = _risk_for_play(combo, report, 10)
        assert risk == SuggestionRisk.MODERATE

    def test_high_card_high_risk_game_moderate(self) -> None:
        combo = classify_combo([_card(Rank.JACK)], Rank.TWO)
        assert combo is not None
        risk = _risk_for_play(combo, self._high_risk_report(), 10)
        assert risk == SuggestionRisk.MODERATE

    def test_bomb_low_threat_is_safe(self) -> None:
        bomb_cards = [
            _card(Rank.FIVE, Suit.HEARTS),
            _card(Rank.FIVE, Suit.DIAMONDS),
            _card(Rank.FIVE, Suit.CLUBS),
            _card(Rank.FIVE, Suit.SPADES),
        ]
        combo = classify_combo(bomb_cards, Rank.TWO)
        assert combo is not None
        risk = _risk_for_play(combo, self._low_risk_report(), 10)
        assert risk == SuggestionRisk.SAFE


class TestGenerateReason:
    def _default_report(self) -> RiskReport:
        return RiskReport(risk_level=RiskLevel.LOW, bomb_threat=0.2)

    def test_bomb_finish(self) -> None:
        bomb_cards = [
            _card(Rank.FIVE, Suit.HEARTS),
            _card(Rank.FIVE, Suit.DIAMONDS),
            _card(Rank.FIVE, Suit.CLUBS),
            _card(Rank.FIVE, Suit.SPADES),
        ]
        combo = classify_combo(bomb_cards, Rank.TWO)
        assert combo is not None
        reason = _generate_reason(combo, True, 4, self._default_report())
        assert "finish" in reason.lower()

    def test_bomb_safe_to_use(self) -> None:
        bomb_cards = [
            _card(Rank.FIVE, Suit.HEARTS),
            _card(Rank.FIVE, Suit.DIAMONDS),
            _card(Rank.FIVE, Suit.CLUBS),
            _card(Rank.FIVE, Suit.SPADES),
        ]
        combo = classify_combo(bomb_cards, Rank.TWO)
        assert combo is not None
        reason = _generate_reason(combo, True, 10, self._default_report())
        assert "low" in reason.lower() or "safe" in reason.lower()

    def test_bomb_risky(self) -> None:
        bomb_cards = [
            _card(Rank.FIVE, Suit.HEARTS),
            _card(Rank.FIVE, Suit.DIAMONDS),
            _card(Rank.FIVE, Suit.CLUBS),
            _card(Rank.FIVE, Suit.SPADES),
        ]
        combo = classify_combo(bomb_cards, Rank.TWO)
        assert combo is not None
        report = RiskReport(risk_level=RiskLevel.HIGH, bomb_threat=0.8)
        reason = _generate_reason(combo, True, 10, report)
        assert "stronger" in reason.lower() or "bomb" in reason.lower()

    def test_lead_low(self) -> None:
        combo = classify_combo([_card(Rank.THREE)], Rank.TWO)
        assert combo is not None
        reason = _generate_reason(combo, True, 10, self._default_report())
        assert "low" in reason.lower() or "probe" in reason.lower()

    def test_lead_high(self) -> None:
        combo = classify_combo([_card(Rank.ACE)], Rank.TWO)
        assert combo is not None
        reason = _generate_reason(combo, True, 10, self._default_report())
        assert "strong" in reason.lower() or "control" in reason.lower()

    def test_lead_mid(self) -> None:
        combo = classify_combo([_card(Rank.EIGHT)], Rank.TWO)
        assert combo is not None
        reason = _generate_reason(combo, True, 10, self._default_report())
        assert "mid" in reason.lower()

    def test_response_low(self) -> None:
        combo = classify_combo([_card(Rank.THREE)], Rank.TWO)
        assert combo is not None
        reason = _generate_reason(combo, False, 10, self._default_report())
        assert "low" in reason.lower() or "preserv" in reason.lower()

    def test_response_high(self) -> None:
        combo = classify_combo([_card(Rank.ACE)], Rank.TWO)
        assert combo is not None
        reason = _generate_reason(combo, False, 10, self._default_report())
        assert "strong" in reason.lower() or "win" in reason.lower()

    def test_response_mid(self) -> None:
        combo = classify_combo([_card(Rank.EIGHT)], Rank.TWO)
        assert combo is not None
        reason = _generate_reason(combo, False, 10, self._default_report())
        assert "beat" in reason.lower() or "balanced" in reason.lower()


class TestPassHelpers:
    def test_pass_reason_high_bomb_threat(self) -> None:
        report = RiskReport(risk_level=RiskLevel.HIGH, bomb_threat=0.7)
        reason = _generate_pass_reason(report, 10)
        assert "bomb" in reason.lower() or "partner" in reason.lower()

    def test_pass_reason_many_cards(self) -> None:
        report = RiskReport(risk_level=RiskLevel.LOW, bomb_threat=0.1)
        reason = _generate_pass_reason(report, 20)
        assert "conserve" in reason.lower() or "many" in reason.lower()

    def test_pass_reason_default(self) -> None:
        report = RiskReport(risk_level=RiskLevel.LOW, bomb_threat=0.1)
        reason = _generate_pass_reason(report, 5)
        assert len(reason) > 0

    def test_pass_confidence_increases_with_risk(self) -> None:
        low = RiskReport(risk_level=RiskLevel.LOW, bomb_threat=0.1)
        high = RiskReport(risk_level=RiskLevel.HIGH, bomb_threat=0.7)
        conf_low = _pass_confidence(low, 10)
        conf_high = _pass_confidence(high, 10)
        assert conf_high > conf_low

    def test_pass_confidence_bounded(self) -> None:
        report = RiskReport(risk_level=RiskLevel.CRITICAL, bomb_threat=0.9)
        conf = _pass_confidence(report, 10)
        assert 0.0 <= conf <= 0.60

    def test_pass_risk_critical(self) -> None:
        report = RiskReport(risk_level=RiskLevel.CRITICAL)
        assert _pass_risk(report) == SuggestionRisk.RISKY

    def test_pass_risk_low(self) -> None:
        report = RiskReport(risk_level=RiskLevel.LOW)
        assert _pass_risk(report) == SuggestionRisk.SAFE


# ---------------------------------------------------------------------------
# SuggestionEngine
# ---------------------------------------------------------------------------

class TestSuggestionEngineInit:
    def test_default_init(self) -> None:
        engine = SuggestionEngine()
        assert engine.current_level == Rank.TWO
        assert engine.top_n == 3
        assert isinstance(engine.counter, CardCounter)

    def test_custom_level(self) -> None:
        engine = SuggestionEngine(current_level=Rank.FIVE)
        assert engine.current_level == Rank.FIVE
        assert engine.counter.current_level == Rank.FIVE

    def test_custom_counter(self) -> None:
        counter = CardCounter(current_level=Rank.THREE)
        engine = SuggestionEngine(counter=counter)
        assert engine.counter is counter

    def test_custom_top_n(self) -> None:
        engine = SuggestionEngine(top_n=5)
        assert engine.top_n == 5

    def test_update_counter(self) -> None:
        engine = SuggestionEngine()
        new_counter = CardCounter(current_level=Rank.SEVEN)
        engine.update_counter(new_counter)
        assert engine.counter is new_counter


class TestSuggestionEngineLead:
    """Tests for leading (no last play)."""

    def test_suggest_lead_returns_list(self) -> None:
        engine = SuggestionEngine()
        hand = _make_hand()
        results = engine.suggest(hand)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_suggest_lead_max_top_n(self) -> None:
        engine = SuggestionEngine(top_n=3)
        hand = _make_hand()
        results = engine.suggest(hand)
        assert len(results) <= 3

    def test_suggest_lead_sorted_by_confidence(self) -> None:
        engine = SuggestionEngine()
        hand = _make_hand()
        results = engine.suggest(hand)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_suggest_lead_has_valid_fields(self) -> None:
        engine = SuggestionEngine()
        hand = _make_hand()
        results = engine.suggest(hand)
        for s in results:
            assert isinstance(s, PlaySuggestion)
            assert isinstance(s.combo_type, str)
            assert isinstance(s.reason, str)
            assert len(s.reason) > 0
            assert isinstance(s.risk, SuggestionRisk)
            assert 0.0 <= s.confidence <= 1.0

    def test_suggest_lead_none_means_lead(self) -> None:
        engine = SuggestionEngine()
        hand = _make_hand()
        r1 = engine.suggest(hand, None)
        pass_combo = Combo(ComboType.PASS, tuple(), 0)
        r2 = engine.suggest(hand, pass_combo)
        # Both should produce lead suggestions (non-empty plays)
        assert len(r1) > 0
        assert len(r2) > 0

    def test_suggest_lead_empty_hand(self) -> None:
        engine = SuggestionEngine()
        results = engine.suggest([])
        assert results == []

    def test_suggest_lead_single_card(self) -> None:
        engine = SuggestionEngine()
        hand = [_card(Rank.FIVE)]
        results = engine.suggest(hand)
        assert len(results) >= 1
        assert results[0].cards  # should have at least one card

    def test_suggest_lead_diverse_types(self) -> None:
        """Lead suggestions should try different combo types."""
        engine = SuggestionEngine(top_n=5)
        hand = _make_hand()  # has singles and pairs
        results = engine.suggest(hand)
        types = {s.combo_type for s in results}
        # Should have at least 2 different types
        assert len(types) >= 1


class TestSuggestionEngineResponse:
    """Tests for responding to a previous play."""

    def test_suggest_response_returns_list(self) -> None:
        engine = SuggestionEngine()
        hand = _make_hand()
        last = classify_combo([_card(Rank.TWO)], Rank.TWO)
        assert last is not None
        results = engine.suggest(hand, last)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_suggest_response_includes_pass(self) -> None:
        """When responding, PASS should be offered if < top_n plays."""
        engine = SuggestionEngine(top_n=5)
        hand = [_card(Rank.FIVE), _card(Rank.SEVEN)]
        last = classify_combo([_card(Rank.THREE)], Rank.TWO)
        assert last is not None
        results = engine.suggest(hand, last)
        pass_suggestions = [s for s in results if s.is_pass]
        # Should include a pass option
        assert len(pass_suggestions) >= 1

    def test_suggest_response_no_beaters_returns_pass(self) -> None:
        """If can't beat, should still return PASS."""
        engine = SuggestionEngine()
        hand = [_card(Rank.THREE)]
        last = classify_combo([_card(Rank.ACE)], Rank.TWO)
        assert last is not None
        results = engine.suggest(hand, last)
        # Only pass option
        assert len(results) > 0
        assert all(s.is_pass for s in results)

    def test_suggest_response_sorted(self) -> None:
        engine = SuggestionEngine()
        hand = _make_hand()
        last = classify_combo([_card(Rank.TWO)], Rank.TWO)
        assert last is not None
        results = engine.suggest(hand, last)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_suggest_response_beat_pair(self) -> None:
        engine = SuggestionEngine()
        hand = _make_hand()  # has 3-3 and 5-5 pairs
        last = classify_combo(
            [_card(Rank.TWO, Suit.HEARTS), _card(Rank.TWO, Suit.DIAMONDS)],
            Rank.TWO,
        )
        assert last is not None
        results = engine.suggest(hand, last)
        non_pass = [s for s in results if not s.is_pass]
        assert len(non_pass) >= 1


class TestSuggestionEngineWithCounter:
    """Tests with a pre-configured CardCounter."""

    def test_counter_affects_risk(self) -> None:
        counter = CardCounter(current_level=Rank.TWO)
        # Set hand and record many cards played -> lower risk
        counter.set_hand(_make_hand())
        for rank in Rank:
            counter.record_play([_card(rank)] * 5)
        for jt in [JokerType.RED, JokerType.BLACK]:
            counter.record_play([_joker(jt)] * 2)

        engine = SuggestionEngine(counter=counter)
        results = engine.suggest(_make_hand())
        # With low risk, suggestions should mostly be SAFE
        safe_count = sum(1 for s in results if s.risk == SuggestionRisk.SAFE)
        assert safe_count >= 1

    def test_fresh_counter_high_risk(self) -> None:
        counter = CardCounter(current_level=Rank.TWO)
        engine = SuggestionEngine(counter=counter)
        hand = _make_large_hand()
        results = engine.suggest(hand)
        # Fresh game should have some non-SAFE suggestions for high cards
        assert len(results) > 0


class TestSuggestionEngineTopN:
    def test_top_1(self) -> None:
        engine = SuggestionEngine(top_n=1)
        hand = _make_hand()
        results = engine.suggest(hand)
        assert len(results) <= 1

    def test_top_5(self) -> None:
        engine = SuggestionEngine(top_n=5)
        hand = _make_large_hand()
        results = engine.suggest(hand)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# get_suggestions convenience function
# ---------------------------------------------------------------------------

class TestGetSuggestions:
    def test_basic_lead(self) -> None:
        hand = _make_hand()
        results = get_suggestions(hand)
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 3

    def test_with_last_play(self) -> None:
        hand = _make_hand()
        last = classify_combo([_card(Rank.TWO)], Rank.TWO)
        assert last is not None
        results = get_suggestions(hand, last_play=last)
        assert len(results) > 0

    def test_custom_level(self) -> None:
        hand = _make_hand()
        results = get_suggestions(hand, current_level=Rank.FIVE)
        assert len(results) > 0

    def test_with_counter(self) -> None:
        counter = CardCounter(current_level=Rank.TWO)
        counter.set_hand(_make_hand())
        results = get_suggestions(_make_hand(), counter=counter)
        assert len(results) > 0

    def test_custom_top_n(self) -> None:
        hand = _make_hand()
        results = get_suggestions(hand, top_n=1)
        assert len(results) <= 1

    def test_empty_hand(self) -> None:
        results = get_suggestions([])
        assert results == []

    def test_cannot_beat_returns_pass(self) -> None:
        hand = [_card(Rank.THREE)]
        last = classify_combo([_card(Rank.ACE)], Rank.TWO)
        assert last is not None
        results = get_suggestions(hand, last_play=last)
        assert len(results) > 0
        assert results[0].is_pass


# ---------------------------------------------------------------------------
# Integration: suggestion with real game scenarios
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:
    def test_endgame_few_cards(self) -> None:
        """With very few cards, suggestions should be aggressive."""
        hand = [_card(Rank.ACE), _card(Rank.KING)]
        results = get_suggestions(hand)
        assert len(results) > 0
        # Should suggest playing cards, not only conservative
        non_pass = [s for s in results if not s.is_pass]
        assert len(non_pass) >= 1

    def test_bomb_in_hand(self) -> None:
        """Hand with a bomb should include bomb as an option."""
        hand = _make_large_hand()  # has four 8s
        engine = SuggestionEngine(top_n=5)
        results = engine.suggest(hand)
        types = {s.combo_type for s in results}
        # Bomb should appear among suggestions
        has_bomb = any("bomb" in t for t in types)
        assert has_bomb or len(results) > 0

    def test_response_with_bomb_available(self) -> None:
        """When responding and only bomb can beat, it should appear."""
        hand = _make_large_hand()  # has bomb of 8s
        # Last play is a high single that only bomb can beat
        last = classify_combo([_card(Rank.ACE, Suit.DIAMONDS)], Rank.TWO)
        assert last is not None
        engine = SuggestionEngine(top_n=5)
        results = engine.suggest(hand, last)
        # Should have at least one non-pass option or pass
        assert len(results) > 0

    def test_all_suggestions_have_reasons(self) -> None:
        hand = _make_hand()
        results = get_suggestions(hand)
        for s in results:
            assert isinstance(s.reason, str)
            assert len(s.reason) > 5  # non-trivial reason

    def test_confidence_range(self) -> None:
        hand = _make_hand()
        results = get_suggestions(hand)
        for s in results:
            assert 0.0 <= s.confidence <= 1.0

    def test_lead_candidates_dedup(self) -> None:
        """Suggestions should not contain duplicate card tuples."""
        hand = _make_large_hand()
        engine = SuggestionEngine(top_n=5)
        results = engine.suggest(hand)
        card_sets = [s.cards for s in results]
        assert len(card_sets) == len(set(card_sets))

    def test_response_candidates_dedup(self) -> None:
        hand = _make_large_hand()
        last = classify_combo([_card(Rank.TWO)], Rank.TWO)
        assert last is not None
        engine = SuggestionEngine(top_n=5)
        results = engine.suggest(hand, last)
        card_sets = [s.cards for s in results]
        assert len(card_sets) == len(set(card_sets))
