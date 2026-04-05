"""Tests for guandan.card_counter module (Atom 4.2)."""
from __future__ import annotations

import unittest

from guandan.models import Card, Rank, Suit, JokerType
from guandan.card_counter import (
    CardCounter,
    KeyCardStatus,
    RiskLevel,
    RiskReport,
    TOTAL_CARDS,
    TOTAL_JOKERS,
    TOTAL_PER_RANK,
)


def _card(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank=rank, suit=suit)


def _joker(color: JokerType) -> Card:
    return Card(joker=color)


class TestKeyCardStatus(unittest.TestCase):
    def test_remaining(self) -> None:
        s = KeyCardStatus("test", total=8, played=3, in_hand=2)
        self.assertEqual(s.remaining, 3)

    def test_remaining_clamped(self) -> None:
        s = KeyCardStatus("test", total=4, played=3, in_hand=3)
        self.assertEqual(s.remaining, 0)

    def test_depletion_ratio(self) -> None:
        s = KeyCardStatus("test", total=8, played=4, in_hand=2)
        self.assertAlmostEqual(s.depletion_ratio, 0.75)

    def test_depletion_ratio_zero_total(self) -> None:
        s = KeyCardStatus("test", total=0)
        self.assertAlmostEqual(s.depletion_ratio, 1.0)


class TestCardCounterBasics(unittest.TestCase):
    def test_init_defaults(self) -> None:
        cc = CardCounter()
        self.assertEqual(cc.current_level, Rank.TWO)
        self.assertEqual(cc.total_played, 0)
        self.assertEqual(cc.play_history, [])

    def test_set_hand(self) -> None:
        cc = CardCounter()
        hand = [_card(Rank.ACE), _card(Rank.KING), _joker(JokerType.RED)]
        cc.set_hand(hand)
        statuses = cc.get_key_card_statuses()
        self.assertEqual(statuses["ace"].in_hand, 1)
        self.assertEqual(statuses["king"].in_hand, 1)
        self.assertEqual(statuses["joker"].in_hand, 1)

    def test_record_play(self) -> None:
        cc = CardCounter()
        cards = [_card(Rank.ACE), _card(Rank.ACE, Suit.HEARTS)]
        cc.record_play(cards)
        self.assertEqual(cc.total_played, 2)
        self.assertEqual(len(cc.play_history), 1)
        statuses = cc.get_key_card_statuses()
        self.assertEqual(statuses["ace"].played, 2)

    def test_multiple_plays(self) -> None:
        cc = CardCounter()
        cc.record_play([_card(Rank.THREE)])
        cc.record_play([_card(Rank.THREE), _card(Rank.THREE)])
        self.assertEqual(cc.total_played, 3)
        self.assertEqual(len(cc.play_history), 2)

    def test_reset(self) -> None:
        cc = CardCounter()
        cc.set_hand([_card(Rank.ACE)])
        cc.record_play([_card(Rank.KING)])
        cc.reset()
        self.assertEqual(cc.total_played, 0)
        self.assertEqual(cc.play_history, [])


class TestKeyCardStatuses(unittest.TestCase):
    def test_joker_tracking(self) -> None:
        cc = CardCounter()
        cc.set_hand([_joker(JokerType.RED)])
        cc.record_play([_joker(JokerType.BLACK)])
        s = cc.get_key_card_statuses()
        self.assertEqual(s["joker"].total, TOTAL_JOKERS)
        self.assertEqual(s["joker"].played, 1)
        self.assertEqual(s["joker"].in_hand, 1)
        self.assertEqual(s["joker"].remaining, 2)

    def test_level_card_tracking(self) -> None:
        cc = CardCounter(current_level=Rank.FIVE)
        cc.record_play([_card(Rank.FIVE), _card(Rank.FIVE)])
        s = cc.get_key_card_statuses()
        self.assertEqual(s["level"].name, "Level (FIVE)")
        self.assertEqual(s["level"].played, 2)
        self.assertEqual(s["level"].remaining, 6)

    def test_all_key_types_present(self) -> None:
        cc = CardCounter()
        s = cc.get_key_card_statuses()
        self.assertIn("joker", s)
        self.assertIn("ace", s)
        self.assertIn("level", s)
        self.assertIn("king", s)


class TestBombProbability(unittest.TestCase):
    def test_fresh_game_high_probability(self) -> None:
        cc = CardCounter()
        prob = cc.estimate_bomb_probability()
        # At start, all ranks unseen -> high probability
        self.assertGreater(prob, 0.7)

    def test_all_seen_low_probability(self) -> None:
        cc = CardCounter()
        # Play 5+ of every rank so no quad is possible
        for rank in Rank:
            for _ in range(5):
                cc.record_play([_card(rank)])
        # Also see all jokers
        cc.record_play([_joker(JokerType.RED)])
        cc.record_play([_joker(JokerType.BLACK)])
        prob = cc.estimate_bomb_probability()
        self.assertLess(prob, 0.3)

    def test_partial_info(self) -> None:
        cc = CardCounter()
        # See some cards of several ranks
        for rank in [Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX]:
            cc.record_play([_card(rank)] * 5)
        prob = cc.estimate_bomb_probability()
        # Should be lower than fresh but still significant
        self.assertGreater(prob, 0.3)
        self.assertLess(prob, 1.0)


class TestBigCardThreat(unittest.TestCase):
    def test_fresh_game_full_threat(self) -> None:
        cc = CardCounter()
        threat = cc.estimate_big_card_threat()
        self.assertAlmostEqual(threat, 1.0)

    def test_all_big_cards_accounted(self) -> None:
        cc = CardCounter()
        # Put all aces, kings in hand/played
        for _ in range(8):
            cc.record_play([_card(Rank.ACE)])
            cc.record_play([_card(Rank.KING)])
        for jt in [JokerType.RED, JokerType.BLACK]:
            cc.record_play([_joker(jt)])
            cc.record_play([_joker(jt)])
        threat = cc.estimate_big_card_threat()
        self.assertAlmostEqual(threat, 0.0)

    def test_some_in_hand_reduces_threat(self) -> None:
        cc = CardCounter()
        cc.set_hand([
            _card(Rank.ACE), _card(Rank.ACE),
            _card(Rank.KING), _joker(JokerType.RED),
        ])
        threat = cc.estimate_big_card_threat()
        self.assertLess(threat, 1.0)
        self.assertGreater(threat, 0.5)


class TestRiskAssessment(unittest.TestCase):
    def test_fresh_game_risk(self) -> None:
        cc = CardCounter()
        report = cc.assess_risk()
        self.assertIsInstance(report, RiskReport)
        # Fresh game should have high risk
        self.assertIn(report.risk_level, [RiskLevel.HIGH, RiskLevel.CRITICAL])
        self.assertGreater(len(report.warnings), 0)
        self.assertIn("Risk:", report.summary)

    def test_low_risk_scenario(self) -> None:
        cc = CardCounter()
        # See most cards
        for rank in Rank:
            cc.record_play([_card(rank)] * 6)
        for jt in [JokerType.RED, JokerType.BLACK]:
            cc.record_play([_joker(jt)] * 2)
        cc.set_hand([_card(Rank.ACE)] * 4)
        report = cc.assess_risk()
        self.assertEqual(report.risk_level, RiskLevel.LOW)

    def test_warnings_include_joker_bomb(self) -> None:
        cc = CardCounter()
        report = cc.assess_risk()
        joker_warnings = [w for w in report.warnings if "Joker" in w]
        self.assertGreater(len(joker_warnings), 0)

    def test_key_cards_in_report(self) -> None:
        cc = CardCounter()
        report = cc.assess_risk()
        self.assertIn("joker", report.key_cards)
        self.assertIn("ace", report.key_cards)


class TestFormatDisplay(unittest.TestCase):
    def test_format_returns_string(self) -> None:
        cc = CardCounter()
        output = cc.format_display()
        self.assertIsInstance(output, str)
        self.assertIn("Card Counter", output)
        self.assertIn("Risk Level", output)
        self.assertIn("Key Cards", output)

    def test_format_with_data(self) -> None:
        cc = CardCounter(current_level=Rank.THREE)
        cc.set_hand([_card(Rank.ACE), _joker(JokerType.RED)])
        cc.record_play([_card(Rank.KING)])
        output = cc.format_display()
        self.assertIn("Aces", output)
        self.assertIn("Kings", output)
        self.assertIn("Jokers", output)


class TestConstants(unittest.TestCase):
    def test_total_cards(self) -> None:
        self.assertEqual(TOTAL_CARDS, 108)

    def test_total_jokers(self) -> None:
        self.assertEqual(TOTAL_JOKERS, 4)

    def test_total_per_rank(self) -> None:
        self.assertEqual(TOTAL_PER_RANK, 8)


if __name__ == "__main__":
    unittest.main()
