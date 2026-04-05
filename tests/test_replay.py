"""Tests for guandan.replay module (Atom 4.4)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guandan.replay import (
    ActionType,
    GameReplay,
    GameStats,
    ReplayAnalyzer,
    ReplayRecorder,
    RoundAction,
    TurningPoint,
    save_replay,
    load_replay_summary,
)


class TestRoundAction(unittest.TestCase):
    def test_followed_suggestion_true(self) -> None:
        a = RoundAction(
            round_num=1, player="self", action_type=ActionType.PLAY,
            cards_played=["S6", "H6"], combo_type="pair",
            suggested_cards=["H6", "S6"], suggested_type="pair",
        )
        self.assertTrue(a.followed_suggestion)

    def test_followed_suggestion_false(self) -> None:
        a = RoundAction(
            round_num=1, player="self", action_type=ActionType.PLAY,
            cards_played=["S3", "H3"], combo_type="pair",
            suggested_cards=["S6", "H6"], suggested_type="pair",
        )
        self.assertFalse(a.followed_suggestion)

    def test_followed_suggestion_none(self) -> None:
        a = RoundAction(
            round_num=1, player="self", action_type=ActionType.PLAY,
            cards_played=["S3"],
        )
        self.assertIsNone(a.followed_suggestion)

    def test_followed_suggestion_type_mismatch(self) -> None:
        a = RoundAction(
            round_num=1, player="self", action_type=ActionType.PLAY,
            cards_played=["S6", "H6"], combo_type="pair",
            suggested_cards=["S6", "H6"], suggested_type="straight",
        )
        self.assertFalse(a.followed_suggestion)


class TestGameStats(unittest.TestCase):
    def test_defaults(self) -> None:
        s = GameStats()
        self.assertEqual(s.total_rounds, 0)
        self.assertAlmostEqual(s.pass_rate, 0.0)
        self.assertAlmostEqual(s.bomb_rate, 0.0)
        self.assertAlmostEqual(s.suggestion_follow_rate, 0.0)

    def test_rates(self) -> None:
        s = GameStats(plays=6, passes=4, bombs_used=2,
                      suggestions_followed=3, suggestions_total=5)
        self.assertAlmostEqual(s.pass_rate, 0.4)
        self.assertAlmostEqual(s.bomb_rate, 0.2)
        self.assertAlmostEqual(s.suggestion_follow_rate, 0.6)


class TestReplayRecorder(unittest.TestCase):
    def test_init(self) -> None:
        r = ReplayRecorder(game_id="test1")
        self.assertEqual(r.replay.game_id, "test1")
        self.assertEqual(r.round_count, 0)

    def test_record_action(self) -> None:
        r = ReplayRecorder()
        a = r.record_action(
            player="self", action_type=ActionType.PLAY,
            cards_played=["S3", "H3"], combo_type="pair",
        )
        self.assertEqual(r.round_count, 1)
        self.assertEqual(a.round_num, 1)
        self.assertEqual(len(r.replay.actions), 1)

    def test_multiple_actions(self) -> None:
        r = ReplayRecorder()
        r.record_action(player="self", action_type=ActionType.PLAY)
        r.record_action(player="left", action_type=ActionType.PASS)
        r.record_action(player="self", action_type=ActionType.BOMB,
                        cards_played=["S5", "H5", "C5", "D5"], combo_type="bomb")
        self.assertEqual(r.round_count, 3)

    def test_finish_game(self) -> None:
        r = ReplayRecorder(game_id="g1")
        r.record_action(player="self", action_type=ActionType.PLAY,
                        cards_played=["S3"], combo_type="single")
        r.record_action(player="self", action_type=ActionType.PASS)
        r.record_action(player="self", action_type=ActionType.BOMB,
                        cards_played=["S5", "H5", "C5", "D5"], combo_type="bomb")
        replay = r.finish_game("win")
        self.assertEqual(replay.result, "win")
        self.assertNotEqual(replay.end_time, "")
        self.assertEqual(replay.stats.total_rounds, 3)
        self.assertEqual(replay.stats.plays, 2)  # PLAY + BOMB
        self.assertEqual(replay.stats.passes, 1)
        self.assertEqual(replay.stats.bombs_used, 1)

    def test_turning_points_bomb(self) -> None:
        r = ReplayRecorder()
        r.record_action(player="right", action_type=ActionType.BOMB,
                        cards_played=["SA", "HA", "CA", "DA"], combo_type="bomb")
        replay = r.finish_game("loss")
        self.assertGreater(len(replay.turning_points), 0)
        self.assertEqual(replay.turning_points[0].impact, "negative")

    def test_turning_points_diverged(self) -> None:
        r = ReplayRecorder()
        r.record_action(
            player="self", action_type=ActionType.PLAY,
            cards_played=["S3", "H3"], combo_type="pair",
            suggested_cards=["S6", "H6"], suggested_type="pair",
        )
        replay = r.finish_game("win")
        tps = [tp for tp in replay.turning_points if "Diverged" in tp.description]
        self.assertGreater(len(tps), 0)

    def test_key_rounds_marked(self) -> None:
        r = ReplayRecorder()
        r.record_action(player="self", action_type=ActionType.PLAY)
        r.record_action(player="self", action_type=ActionType.BOMB,
                        cards_played=["S5", "H5", "C5", "D5"], combo_type="bomb")
        r.record_action(player="self", action_type=ActionType.PLAY)
        replay = r.finish_game("win")
        key = [a for a in replay.actions if a.is_key_round]
        self.assertGreater(len(key), 0)

    def test_suggestion_stats(self) -> None:
        r = ReplayRecorder()
        # Followed suggestion
        r.record_action(
            player="self", action_type=ActionType.PLAY,
            cards_played=["S6", "H6"], combo_type="pair",
            suggested_cards=["S6", "H6"], suggested_type="pair",
        )
        # Did not follow
        r.record_action(
            player="self", action_type=ActionType.PLAY,
            cards_played=["S3"], combo_type="single",
            suggested_cards=["S6", "H6"], suggested_type="pair",
        )
        replay = r.finish_game("win")
        self.assertEqual(replay.stats.suggestions_total, 2)
        self.assertEqual(replay.stats.suggestions_followed, 1)
        self.assertAlmostEqual(replay.stats.suggestion_follow_rate, 0.5)


class TestReplayAnalyzer(unittest.TestCase):
    def _make_replay(self) -> GameReplay:
        r = ReplayRecorder(game_id="test")
        r.record_action(player="self", action_type=ActionType.PLAY,
                        cards_played=["S3"], combo_type="single")
        r.record_action(player="self", action_type=ActionType.BOMB,
                        cards_played=["S5", "H5", "C5", "D5"], combo_type="bomb")
        r.record_action(
            player="self", action_type=ActionType.PLAY,
            cards_played=["SA", "HA"], combo_type="pair",
            suggested_cards=["S6", "H6"], suggested_type="pair",
        )
        return r.finish_game("win")

    def test_get_key_rounds(self) -> None:
        replay = self._make_replay()
        analyzer = ReplayAnalyzer(replay)
        keys = analyzer.get_key_rounds()
        self.assertGreater(len(keys), 0)

    def test_get_action_diffs(self) -> None:
        replay = self._make_replay()
        analyzer = ReplayAnalyzer(replay)
        diffs = analyzer.get_action_diffs()
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].cards_played, ["SA", "HA"])

    def test_format_summary(self) -> None:
        replay = self._make_replay()
        analyzer = ReplayAnalyzer(replay)
        summary = analyzer.format_summary()
        self.assertIn("Game Replay", summary)
        self.assertIn("win", summary)
        self.assertIn("Turning Points", summary)
        self.assertIn("Action Diffs", summary)


class TestSaveLoad(unittest.TestCase):
    def test_save_and_load(self) -> None:
        r = ReplayRecorder(game_id="save_test")
        r.record_action(player="self", action_type=ActionType.PLAY,
                        cards_played=["S3"], combo_type="single")
        replay = r.finish_game("win")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.json"
            save_replay(replay, path)
            self.assertTrue(path.exists())

            data = load_replay_summary(path)
            self.assertEqual(data["game_id"], "save_test")
            self.assertEqual(data["result"], "win")
            self.assertIn("stats", data)
            self.assertEqual(data["stats"]["plays"], 1)


if __name__ == "__main__":
    unittest.main()
