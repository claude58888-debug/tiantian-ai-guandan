"""Tests for Atom 1.4 - GameState machine."""
import pytest
from unittest.mock import patch
from guandan.models import Card, Rank, Suit, JokerType, Hand
from guandan.combos import Combo, ComboType, classify_combo
from guandan.game import GameState, Phase, Player, StateTransactionError


class TestPlayer:
    def test_player_defaults(self):
        p = Player(index=0)
        assert p.index == 0
        assert p.team == 0
        assert not p.finished
        assert p.finish_order == -1
        assert len(p.hand) == 0

    def test_player_repr(self):
        p = Player(index=2, team=1)
        r = repr(p)
        assert 'Player(2' in r
        assert 'team=1' in r


class TestGameStateInit:
    def test_default_players(self):
        gs = GameState()
        assert len(gs.players) == 4
        assert gs.players[0].team == 0
        assert gs.players[1].team == 1
        assert gs.players[2].team == 0
        assert gs.players[3].team == 1

    def test_initial_phase(self):
        gs = GameState()
        assert gs.phase == Phase.DEALING

    def test_repr(self):
        gs = GameState()
        r = repr(gs)
        assert 'GameState' in r
        assert 'dealing' in r


class TestDeal:
    def test_deal_gives_27_cards(self):
        gs = GameState()
        gs.deal(seed=42)
        for p in gs.players:
            assert len(p.hand) == 27

    def test_deal_sets_playing(self):
        gs = GameState()
        gs.deal(seed=42)
        assert gs.phase == Phase.PLAYING

    def test_deal_resets_state(self):
        gs = GameState()
        gs.deal(seed=42)
        assert gs.last_play is None
        assert gs.pass_count == 0
        assert gs.finish_order == []


class TestPlayCards:
    def setup_method(self):
        self.gs = GameState()
        self.gs.deal(seed=42)

    def test_wrong_phase(self):
        self.gs.phase = Phase.DEALING
        assert not self.gs.play_cards(0, [])

    def test_wrong_player(self):
        assert not self.gs.play_cards(1, [])  # player 0's turn

    def test_cannot_pass_on_lead(self):
        # last_play is None, must lead
        assert not self.gs.play_cards(0, [])

    def test_play_single_card(self):
        card = self.gs.players[0].hand.cards[0]
        ok = self.gs.play_cards(0, [card])
        assert ok
        assert self.gs.last_play is not None
        assert self.gs.last_player == 0

    def test_invalid_combo(self):
        # Pick two cards that don't form a valid pair
        cards = self.gs.players[0].hand.cards
        # Find two cards with different ranks
        c1, c2 = None, None
        for i, c in enumerate(cards):
            for j, d in enumerate(cards):
                if i != j and not c.is_joker and not d.is_joker and c.rank != d.rank:
                    c1, c2 = c, d
                    break
            if c1:
                break
        if c1 and c2:
            ok = self.gs.play_cards(0, [c1, c2])
            assert not ok

    def test_cards_not_in_hand(self):
        fake = Card(rank=Rank.ACE, suit=Suit.HEARTS, deck_id=99)
        ok = self.gs.play_cards(0, [fake])
        assert not ok

    def test_pass_after_play(self):
        # Player 0 leads
        card0 = self.gs.players[0].hand.cards[0]
        self.gs.play_cards(0, [card0])
        # Player 1 passes
        ok = self.gs.play_cards(self.gs.current_player, [])
        assert ok

    def test_all_pass_resets_trick(self):
        card0 = self.gs.players[0].hand.cards[0]
        self.gs.play_cards(0, [card0])
        # All 3 others pass
        for _ in range(3):
            cp = self.gs.current_player
            self.gs.play_cards(cp, [])
        assert self.gs.last_play is None
        assert self.gs.current_player == 0  # trick winner leads


class TestFinishAndRoundResult:
    def test_round_result_not_finished(self):
        gs = GameState()
        assert gs.get_round_result() == (0, 0)

    def test_double_kill_team0(self):
        gs = GameState()
        gs.finish_order = [0, 2, 1, 3]
        gs.players[0].team = 0
        gs.players[2].team = 0
        gs.players[1].team = 1
        gs.players[3].team = 1
        assert gs.get_round_result() == (3, 0)

    def test_double_kill_team1(self):
        gs = GameState()
        gs.finish_order = [1, 3, 0, 2]
        assert gs.get_round_result() == (0, 3)

    def test_first_third_team0(self):
        gs = GameState()
        gs.finish_order = [0, 1, 2, 3]
        assert gs.get_round_result() == (2, 0)

    def test_first_fourth(self):
        gs = GameState()
        gs.finish_order = [0, 1, 3, 2]
        assert gs.get_round_result() == (1, 0)

    def test_first_fourth_team1(self):
        gs = GameState()
        gs.finish_order = [1, 0, 2, 3]
        assert gs.get_round_result() == (0, 1)


class TestAdvanceLevels:
    def test_advance_team0(self):
        gs = GameState()
        gs.finish_order = [0, 2, 1, 3]  # double kill
        gs.advance_levels()
        assert gs.team_levels[0].value == Rank.FIVE.value

    def test_advance_cap_at_ace(self):
        gs = GameState()
        gs.team_levels[0] = Rank.KING
        gs.finish_order = [0, 2, 1, 3]  # +3 from king
        gs.advance_levels()
        # Should cap at ACE
        assert gs.team_levels[0].value <= Rank.ACE.value + 1


class TestActivePlayerCount:
    def test_all_active(self):
        gs = GameState()
        assert gs.active_player_count() == 4

    def test_one_finished(self):
        gs = GameState()
        gs.players[0].finished = True
        assert gs.active_player_count() == 3


class TestNextActivePlayer:
    def test_skip_finished(self):
        gs = GameState()
        gs.players[1].finished = True
        nxt = gs._next_active_player(0)
        assert nxt == 2

    def test_wrap_around(self):
        gs = GameState()
        nxt = gs._next_active_player(3)
        assert nxt == 0


class TestPhase:
    def test_phase_values(self):
        assert Phase.DEALING.value == 'dealing'
        assert Phase.PLAYING.value == 'playing'
        assert Phase.ROUND_END.value == 'round_end'
        assert Phase.GAME_OVER.value == 'game_over'


class TestHistory:
    def test_history_recorded(self):
        gs = GameState()
        gs.deal(seed=42)
        card = gs.players[0].hand.cards[0]
        gs.play_cards(0, [card])
        assert len(gs.history) == 1
        assert gs.history[0][0] == 0

    def test_pass_recorded(self):
        gs = GameState()
        gs.deal(seed=42)
        card = gs.players[0].hand.cards[0]
        gs.play_cards(0, [card])
        cp = gs.current_player
        gs.play_cards(cp, [])
        assert gs.history[-1][1] is None


# ── V0.3 C-3: State machine atomicity / rollback tests ───────────────


class TestSnapshotRestore:
    """Tests for _snapshot and _restore helpers."""

    def test_snapshot_captures_state(self):
        gs = GameState()
        gs.deal(seed=42)
        snap = gs._snapshot()
        assert snap['phase'] == Phase.PLAYING
        assert len(snap['players']) == 4
        assert snap['pass_count'] == 0

    def test_restore_reverts_state(self):
        gs = GameState()
        gs.deal(seed=42)
        snap = gs._snapshot()
        original_player = gs.current_player

        # Mutate state
        card = gs.players[0].hand.cards[0]
        gs.play_cards(0, [card])
        assert gs.current_player != original_player

        # Restore
        gs._restore(snap)
        assert gs.current_player == original_player
        assert len(gs.history) == 0

    def test_snapshot_is_deep_copy(self):
        gs = GameState()
        gs.deal(seed=42)
        snap = gs._snapshot()
        original_hand_len = len(gs.players[0].hand)

        card = gs.players[0].hand.cards[0]
        gs.play_cards(0, [card])

        # Snapshot players should be unaffected
        assert len(snap['players'][0].hand) == original_hand_len


class TestPlayCardsAtomicity:
    """Tests that play_cards rolls back on unexpected exceptions."""

    def test_rollback_on_exception(self):
        gs = GameState()
        gs.deal(seed=42)
        card = gs.players[0].hand.cards[0]
        original_hand_len = len(gs.players[0].hand)

        # Patch _play_cards_inner to raise after partial mutation
        with patch.object(
            GameState, '_play_cards_inner',
            side_effect=RuntimeError('simulated crash'),
        ):
            with pytest.raises(StateTransactionError):
                gs.play_cards(0, [card])

        # State should be fully restored
        assert len(gs.players[0].hand) == original_hand_len
        assert len(gs.history) == 0
        assert gs.current_player == 0

    def test_successful_play_not_rolled_back(self):
        gs = GameState()
        gs.deal(seed=42)
        card = gs.players[0].hand.cards[0]
        ok = gs.play_cards(0, [card])
        assert ok
        assert len(gs.history) == 1

    def test_failed_validation_no_snapshot(self):
        gs = GameState()
        gs.deal(seed=42)
        # Wrong player doesn't even create a snapshot
        ok = gs.play_cards(1, [])
        assert not ok
        assert len(gs.history) == 0

    def test_rollback_preserves_finish_order(self):
        gs = GameState()
        gs.deal(seed=42)
        original_finish_order = list(gs.finish_order)

        with patch.object(
            GameState, '_play_cards_inner',
            side_effect=RuntimeError('boom'),
        ):
            with pytest.raises(StateTransactionError):
                gs.play_cards(0, [gs.players[0].hand.cards[0]])

        assert gs.finish_order == original_finish_order


class TestStateTransactionError:
    def test_exception_is_raised(self):
        gs = GameState()
        gs.deal(seed=42)
        with patch.object(
            GameState, '_play_cards_inner',
            side_effect=ValueError('inner error'),
        ):
            with pytest.raises(StateTransactionError, match='rolled back'):
                gs.play_cards(0, [gs.players[0].hand.cards[0]])

    def test_original_exception_chained(self):
        gs = GameState()
        gs.deal(seed=42)
        with patch.object(
            GameState, '_play_cards_inner',
            side_effect=ValueError('inner'),
        ):
            with pytest.raises(StateTransactionError) as exc_info:
                gs.play_cards(0, [gs.players[0].hand.cards[0]])
            assert exc_info.value.__cause__ is not None
