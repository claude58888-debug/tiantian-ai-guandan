"""Guandan game state machine (Atom 1.4).

Manages turn order, round progression, team scoring,
and level advancement for a complete Guandan game.

Teams: Players 0,2 vs Players 1,3 (seated across from partner).
Levels: Both teams start at 2, first to pass Ace wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Optional, Sequence, Tuple

from guandan.models import Card, Deck, Hand, Rank
from guandan.combos import Combo, ComboType, classify_combo
from guandan.compare import can_beat


@unique
class Phase(Enum):
    DEALING = 'dealing'
    PLAYING = 'playing'
    ROUND_END = 'round_end'
    GAME_OVER = 'game_over'


@dataclass
class Player:
    index: int
    hand: Hand = field(default_factory=Hand)
    team: int = 0  # 0 or 1
    finished: bool = False
    finish_order: int = -1  # 1st=1, 2nd=2, etc.

    def __repr__(self) -> str:
        return f'Player({self.index}, team={self.team}, cards={len(self.hand)})'


@dataclass
class GameState:
    """Core game state for one round of Guandan."""
    players: List[Player] = field(default_factory=list)
    current_player: int = 0
    current_level: Rank = Rank.TWO
    team_levels: List[Rank] = field(default_factory=lambda: [Rank.TWO, Rank.TWO])
    phase: Phase = Phase.DEALING
    last_play: Optional[Combo] = None
    last_player: int = -1
    pass_count: int = 0
    finish_order: List[int] = field(default_factory=list)
    trick_starter: int = 0  # who started this trick
    history: List[Tuple[int, Optional[Combo]]] = field(default_factory=list)

    def __post_init__(self):
        if not self.players:
            self.players = [
                Player(0, team=0), Player(1, team=1),
                Player(2, team=0), Player(3, team=1)
            ]

    def deal(self, seed: Optional[int] = None) -> None:
        """Shuffle and deal 27 cards to each player."""
        deck = Deck()
        deck.shuffle(seed)
        hands = deck.deal(4)
        for i, cards in enumerate(hands):
            self.players[i].hand = Hand(cards)
            self.players[i].hand.sort(self.current_level)
            self.players[i].finished = False
            self.players[i].finish_order = -1
        self.phase = Phase.PLAYING
        self.finish_order = []
        self.last_play = None
        self.last_player = -1
        self.pass_count = 0

    def _next_active_player(self, from_idx: int) -> int:
        """Find next player who hasn't finished."""
        idx = (from_idx + 1) % 4
        while self.players[idx].finished:
            idx = (idx + 1) % 4
            if idx == from_idx:
                break
        return idx

    def active_player_count(self) -> int:
        return sum(1 for p in self.players if not p.finished)

    def play_cards(self, player_idx: int, cards: Sequence[Card]) -> bool:
        """Attempt to play cards. Returns True if successful."""
        if self.phase != Phase.PLAYING:
            return False
        if player_idx != self.current_player:
            return False

        player = self.players[player_idx]

        # Pass (empty cards)
        if len(cards) == 0:
            if self.last_play is None or self.last_play.combo_type == ComboType.PASS:
                return False  # can't pass when you must lead
            self.pass_count += 1
            self.history.append((player_idx, None))
            # Check if all others passed -> trick winner leads
            active = self.active_player_count()
            if self.pass_count >= active - 1:
                self.current_player = self.last_player
                self.last_play = None
                self.pass_count = 0
                self.trick_starter = self.last_player
            else:
                self.current_player = self._next_active_player(player_idx)
            return True

        # Classify the play
        combo = classify_combo(cards, self.current_level)
        if combo is None:
            return False  # invalid combo

        # Must beat previous play (if any)
        if self.last_play is not None and self.last_play.combo_type != ComboType.PASS:
            if not can_beat(combo, self.last_play):
                return False

        # Remove cards from hand
        try:
            player.hand.remove_cards(list(cards))
        except ValueError:
            return False  # cards not in hand

        self.last_play = combo
        self.last_player = player_idx
        self.pass_count = 0
        self.history.append((player_idx, combo))

        # Check if player finished
        if len(player.hand) == 0:
            player.finished = True
            self.finish_order.append(player_idx)
            player.finish_order = len(self.finish_order)

            if self.active_player_count() <= 1:
                # Add last remaining player
                for p in self.players:
                    if not p.finished:
                        self.finish_order.append(p.index)
                        p.finished = True
                        p.finish_order = len(self.finish_order)
                self.phase = Phase.ROUND_END
                return True

        self.current_player = self._next_active_player(player_idx)
        return True

    def get_round_result(self) -> Tuple[int, int]:
        """Calculate level advancement after round.
        Returns (team0_advance, team1_advance).

        Scoring based on finish order:
        - Both partners 1st+2nd (double kill): advance 3 levels
        - Both partners 1st+3rd: advance 2 levels
        - 1st place partner finishes 4th: opponent advances 1
        """
        if len(self.finish_order) < 4:
            return (0, 0)

        first = self.finish_order[0]
        second = self.finish_order[1]
        first_team = self.players[first].team
        second_team = self.players[second].team

        if first_team == second_team:
            # Double kill
            return (3, 0) if first_team == 0 else (0, 3)

        third = self.finish_order[2]
        third_team = self.players[third].team
        if first_team == third_team:
            return (2, 0) if first_team == 0 else (0, 2)

        return (1, 0) if first_team == 0 else (0, 1)

    def advance_levels(self) -> None:
        """Advance team levels based on round result."""
        t0, t1 = self.get_round_result()
        for _ in range(t0):
            if self.team_levels[0].value < Rank.ACE.value:
                self.team_levels[0] = Rank(self.team_levels[0].value + 1)
        for _ in range(t1):
            if self.team_levels[1].value < Rank.ACE.value:
                self.team_levels[1] = Rank(self.team_levels[1].value + 1)

        # Check game over
        if self.team_levels[0].value > Rank.ACE.value or self.team_levels[1].value > Rank.ACE.value:
            self.phase = Phase.GAME_OVER

    def __repr__(self) -> str:
        return (f'GameState(phase={self.phase.value}, '
                f'current={self.current_player}, '
                f'levels={[r.label() for r in self.team_levels]})')
