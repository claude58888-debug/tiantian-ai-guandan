"""Post-game replay and analysis module (Atom 4.3).

Records game rounds and provides post-game analysis:
- Key round identification
- Action vs suggestion diff analysis
- Turning point detection
- Statistics: bomb usage, pass rate, key card success rate

Per PRD: outputs key rounds, user vs suggested action diffs,
turning points, and simple data stats.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ActionType(Enum):
    """Types of actions in a round."""
    PLAY = auto()
    PASS = auto()
    BOMB = auto()


@dataclass
class RoundAction:
    """A single action taken in a round."""
    round_num: int
    player: str  # "self", "left", "partner", "right"
    action_type: ActionType
    cards_played: List[str] = field(default_factory=list)
    combo_type: str = ""
    suggested_cards: List[str] = field(default_factory=list)
    suggested_type: str = ""
    is_key_round: bool = False
    note: str = ""

    @property
    def followed_suggestion(self) -> Optional[bool]:
        """Whether the player followed the AI suggestion."""
        if not self.suggested_cards and not self.suggested_type:
            return None  # No suggestion available
        return (
            sorted(self.cards_played) == sorted(self.suggested_cards)
            and self.combo_type == self.suggested_type
        )


@dataclass
class TurningPoint:
    """A moment that significantly affected the game outcome."""
    round_num: int
    description: str
    impact: str  # "positive" or "negative"
    cards_involved: List[str] = field(default_factory=list)


@dataclass
class GameStats:
    """Aggregated statistics for a completed game."""
    total_rounds: int = 0
    bombs_used: int = 0
    passes: int = 0
    plays: int = 0
    suggestions_followed: int = 0
    suggestions_total: int = 0
    key_rounds_count: int = 0
    turning_points_count: int = 0

    @property
    def pass_rate(self) -> float:
        total = self.plays + self.passes
        return self.passes / total if total > 0 else 0.0

    @property
    def bomb_rate(self) -> float:
        total = self.plays + self.passes
        return self.bombs_used / total if total > 0 else 0.0

    @property
    def suggestion_follow_rate(self) -> float:
        if self.suggestions_total == 0:
            return 0.0
        return self.suggestions_followed / self.suggestions_total


@dataclass
class GameReplay:
    """Complete replay data for one game."""
    game_id: str = ""
    start_time: str = ""
    end_time: str = ""
    result: str = ""  # "win" or "loss"
    current_level: str = "2"
    actions: List[RoundAction] = field(default_factory=list)
    turning_points: List[TurningPoint] = field(default_factory=list)
    stats: GameStats = field(default_factory=GameStats)


class ReplayRecorder:
    """Records game actions for post-game analysis."""

    def __init__(self, game_id: str = "", current_level: str = "2") -> None:
        self._replay = GameReplay(
            game_id=game_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat(),
            current_level=current_level,
        )
        self._round_counter = 0

    @property
    def replay(self) -> GameReplay:
        return self._replay

    @property
    def round_count(self) -> int:
        return self._round_counter

    def record_action(
        self,
        player: str,
        action_type: ActionType,
        cards_played: Optional[List[str]] = None,
        combo_type: str = "",
        suggested_cards: Optional[List[str]] = None,
        suggested_type: str = "",
        note: str = "",
    ) -> RoundAction:
        """Record a single game action."""
        self._round_counter += 1
        action = RoundAction(
            round_num=self._round_counter,
            player=player,
            action_type=action_type,
            cards_played=cards_played or [],
            combo_type=combo_type,
            suggested_cards=suggested_cards or [],
            suggested_type=suggested_type,
            note=note,
        )
        self._replay.actions.append(action)
        return action

    def finish_game(self, result: str) -> GameReplay:
        """Mark game as finished and compute stats."""
        self._replay.end_time = datetime.now().isoformat()
        self._replay.result = result
        self._replay.stats = self._compute_stats()
        self._replay.turning_points = self._detect_turning_points()
        self._mark_key_rounds()
        return self._replay

    def _compute_stats(self) -> GameStats:
        """Compute aggregate statistics."""
        stats = GameStats(total_rounds=len(self._replay.actions))
        for action in self._replay.actions:
            if action.player != "self":
                continue
            if action.action_type == ActionType.BOMB:
                stats.bombs_used += 1
                stats.plays += 1
            elif action.action_type == ActionType.PLAY:
                stats.plays += 1
            elif action.action_type == ActionType.PASS:
                stats.passes += 1

            followed = action.followed_suggestion
            if followed is not None:
                stats.suggestions_total += 1
                if followed:
                    stats.suggestions_followed += 1

        return stats

    def _detect_turning_points(self) -> List[TurningPoint]:
        """Detect key turning points in the game."""
        points: List[TurningPoint] = []
        for action in self._replay.actions:
            # Bomb usage is always a turning point
            if action.action_type == ActionType.BOMB:
                points.append(TurningPoint(
                    round_num=action.round_num,
                    description=f"{action.player} used a bomb ({action.combo_type})",
                    impact="positive" if action.player in ("self", "partner") else "negative",
                    cards_involved=action.cards_played,
                ))
            # Diverging from suggestion on key plays
            if (
                action.player == "self"
                and action.followed_suggestion is False
                and action.action_type != ActionType.PASS
            ):
                points.append(TurningPoint(
                    round_num=action.round_num,
                    description=(
                        f"Diverged from suggestion: played {action.combo_type} "
                        f"instead of {action.suggested_type}"
                    ),
                    impact="neutral",
                    cards_involved=action.cards_played,
                ))
        self._replay.turning_points_count = len(points)
        return points

    def _mark_key_rounds(self) -> None:
        """Mark rounds that are significant."""
        key_rounds = {tp.round_num for tp in self._replay.turning_points}
        count = 0
        for action in self._replay.actions:
            if action.round_num in key_rounds or action.action_type == ActionType.BOMB:
                action.is_key_round = True
                count += 1
        self._replay.stats.key_rounds_count = count
        self._replay.stats.turning_points_count = len(self._replay.turning_points)


class ReplayAnalyzer:
    """Analyzes a completed GameReplay."""

    def __init__(self, replay: GameReplay) -> None:
        self.replay = replay

    def get_key_rounds(self) -> List[RoundAction]:
        """Get all key rounds."""
        return [a for a in self.replay.actions if a.is_key_round]

    def get_action_diffs(self) -> List[RoundAction]:
        """Get rounds where player diverged from suggestions."""
        return [
            a for a in self.replay.actions
            if a.player == "self" and a.followed_suggestion is False
        ]

    def format_summary(self) -> str:
        """Format a human-readable game summary."""
        s = self.replay.stats
        lines = [
            f"=== Game Replay: {self.replay.game_id} ===",
            f"Result: {self.replay.result}",
            f"Level: {self.replay.current_level}",
            f"Total Rounds: {s.total_rounds}",
            "",
            "--- Statistics ---",
            f"Plays: {s.plays} | Passes: {s.passes} | Bombs: {s.bombs_used}",
            f"Pass Rate: {s.pass_rate:.0%}",
            f"Bomb Rate: {s.bomb_rate:.0%}",
            f"Suggestion Follow Rate: {s.suggestion_follow_rate:.0%}",
            "",
        ]

        if self.replay.turning_points:
            lines.append("--- Turning Points ---")
            for tp in self.replay.turning_points:
                lines.append(
                    f"  Round {tp.round_num}: {tp.description} [{tp.impact}]"
                )
            lines.append("")

        diffs = self.get_action_diffs()
        if diffs:
            lines.append("--- Action Diffs (vs Suggestion) ---")
            for d in diffs:
                lines.append(
                    f"  Round {d.round_num}: Played {' '.join(d.cards_played)} "
                    f"({d.combo_type}) vs suggested {' '.join(d.suggested_cards)} "
                    f"({d.suggested_type})"
                )

        return "\n".join(lines)


def save_replay(replay: GameReplay, path: Path) -> None:
    """Save replay to a JSON file."""
    data = {
        "game_id": replay.game_id,
        "start_time": replay.start_time,
        "end_time": replay.end_time,
        "result": replay.result,
        "current_level": replay.current_level,
        "stats": {
            "total_rounds": replay.stats.total_rounds,
            "bombs_used": replay.stats.bombs_used,
            "passes": replay.stats.passes,
            "plays": replay.stats.plays,
            "pass_rate": round(replay.stats.pass_rate, 3),
            "bomb_rate": round(replay.stats.bomb_rate, 3),
            "suggestion_follow_rate": round(replay.stats.suggestion_follow_rate, 3),
        },
        "turning_points": [
            {
                "round_num": tp.round_num,
                "description": tp.description,
                "impact": tp.impact,
            }
            for tp in replay.turning_points
        ],
        "actions_count": len(replay.actions),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_replay_summary(path: Path) -> dict:
    """Load replay summary from JSON file."""
    return json.loads(path.read_text())
