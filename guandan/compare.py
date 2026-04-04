"""Guandan hand comparison logic (Atom 1.3).

Rules for comparing plays:
1. Bombs beat all non-bomb combos.
2. Higher-level bombs beat lower-level bombs:
   BOMB_4 < BOMB_5 < BOMB_6 < BOMB_7 < BOMB_8 < JOKER_BOMB
3. Within same combo type and size, higher rank_key wins.
4. Non-bomb combos can only be beaten by same type+size with higher rank.
"""
from __future__ import annotations

from typing import Optional

from guandan.combos import Combo, ComboType, classify_combo
from guandan.models import Card, Rank


def can_beat(play: Combo, prev: Combo) -> bool:
    """Check if 'play' can beat 'prev'.

    Returns True if play legally beats prev.
    """
    if prev.combo_type == ComboType.PASS:
        return True  # anything beats a pass

    # Bombs beat non-bombs
    if play.is_bomb and not prev.is_bomb:
        return True

    # Non-bomb cannot beat bomb
    if not play.is_bomb and prev.is_bomb:
        return False

    # Both are bombs
    if play.is_bomb and prev.is_bomb:
        # Higher bomb type wins
        if play.combo_type > prev.combo_type:
            return True
        if play.combo_type < prev.combo_type:
            return False
        # Same bomb type: higher rank wins
        return play.rank_key > prev.rank_key

    # Both non-bombs: must be same type and same size
    if play.combo_type != prev.combo_type:
        return False
    if play.size != prev.size:
        return False
    return play.rank_key > prev.rank_key


def compare_combos(a: Combo, b: Combo) -> int:
    """Compare two combos. Returns positive if a > b, negative if a < b, 0 if equal."""
    if can_beat(a, b) and can_beat(b, a):
        return 0  # shouldn't happen in practice
    if can_beat(a, b):
        return 1
    if can_beat(b, a):
        return -1
    return 0  # incomparable (different non-bomb types)
