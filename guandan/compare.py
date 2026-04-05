"""Guandan hand comparison logic (Atom 1.3).

Rules for comparing plays:
1. Bombs beat all non-bomb combos.
2. Higher-level bombs beat lower-level bombs:
   BOMB_4 < BOMB_5 < BOMB_6 < BOMB_7 < BOMB_8 < JOKER_BOMB
3. Within same combo type and size, higher rank_key wins.
4. Non-bomb combos can only be beaten by same type+size with higher rank.
"""
from __future__ import annotations

from enum import IntEnum, unique
from typing import Optional, Tuple

from guandan.combos import Combo, ComboType, classify_combo
from guandan.models import Card, Rank


# ── Combo integrity validation ────────────────────────────────────────

# Expected card-count ranges per combo type.
# BOMB_4 allows 2 cards because a Rocket (two-joker pair) is classified
# as BOMB_4 with rank_key 150.
_SIZE_RANGES: dict[ComboType, Tuple[int, int]] = {
    ComboType.PASS: (0, 0),
    ComboType.SINGLE: (1, 1),
    ComboType.PAIR: (2, 2),
    ComboType.TRIPLE: (3, 3),
    ComboType.FULL_HOUSE: (5, 5),
    ComboType.STRAIGHT: (5, 13),
    ComboType.CONSECUTIVE_PAIRS: (6, 26),
    ComboType.PLATE: (6, 39),
    ComboType.BOMB_4: (2, 4),       # 2 for Rocket, 4 for normal
    ComboType.BOMB_5: (5, 5),
    ComboType.BOMB_6: (6, 6),
    ComboType.BOMB_7: (7, 7),
    ComboType.BOMB_8: (8, 8),
    ComboType.JOKER_BOMB: (4, 4),
}


def validate_combo(combo: Combo) -> bool:
    """Check that a combo's card count is consistent with its type.

    Returns True if valid, False otherwise.
    """
    size_range = _SIZE_RANGES.get(combo.combo_type)
    if size_range is None:
        return False
    lo, hi = size_range
    return lo <= combo.size <= hi


@unique
class CompareResult(IntEnum):
    """Result of comparing two combos."""
    LESS = -1
    EQUAL = 0
    GREATER = 1
    INCOMPARABLE = 2


def can_beat(play: Combo, prev: Combo) -> bool:
    """Check if 'play' can beat 'prev'.

    Returns True if play legally beats prev.
    Raises ValueError if either combo has inconsistent size for its type.
    """
    if not validate_combo(play):
        raise ValueError(
            f'Invalid play combo: {play.combo_type.name} with {play.size} cards'
        )
    if not validate_combo(prev):
        raise ValueError(
            f'Invalid prev combo: {prev.combo_type.name} with {prev.size} cards'
        )

    if prev.combo_type == ComboType.PASS:
        return True  # anything beats a pass

    # Bombs beat non-bombs
    if play.is_bomb and not prev.is_bomb:
        return True

    # Non-bomb cannot beat bomb
    if not play.is_bomb and prev.is_bomb:
        return False

    # Both are bombs – compare by bomb tier first, then rank
    if play.is_bomb and prev.is_bomb:
        if play.combo_type != prev.combo_type:
            return play.combo_type > prev.combo_type
        return play.rank_key > prev.rank_key

    # Both non-bombs: must be same type and same size
    if play.combo_type != prev.combo_type:
        return False
    if play.size != prev.size:
        return False
    return play.rank_key > prev.rank_key


def compare_combos(a: Combo, b: Combo) -> int:
    """Compare two combos.

    Returns 1 if *a* beats *b*, -1 if *b* beats *a*, 0 if equal rank.
    For structured comparison that distinguishes equal from incomparable,
    use :func:`compare_combos_full`.
    """
    if can_beat(a, b) and can_beat(b, a):
        return 0  # shouldn't happen in practice
    if can_beat(a, b):
        return 1
    if can_beat(b, a):
        return -1
    return 0  # incomparable (different non-bomb types)


def compare_combos_full(a: Combo, b: Combo) -> CompareResult:
    """Structured comparison that distinguishes equal from incomparable.

    Returns CompareResult.GREATER if *a* beats *b*,
    CompareResult.LESS if *b* beats *a*,
    CompareResult.EQUAL if they have the same type, size and rank,
    CompareResult.INCOMPARABLE if they cannot be compared (different
    non-bomb types or different sizes).
    """
    a_beats_b = can_beat(a, b)
    b_beats_a = can_beat(b, a)

    if a_beats_b and not b_beats_a:
        return CompareResult.GREATER
    if b_beats_a and not a_beats_b:
        return CompareResult.LESS
    if a_beats_b and b_beats_a:
        return CompareResult.EQUAL  # shouldn't happen in practice

    # Neither beats the other
    if (a.combo_type == b.combo_type and a.size == b.size
            and a.rank_key == b.rank_key):
        return CompareResult.EQUAL
    return CompareResult.INCOMPARABLE
