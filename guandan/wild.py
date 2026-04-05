"""Wild card (逢人配) logic for Guandan (Atom 1.4).

In Guandan, the Hearts card of the current level rank is a wild card
(逢人配/万能牌). It can substitute for any card to form valid combos.

Rules:
- Only the Heart suit level-rank card is wild (e.g., if level=5, Heart 5 is wild)
- Wild cards can substitute any rank/suit to complete combos
- A combo formed using wild cards is valid but ranks lower than
  the same combo formed with natural cards
- Wild cards cannot be used in: Joker Bomb (4 jokers)
- Wild cards CAN be played as themselves (their natural rank)
- Two wild cards together form a pair of their natural rank

V0.3 fix (C-2): When multiple wild cards are present, the substitution
engine now searches all combinations and uses a proper priority ranking
that considers bomb hierarchy, card count, and strategic value rather
than a simple greedy (combo_type.value, rank_key) comparison.
"""
from __future__ import annotations

from itertools import product
from typing import List, Optional, Sequence, Tuple

from guandan.models import Card, Rank, Suit
from guandan.combos import Combo, ComboType, classify_combo


def is_wild(card: Card, level: Rank) -> bool:
    """Check if a card is a wild card at the given level.
    
    Wild card = Heart suit + current level rank.
    Jokers are never wild.
    """
    if card.is_joker:
        return False
    return card.rank == level and card.suit == Suit.HEARTS


def count_wilds(cards: Sequence[Card], level: Rank) -> int:
    """Count wild cards in a hand."""
    return sum(1 for c in cards if is_wild(c, level))


def split_wilds(
    cards: Sequence[Card], level: Rank
) -> Tuple[List[Card], List[Card]]:
    """Split cards into (normal_cards, wild_cards)."""
    normal = [c for c in cards if not is_wild(c, level)]
    wilds = [c for c in cards if is_wild(c, level)]
    return normal, wilds


def _make_substitute(rank: Rank, suit: Suit, deck_id: int = 0) -> Card:
    """Create a substitute card for a wild card."""
    return Card(rank=rank, suit=suit, deck_id=deck_id)


def _possible_ranks() -> List[Rank]:
    """All possible ranks a wild card can substitute for."""
    return list(Rank)


def _combo_priority(combo: Combo) -> Tuple[int, int, int]:
    """Compute a priority tuple for ranking wild-card substitution results.

    Priority order (higher is better):
    1. Bomb tier: bombs always beat non-bombs; higher bomb type wins.
    2. Card count: among non-bomb combos, using more cards in a single
       combo is strategically preferable (clears the hand faster).
    3. Rank key: within the same type, higher rank is better.
    """
    bomb_tier = combo.combo_type.value if combo.is_bomb else 0
    return (bomb_tier, combo.size, combo.rank_key)


def _collect_all_substitutions(
    normal: List[Card],
    wilds: List[Card],
    level: Rank,
) -> List[Combo]:
    """Try every rank/suit substitution for each wild card.

    Returns a deduplicated list of all valid Combo objects found,
    each carrying the *original* cards (including wilds).
    """
    n_wilds = len(wilds)
    original_cards = tuple(normal + wilds)
    ranks = _possible_ranks()
    suits = list(Suit)
    seen: set[Tuple[ComboType, int]] = set()
    results: List[Combo] = []

    for subs in product(product(ranks, suits), repeat=n_wilds):
        trial_cards = list(normal)
        for i, (r, s) in enumerate(subs):
            trial_cards.append(_make_substitute(r, s, wilds[i].deck_id))

        combo = classify_combo(trial_cards, level)
        if combo is not None:
            key = (combo.combo_type, combo.rank_key)
            if key not in seen:
                seen.add(key)
                results.append(Combo(
                    combo_type=combo.combo_type,
                    cards=original_cards,
                    rank_key=combo.rank_key,
                ))

    return results


def classify_with_wilds(
    cards: Sequence[Card], level: Rank
) -> Optional[Combo]:
    """Classify a combo, considering wild card substitutions.

    First tries classifying as-is (natural). If that fails and there
    are wild cards, tries all possible substitutions to find the best
    valid combo using a proper priority ranking (bombs > larger combos
    > higher rank) rather than a simple greedy comparison.

    Returns the best valid combo found, or None.
    """
    # Try natural classification first
    natural = classify_combo(cards, level)
    if natural is not None:
        return natural

    normal, wilds = split_wilds(cards, level)
    if not wilds:
        return None  # No wilds, can't form anything

    n = len(cards)

    # Don't allow wild substitution for joker bomb
    if n == 4 and all(c.is_joker for c in normal):
        return None

    candidates = _collect_all_substitutions(normal, wilds, level)
    if not candidates:
        return None

    # Pick the best candidate using the full priority ranking
    return max(candidates, key=_combo_priority)


def find_wild_combos(
    cards: Sequence[Card], level: Rank
) -> List[Combo]:
    """Find all possible combos from a set of cards using wild substitution.

    This is an exhaustive search - for a hand with wilds, finds every
    valid combo that can be formed by substituting wild cards.
    Returns deduplicated list of combos sorted by priority (best first).
    """
    normal, wilds = split_wilds(cards, level)
    if not wilds:
        result = classify_combo(cards, level)
        return [result] if result else []

    results = _collect_all_substitutions(normal, wilds, level)
    results.sort(key=_combo_priority, reverse=True)
    return results


def can_beat_with_wilds(
    cards: Sequence[Card],
    last_play: Combo,
    level: Rank,
) -> Optional[Combo]:
    """Check if cards (possibly with wilds) can beat last_play.

    Searches *all* valid substitution combos (not just the single
    "best" one) so that a beating combo is found whenever one exists.
    Returns the highest-priority combo that beats last_play, or None.
    """
    from guandan.compare import can_beat

    all_combos = find_wild_combos(cards, level)
    beaters = [c for c in all_combos if can_beat(c, last_play)]
    if not beaters:
        return None
    return max(beaters, key=_combo_priority)
