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


def classify_with_wilds(
    cards: Sequence[Card], level: Rank
) -> Optional[Combo]:
    """Classify a combo, considering wild card substitutions.
    
    First tries classifying as-is (natural). If that fails and there
    are wild cards, tries all possible substitutions to find a valid combo.
    Returns the best (highest-ranked) valid combo found, or None.
    """
    # Try natural classification first
    natural = classify_combo(cards, level)
    if natural is not None:
        return natural
    
    normal, wilds = split_wilds(cards, level)
    if not wilds:
        return None  # No wilds, can't form anything
    
    n_wilds = len(wilds)
    n = len(cards)
    
    # Don't allow wild substitution for joker bomb
    if n == 4 and all(c.is_joker for c in normal):
        return None
    
    # Try substituting wilds with every possible rank/suit
    best: Optional[Combo] = None
    ranks = _possible_ranks()
    suits = list(Suit)
    
    # Generate all possible substitutions for wild cards
    for subs in product(product(ranks, suits), repeat=n_wilds):
        trial_cards = list(normal)
        for i, (r, s) in enumerate(subs):
            trial_cards.append(_make_substitute(r, s, wilds[i].deck_id))
        
        combo = classify_combo(trial_cards, level)
        if combo is not None:
            # Prefer: higher combo type, then higher rank_key
            if best is None:
                best = combo
            elif (combo.combo_type.value, combo.rank_key) > (
                best.combo_type.value, best.rank_key
            ):
                best = combo
    
    if best is None:
        return None
    
    # Return combo with original cards (including wilds), not substitutes
    return Combo(
        combo_type=best.combo_type,
        cards=tuple(cards),
        rank_key=best.rank_key,
    )


def find_wild_combos(
    cards: Sequence[Card], level: Rank
) -> List[Combo]:
    """Find all possible combos from a set of cards using wild substitution.
    
    This is an exhaustive search - for a hand with wilds, finds every
    valid combo that can be formed by substituting wild cards.
    Returns deduplicated list of combos.
    """
    normal, wilds = split_wilds(cards, level)
    if not wilds:
        result = classify_combo(cards, level)
        return [result] if result else []
    
    n_wilds = len(wilds)
    seen = set()
    results = []
    
    ranks = _possible_ranks()
    suits = list(Suit)
    
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
                    cards=tuple(cards),
                    rank_key=combo.rank_key,
                ))
    
    return results


def can_beat_with_wilds(
    cards: Sequence[Card],
    last_play: Combo,
    level: Rank,
) -> Optional[Combo]:
    """Check if cards (possibly with wilds) can beat last_play.
    
    Returns the best combo that beats last_play, or None.
    """
    from guandan.compare import can_beat
    
    combo = classify_with_wilds(cards, level)
    if combo is not None and can_beat(combo, last_play):
        return combo
    return None
