"""Guandan CLI dealer + interactive play (Atom 1.5).

Command-line interface for playing Guandan with AI opponents.
Usage: python -m guandan.cli
"""
from __future__ import annotations

import sys
from typing import List, Optional

from guandan.models import Card, Rank, Suit, JokerType
from guandan.combos import classify_combo, ComboType
from guandan.game import GameState, Phase


def parse_card(s: str) -> Optional[Card]:
    """Parse card string like '3H', '10S', 'BJ', 'RJ'."""
    s = s.strip().upper()
    if s == 'BJ':
        return Card(joker=JokerType.BLACK)
    if s == 'RJ':
        return Card(joker=JokerType.RED)

    suit_map = {'H': Suit.HEARTS, 'D': Suit.DIAMONDS,
                'C': Suit.CLUBS, 'S': Suit.SPADES}
    rank_map = {'2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR,
                '5': Rank.FIVE, '6': Rank.SIX, '7': Rank.SEVEN,
                '8': Rank.EIGHT, '9': Rank.NINE, '10': Rank.TEN,
                'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING, 'A': Rank.ACE}

    if len(s) < 2:
        return None
    suit_char = s[-1]
    rank_str = s[:-1]
    if suit_char not in suit_map or rank_str not in rank_map:
        return None
    return Card(rank=rank_map[rank_str], suit=suit_map[suit_char])


def find_matching_cards(hand_cards: List[Card], parsed: List[Card]) -> Optional[List[Card]]:
    """Find actual cards in hand matching the parsed cards (ignoring deck_id)."""
    remaining = list(hand_cards)
    matched = []
    for p in parsed:
        found = False
        for i, c in enumerate(remaining):
            if p.is_joker and c.is_joker and p.joker == c.joker:
                matched.append(remaining.pop(i))
                found = True
                break
            elif not p.is_joker and not c.is_joker and p.rank == c.rank and p.suit == c.suit:
                matched.append(remaining.pop(i))
                found = True
                break
        if not found:
            return None
    return matched


def display_hand(cards: List[Card]) -> str:
    return ' '.join(c.display() for c in cards)


def random_ai_play(game: GameState, player_idx: int) -> bool:
    """Simple AI: play first valid single card, or pass."""
    player = game.players[player_idx]
    if game.last_play is None or game.last_play.combo_type == ComboType.PASS:
        # Must lead: play lowest single
        if player.hand.cards:
            card = player.hand.cards[0]
            return game.play_cards(player_idx, [card])
    else:
        # Try each card as single to beat
        for card in player.hand.cards:
            combo = classify_combo([card], game.current_level)
            if combo and game.last_play:
                from guandan.compare import can_beat
                if can_beat(combo, game.last_play):
                    return game.play_cards(player_idx, [card])
        # Pass
        return game.play_cards(player_idx, [])
    return False


def main():
    print('=== Guandan CLI ===')  
    print('Teams: You(P0)+P2 vs P1+P3')
    print('Commands: type cards like "3H 3D" or "pass"')
    print()

    game = GameState()
    game.deal(seed=42)

    print(f'Current level: {game.current_level.label()}')
    print(f'Your hand: {display_hand(game.players[0].hand.cards)}')
    print(f'({len(game.players[0].hand)} cards)')
    print()

    while game.phase == Phase.PLAYING:
        cp = game.current_player
        player = game.players[cp]

        if cp == 0:  # Human player
            if game.last_play and game.last_play.combo_type != ComboType.PASS:
                print(f'  Last play by P{game.last_player}: {game.last_play}')
            else:
                print('  You lead this trick.')

            print(f'  Your hand: {display_hand(player.hand.cards)}')
            inp = input('  Your play (or "pass"): ').strip()

            if inp.lower() == 'pass':
                ok = game.play_cards(0, [])
                if ok:
                    print('  You passed.')
                else:
                    print('  Cannot pass - you must lead!')
                continue

            tokens = inp.split()
            parsed = [parse_card(t) for t in tokens]
            if any(p is None for p in parsed):
                print('  Invalid card format. Use like: 3H 10S BJ')
                continue

            matched = find_matching_cards(player.hand.cards, parsed)
            if matched is None:
                print('  Cards not in your hand!')
                continue

            ok = game.play_cards(0, matched)
            if ok:
                combo = classify_combo(matched, game.current_level)
                print(f'  Played: {combo}')
            else:
                print('  Invalid play!')
        else:
            # AI turn
            ok = random_ai_play(game, cp)
            if ok:
                last = game.history[-1]
                if last[1] is None:
                    print(f'  P{cp} passed.')
                else:
                    print(f'  P{cp} played: {last[1]}')

        if game.phase == Phase.ROUND_END:
            print()
            print('=== Round Over ===')
            print(f'Finish order: {game.finish_order}')
            t0, t1 = game.get_round_result()
            print(f'Team 0 advances {t0}, Team 1 advances {t1}')
            break

    print('Game ended.')


if __name__ == '__main__':
    main()
