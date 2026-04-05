"""Guandan AI main entry point with --realtime mode (M7).

Extends the CLI with a ``--realtime`` flag that launches the
real-time decision AI overlay.

Usage::

    python -m guandan.main --realtime
    python -m guandan.main          # falls back to CLI mode
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from guandan.models import Rank

log = logging.getLogger(__name__)

# Rank label → enum mapping for CLI parsing
_RANK_LABELS = {r.label(): r for r in Rank}


def _parse_level(value: str) -> Rank:
    """Parse a rank label (2-A) into a Rank enum."""
    upper = value.upper()
    if upper in _RANK_LABELS:
        return _RANK_LABELS[upper]
    raise argparse.ArgumentTypeError(
        f'Invalid level rank: {value!r}. '
        f'Expected one of: {", ".join(_RANK_LABELS)}'
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Guandan AI."""
    parser = argparse.ArgumentParser(
        prog='guandan',
        description='Guandan (掼蛋) AI assistant',
    )
    parser.add_argument(
        '--realtime',
        action='store_true',
        default=False,
        help='Launch in real-time decision AI mode with overlay',
    )
    parser.add_argument(
        '--level',
        type=_parse_level,
        default=Rank.TWO,
        help='Current game level rank (default: 2)',
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=3.0,
        help='Frames per second for the real-time loop (default: 3)',
    )
    parser.add_argument(
        '--hotkeys',
        action='store_true',
        default=False,
        help='Enable global hotkeys (F1=start, F2=stop, F3=recalibrate)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Enable verbose/debug logging',
    )
    return parser


def run_realtime(
    level: Rank = Rank.TWO,
    fps: float = 3.0,
    enable_hotkeys: bool = False,
) -> None:
    """Launch the real-time AI decision loop.

    Creates a :class:`DecisionEngine`, :class:`OverlayWindow`,
    and :class:`RealtimeController`, then runs until interrupted.
    """
    from guandan.card_recognition import CardRecognizer
    from guandan.decision_engine import DecisionEngine
    from guandan.overlay_display import OverlayWindow
    from guandan.realtime_controller import (
        GameWindowRegion,
        RealtimeController,
    )

    print(f'Guandan AI — Real-time mode (level={level.label()}, fps={fps})')
    print('Press Ctrl+C to stop.')

    recognizer = CardRecognizer()
    engine = DecisionEngine(
        recognizer=recognizer,
        current_level=level,
    )
    overlay = OverlayWindow()
    controller = RealtimeController(
        engine=engine,
        overlay=overlay,
        fps=fps,
        enable_hotkeys=enable_hotkeys,
    )

    region = GameWindowRegion()
    controller.start(region)

    try:
        import time
        while controller.state.name == 'RUNNING':
            overlay.update()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        controller.stop()
        overlay.destroy()

    stats = controller.stats
    print(
        f'Done. Frames={stats.frames_processed}, '
        f'Decisions={stats.decisions_made}, '
        f'Avg latency={stats.avg_latency_ms:.0f}ms'
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    )

    if args.realtime:
        run_realtime(
            level=args.level,
            fps=args.fps,
            enable_hotkeys=args.hotkeys,
        )
    else:
        # Fall back to CLI mode
        from guandan.cli import main as cli_main
        cli_main()


if __name__ == '__main__':
    main()
