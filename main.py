#!/usr/bin/env python3
"""Main entry point for the Guandan AI Agent (Atom 3.3).

Supports three run modes:
  - gui   : Launch Tkinter dashboard (default)
  - cli   : Interactive command-line play
  - agent : Headless agent loop

Usage:
  python main.py              # GUI mode
  python main.py --mode cli   # CLI mode
  python main.py --mode agent # Agent mode
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("guandan")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="guandan",
        description="Guandan AI Agent - card game assistant",
    )
    parser.add_argument(
        "--mode",
        choices=["gui", "cli", "agent"],
        default="gui",
        help="Run mode (default: gui)",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "greedy", "smart"],
        default="smart",
        help="AI strategy (default: smart)",
    )
    parser.add_argument(
        "--aggression",
        type=float,
        default=0.5,
        help="Aggression level 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--level",
        default="2",
        help="Current level rank (default: 2)",
    )
    parser.add_argument(
        "--auto-play",
        action="store_true",
        help="Enable auto-play in agent mode",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def run_gui() -> None:
    """Launch the Tkinter GUI dashboard."""
    from guandan.ui import MainWindowViewModel, GuandanUI

    vm = MainWindowViewModel(app_status="observing")
    app = GuandanUI(vm)
    logger.info("GUI launched")
    app.mainloop()


def run_cli(strategy: str, level: str) -> None:
    """Launch the interactive CLI."""
    from guandan.cli import main as cli_main

    cli_main()


def run_agent(
    strategy: str,
    aggression: float,
    level: str,
    auto_play: bool,
) -> None:
    """Run the headless agent loop."""
    from guandan.models import Rank
    from guandan.agent import create_agent

    rank_map = {
        "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR,
        "5": Rank.FIVE, "6": Rank.SIX, "7": Rank.SEVEN,
        "8": Rank.EIGHT, "9": Rank.NINE, "10": Rank.TEN,
        "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING,
        "A": Rank.ACE,
    }
    rank = rank_map.get(level.upper(), Rank.TWO)
    agent = create_agent(
        strategy=strategy,
        aggression=aggression,
        level=rank,
        auto_play=auto_play,
    )
    logger.info(f"Agent started: strategy={strategy}, level={level}")
    agent.run()


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "gui":
        run_gui()
    elif args.mode == "cli":
        run_cli(args.strategy, args.level)
    elif args.mode == "agent":
        run_agent(
            strategy=args.strategy,
            aggression=args.aggression,
            level=args.level,
            auto_play=args.auto_play,
        )
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
