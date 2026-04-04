"""Guandan AI Agent - integration layer (Atom 2.4).

Combines image recognition (screen capture + card detection) with
the rule engine and AI strategy to create an autonomous agent that
can observe and play the Guandan game.

Pipeline: Screenshot -> Card Detection -> Game State -> Strategy -> Action
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Callable

from guandan.models import Card, Rank
from guandan.combos import Combo
from guandan.strategy import Strategy, get_strategy
from guandan.game_state_parser import (
    GameStateParser, VisualGameState, TurnPhase, create_parser,
)

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle state."""
    IDLE = auto()
    OBSERVING = auto()
    THINKING = auto()
    ACTING = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class AgentAction:
    """An action decided by the agent."""
    cards_to_play: Optional[List[Card]] = None
    is_pass: bool = False
    confidence: float = 0.0
    reasoning: str = ''

    @property
    def is_play(self) -> bool:
        return self.cards_to_play is not None and len(self.cards_to_play) > 0

    def __repr__(self) -> str:
        if self.is_pass:
            return f'AgentAction(PASS, conf={self.confidence:.2f})'
        if self.cards_to_play:
            cards_str = ', '.join(c.display() for c in self.cards_to_play)
            return f'AgentAction([{cards_str}], conf={self.confidence:.2f})'
        return 'AgentAction(NONE)'


@dataclass
class AgentConfig:
    """Configuration for the AI agent."""
    strategy_name: str = 'smart'
    aggression: float = 0.5
    poll_interval: float = 0.5  # seconds between observations
    min_confidence: float = 0.7  # minimum detection confidence to act
    auto_play: bool = False  # whether to auto-execute actions
    current_level: Rank = Rank.TWO
    window_title: Optional[str] = None


class GuandanAgent:
    """Main AI agent that observes and plays Guandan.
    
    The agent follows a simple loop:
    1. Observe: capture screenshot -> detect game state
    2. Decide: if it's our turn, use strategy to pick action
    3. Act: execute the action (or suggest to user)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._state = AgentState.IDLE
        self._strategy: Strategy = get_strategy(
            self.config.strategy_name,
            **({
                'aggression': self.config.aggression
            } if self.config.strategy_name == 'smart' else {})
        )
        self._parser: GameStateParser = create_parser(
            current_level=self.config.current_level,
            window_title=self.config.window_title,
        )
        self._last_game_state: Optional[VisualGameState] = None
        self._last_action: Optional[AgentAction] = None
        self._action_callbacks: List[Callable] = []
        self._running = False

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def last_game_state(self) -> Optional[VisualGameState]:
        return self._last_game_state

    @property
    def last_action(self) -> Optional[AgentAction]:
        return self._last_action

    def on_action(self, callback: Callable) -> None:
        """Register a callback for when agent decides an action."""
        self._action_callbacks.append(callback)

    def observe(self) -> Optional[VisualGameState]:
        """Capture and parse current game state."""
        self._state = AgentState.OBSERVING
        try:
            game_state = self._parser.parse_live()
            if game_state is not None:
                self._last_game_state = game_state
            return game_state
        except Exception as e:
            logger.error(f'Observation failed: {e}')
            self._state = AgentState.ERROR
            return None

    def decide(self, game_state: VisualGameState) -> AgentAction:
        """Decide what to play based on game state."""
        self._state = AgentState.THINKING

        # Check confidence threshold
        if game_state.confidence < self.config.min_confidence:
            return AgentAction(
                confidence=game_state.confidence,
                reasoning='Low detection confidence, skipping',
            )

        # Check if it's our turn
        if not game_state.is_my_turn:
            return AgentAction(
                confidence=game_state.confidence,
                reasoning='Not our turn',
            )

        # Use strategy to decide
        hand = game_state.my_hand
        last_play = game_state.last_played
        level = game_state.current_level

        cards_to_play = self._strategy.play(hand, last_play, level)

        if cards_to_play is None:
            action = AgentAction(
                is_pass=True,
                confidence=game_state.confidence,
                reasoning='Strategy chose to pass',
            )
        else:
            action = AgentAction(
                cards_to_play=cards_to_play,
                confidence=game_state.confidence,
                reasoning=f'Strategy chose {len(cards_to_play)} card(s)',
            )

        self._last_action = action
        return action

    def act(self, action: AgentAction) -> bool:
        """Execute an action.
        
        In auto_play mode, this would click the game UI.
        Otherwise, it just notifies callbacks.
        Returns True if action was executed.
        """
        self._state = AgentState.ACTING

        for callback in self._action_callbacks:
            try:
                callback(action)
            except Exception as e:
                logger.error(f'Action callback error: {e}')

        # Auto-play is a placeholder - real implementation would
        # use pyautogui or similar to click cards in the UI
        if self.config.auto_play and action.is_play:
            logger.info(f'Auto-play: {action}')
            # TODO: implement actual UI clicking
            return True

        self._state = AgentState.IDLE
        return action.is_play or action.is_pass

    def step(self) -> Optional[AgentAction]:
        """Run one observe-decide-act cycle."""
        game_state = self.observe()
        if game_state is None:
            return None

        action = self.decide(game_state)
        if action.is_play or action.is_pass:
            self.act(action)

        self._state = AgentState.IDLE
        return action

    def run(self, max_steps: int = 0) -> None:
        """Run the agent loop.
        
        Args:
            max_steps: Max iterations, 0 for infinite.
        """
        self._running = True
        self._state = AgentState.OBSERVING
        step_count = 0

        logger.info('Agent started')
        try:
            while self._running:
                if max_steps > 0 and step_count >= max_steps:
                    break

                self.step()
                step_count += 1
                time.sleep(self.config.poll_interval)
        except KeyboardInterrupt:
            logger.info('Agent interrupted by user')
        finally:
            self._running = False
            self._state = AgentState.IDLE
            logger.info(f'Agent stopped after {step_count} steps')

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False

    def pause(self) -> None:
        """Pause the agent."""
        self._state = AgentState.PAUSED

    def resume(self) -> None:
        """Resume the agent."""
        self._state = AgentState.IDLE


def create_agent(
    strategy: str = 'smart',
    aggression: float = 0.5,
    level: Rank = Rank.TWO,
    auto_play: bool = False,
) -> GuandanAgent:
    """Create a configured Guandan AI agent."""
    config = AgentConfig(
        strategy_name=strategy,
        aggression=aggression,
        current_level=level,
        auto_play=auto_play,
    )
    return GuandanAgent(config)
