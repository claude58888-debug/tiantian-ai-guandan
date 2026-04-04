"""Tests for Atom 2.4 - AI agent integration layer."""
import pytest
from guandan.models import Card, Rank, Suit
from guandan.game_state_parser import VisualGameState, TurnPhase
from guandan.agent import (
    AgentState, AgentAction, AgentConfig,
    GuandanAgent, create_agent,
)


def make_card(rank, suit=Suit.HEARTS):
    return Card(rank=rank, suit=suit)


class TestAgentState:
    def test_enum_values(self):
        assert AgentState.IDLE is not None
        assert AgentState.OBSERVING is not None
        assert AgentState.THINKING is not None
        assert AgentState.ACTING is not None
        assert AgentState.PAUSED is not None
        assert AgentState.ERROR is not None


class TestAgentAction:
    def test_default_action(self):
        action = AgentAction()
        assert action.cards_to_play is None
        assert action.is_pass is False
        assert action.is_play is False

    def test_play_action(self):
        cards = [make_card(Rank.THREE)]
        action = AgentAction(cards_to_play=cards)
        assert action.is_play is True
        assert action.is_pass is False

    def test_pass_action(self):
        action = AgentAction(is_pass=True)
        assert action.is_pass is True
        assert action.is_play is False

    def test_repr_pass(self):
        action = AgentAction(is_pass=True, confidence=0.9)
        assert 'PASS' in repr(action)

    def test_repr_play(self):
        action = AgentAction(cards_to_play=[make_card(Rank.ACE)])
        assert 'A' in repr(action)

    def test_repr_none(self):
        action = AgentAction()
        assert 'NONE' in repr(action)


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.strategy_name == 'smart'
        assert config.aggression == 0.5
        assert config.auto_play is False
        assert config.current_level == Rank.TWO

    def test_custom(self):
        config = AgentConfig(strategy_name='greedy', aggression=0.8)
        assert config.strategy_name == 'greedy'
        assert config.aggression == 0.8


class TestGuandanAgent:
    def test_create_default(self):
        agent = GuandanAgent()
        assert agent.state == AgentState.IDLE
        assert agent.last_game_state is None
        assert agent.last_action is None

    def test_create_with_config(self):
        config = AgentConfig(strategy_name='greedy')
        agent = GuandanAgent(config)
        assert agent.config.strategy_name == 'greedy'

    def test_observe_no_capturer(self):
        agent = GuandanAgent()
        result = agent.observe()
        assert result is None

    def test_decide_low_confidence(self):
        agent = GuandanAgent()
        state = VisualGameState(confidence=0.1, turn_phase=TurnPhase.MY_TURN)
        action = agent.decide(state)
        assert 'confidence' in action.reasoning.lower()

    def test_decide_not_my_turn(self):
        agent = GuandanAgent()
        state = VisualGameState(confidence=0.9, turn_phase=TurnPhase.WAITING)
        action = agent.decide(state)
        assert 'turn' in action.reasoning.lower()

    def test_decide_my_turn_with_hand(self):
        agent = GuandanAgent()
        hand = [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.FIVE, Suit.CLUBS)]
        state = VisualGameState(
            my_hand=hand,
            confidence=0.9,
            turn_phase=TurnPhase.MY_TURN,
        )
        action = agent.decide(state)
        assert action.is_play or action.is_pass

    def test_on_action_callback(self):
        agent = GuandanAgent()
        results = []
        agent.on_action(lambda a: results.append(a))
        action = AgentAction(is_pass=True)
        agent.act(action)
        assert len(results) == 1

    def test_stop(self):
        agent = GuandanAgent()
        agent._running = True
        agent.stop()
        assert agent._running is False

    def test_pause_resume(self):
        agent = GuandanAgent()
        agent.pause()
        assert agent.state == AgentState.PAUSED
        agent.resume()
        assert agent.state == AgentState.IDLE

    def test_step_no_capturer(self):
        agent = GuandanAgent()
        action = agent.step()
        assert action is None


class TestCreateAgent:
    def test_create_default(self):
        agent = create_agent()
        assert isinstance(agent, GuandanAgent)
        assert agent.config.strategy_name == 'smart'

    def test_create_custom(self):
        agent = create_agent(strategy='greedy', aggression=0.8, level=Rank.FIVE)
        assert agent.config.strategy_name == 'greedy'
        assert agent.config.aggression == 0.8
        assert agent.config.current_level == Rank.FIVE
