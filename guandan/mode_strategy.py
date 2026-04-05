"""Game mode strategies for Tiantian Ai Guandan (天天爱掼蛋).

Each game mode has unique rules and requires different AI strategies.
This module defines mode-specific configurations and strategy adjustments.

Supported modes:
- CLASSIC: 经典模式 - Standard Guandan rules
- NO_SHUFFLE: 不洗牌翻倍 - No-shuffle with score multiplier
- TEAM_2V2: 2v2组队 - Team-based 2v2
- EIGHT_HEARTS: 八红桃 - Eight of Hearts special rules
- RANKED: 排位竞技 - Ranked competitive
- LEAGUE: 苏掼联赛 - Su-Guan League tournament

References:
- Official rules: https://www.ttigd.com/
- Bilibili tutorial: https://www.bilibili.com/video/BV1dT4y147Pd/
- National competition rules (2017)
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List

log = logging.getLogger(__name__)


class GameMode(enum.Enum):
    """All supported game modes in Tiantian Ai Guandan."""
    CLASSIC = "classic"
    NO_SHUFFLE = "no_shuffle"
    TEAM_2V2 = "team_2v2"
    EIGHT_HEARTS = "eight_hearts"
    RANKED = "ranked"
    LEAGUE = "league"


# Chinese display names for UI
MODE_DISPLAY_NAMES: Dict[GameMode, str] = {
    GameMode.CLASSIC: "经典模式",
    GameMode.NO_SHUFFLE: "不洗牌翻倍",
    GameMode.TEAM_2V2: "2v2组队",
    GameMode.EIGHT_HEARTS: "八红桃",
    GameMode.RANKED: "排位竞技",
    GameMode.LEAGUE: "苏掼联赛",
}


@dataclass(frozen=True)
class ModeConfig:
    """Configuration parameters specific to a game mode."""
    mode: GameMode
    display_name: str
    # Strategy tuning knobs
    aggression: float = 0.5           # 0.0=conservative, 1.0=aggressive
    bomb_threshold: float = 0.6       # Confidence to use a bomb
    wild_priority: float = 0.5        # Priority for wild card combos
    team_cooperation: float = 0.5     # How much to assist teammate
    score_multiplier_aware: bool = False  # Adjust for score multipliers
    # Mode-specific flags
    expect_many_bombs: bool = False   # No-shuffle yields more bombs
    tribute_enabled: bool = True      # Whether tribute (进贡) applies
    level_starts_at: int = 2          # Starting level (rank)
    max_rounds: Optional[int] = None  # Round limit for tournaments
    description: str = ""
    tips: List[str] = field(default_factory=list)


# ── Mode-specific strategy configurations ──

CLASSIC_CONFIG = ModeConfig(
    mode=GameMode.CLASSIC,
    display_name="经典模式",
    aggression=0.5,
    bomb_threshold=0.6,
    wild_priority=0.5,
    team_cooperation=0.5,
    description=(
        "标准掼蛋规则。从2打到A，两副牌108张。"
        "红桃级牌为逢人配(万能牌)。"
        "双下升3级，单下升2级，队友末游升1级。"
    ),
    tips=[
        "优先清小牌，保留炸弹控场",
        "注意队友配合，不要盲目拆牌",
        "逢人配优先用于组成炸弹或同花顺",
        "记牌：重点关注大小王和级牌数量",
        "打A时必须己方一人头游且队友非末游",
    ],
)

NO_SHUFFLE_CONFIG = ModeConfig(
    mode=GameMode.NO_SHUFFLE,
    display_name="不洗牌翻倍",
    aggression=0.7,
    bomb_threshold=0.4,
    wild_priority=0.6,
    team_cooperation=0.5,
    score_multiplier_aware=True,
    expect_many_bombs=True,
    description=(
        "不洗牌模式：发牌优先同花顺和炸弹，散牌少。"
        "炸弹多、倍数高，赢时翻倍多，输也容易大亏。"
        "需要更激进的策略，敢于炸弹对轰。"
    ),
    tips=[
        "手牌炸弹多时，主动进攻抢头游",
        "小炸早用争节奏，大炸留关键回合",
        "注意倍数累积，避免被翻倍反杀",
        "同花顺概率高，组牌时优先考虑",
        "对手炸弹也多，留足反制手段",
    ],
)

TEAM_2V2_CONFIG = ModeConfig(
    mode=GameMode.TEAM_2V2,
    display_name="2v2组队",
    aggression=0.5,
    bomb_threshold=0.55,
    wild_priority=0.5,
    team_cooperation=0.8,
    description=(
        "2v2组队模式：与好友组队对抗。"
        "队友配合是核心，需要默契传牌和接风。"
        "通讯受限，靠出牌暗示传递信息。"
    ),
    tips=[
        "队友出小牌时配合垫牌，帮助清手",
        "接风权很关键，队友出完牌要及时接风",
        "不要盲目炸队友的牌",
        "弱牌时主动辅助队友上游",
        "出牌节奏暗示手牌强度",
    ],
)

EIGHT_HEARTS_CONFIG = ModeConfig(
    mode=GameMode.EIGHT_HEARTS,
    display_name="八红桃",
    aggression=0.6,
    bomb_threshold=0.5,
    wild_priority=0.7,
    team_cooperation=0.5,
    description=(
        "八红桃玩法：红桃花色有特殊加成。"
        "红桃牌在牌型比较中有额外优势。"
        "需要特别关注红桃牌的收集和使用。"
    ),
    tips=[
        "红桃牌价值更高，优先保留",
        "红桃组合牌型有加成，优先组红桃顺子/连对",
        "级牌(逢人配)为红桃级牌，双重价值",
        "对手红桃多时要警惕大牌型",
        "适当牺牲其他花色保红桃完整性",
    ],
)

RANKED_CONFIG = ModeConfig(
    mode=GameMode.RANKED,
    display_name="排位竞技",
    aggression=0.55,
    bomb_threshold=0.65,
    wild_priority=0.5,
    team_cooperation=0.6,
    description=(
        "排位竞技模式：积分制排名赛。"
        "胜负影响段位，需要稳定发挥。"
        "对手水平较高，策略需更谨慎。"
    ),
    tips=[
        "稳扎稳打，避免冒险拆牌",
        "记牌更加重要，关注关键牌去向",
        "炸弹使用需精打细算",
        "弱牌局保三游，避免双下",
        "强牌局争双上，最大化升级",
    ],
)

LEAGUE_CONFIG = ModeConfig(
    mode=GameMode.LEAGUE,
    display_name="苏掼联赛",
    aggression=0.5,
    bomb_threshold=0.7,
    wild_priority=0.5,
    team_cooperation=0.7,
    max_rounds=8,
    description=(
        "苏掼联赛：正式比赛模式，限时限副。"
        "每轮50分钟或8副牌。过A不打。"
        "积分制，需要全局统筹。"
    ),
    tips=[
        "比赛限时，注意节奏控制",
        "积分制下每副牌都重要",
        "避免冲动，每手牌深思熟虑",
        "关注总比分差距调整策略激进度",
        "落后时适当冒险，领先时稳健防守",
    ],
)

# ── Mode registry ──

MODE_CONFIGS: Dict[GameMode, ModeConfig] = {
    GameMode.CLASSIC: CLASSIC_CONFIG,
    GameMode.NO_SHUFFLE: NO_SHUFFLE_CONFIG,
    GameMode.TEAM_2V2: TEAM_2V2_CONFIG,
    GameMode.EIGHT_HEARTS: EIGHT_HEARTS_CONFIG,
    GameMode.RANKED: RANKED_CONFIG,
    GameMode.LEAGUE: LEAGUE_CONFIG,
}


def get_mode_config(mode: GameMode) -> ModeConfig:
    """Return the strategy config for a given game mode."""
    if mode not in MODE_CONFIGS:
        log.warning("Unknown mode %s, falling back to CLASSIC", mode)
        return CLASSIC_CONFIG
    return MODE_CONFIGS[mode]


def get_strategy_params(mode: GameMode) -> Dict[str, float]:
    """Return strategy parameter dict suitable for Strategy constructor."""
    cfg = get_mode_config(mode)
    return {
        "aggression": cfg.aggression,
        "bomb_threshold": cfg.bomb_threshold,
        "wild_priority": cfg.wild_priority,
        "team_cooperation": cfg.team_cooperation,
    }


def list_modes() -> List[Dict[str, str]]:
    """Return a list of all supported modes with display info."""
    return [
        {
            "mode": cfg.mode.value,
            "name": cfg.display_name,
            "description": cfg.description,
        }
        for cfg in MODE_CONFIGS.values()
    ]
