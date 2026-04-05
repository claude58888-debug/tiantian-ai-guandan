# 天天爱掼蛋 AI Agent

> 掼蛋牌类AI智能体 - 规则引擎、图像识别、AI策略

[![CI](https://github.com/claude58888-debug/tiantian-ai-guandan/actions/workflows/ci.yml/badge.svg)](https://github.com/claude58888-debug/tiantian-ai-guandan/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 项目简介

本项目是一个掼蛋牌类游戏的AI智能体，针对[天天爱掌蛋](https://www.ttigd.com/)客户端设计。包含完整的掼蛋规则引擎、多级AI策略、图像识别层和CLI交互界面。

## 功能特性

- **完整规则引擎**: 支持掼蛋全部牌型（单牌/对子/三条/三带二/顺子/连对/钢板/炸弹/火箭）
- **逢人配逻辑**: 红桃级牌万能牌替代规则
- **多级AI策略**: Random/Greedy/Smart三种策略，可调节侵略度
- **截屏识别**: 游戏窗口截取与牌面OCR识别
- **CLI交互**: 命令行交互式打牌验证

## 安装

```bash
pip install -e .
```

## 使用

```bash
# 命令行模式
python -m guandan

# 在代码中使用
from guandan.models import Card, Rank, Suit, Deck
from guandan.combos import classify_combo
from guandan.strategy import get_strategy
from guandan.wild import is_wild, classify_with_wilds
```

## 项目结构

```
guandan/
├── models.py          # Card, Deck, Hand 数据模型 (Atom 1.1)
├── combos.py          # 牌型识别引擎 (Atom 1.2)
├── compare.py         # 牌型比较与压制逻辑 (Atom 1.3)
├── wild.py            # 逢人配万能牌逻辑 (Atom 1.4)
├── game.py            # 游戏状态机 (Atom 1.5)
├── cli.py             # CLI交互界面 (Atom 1.6)
├── strategy.py        # AI策略模块 (Atom 3.1)
├── card_detector.py   # 牌面识别 (Atom 2.2)
├── screen_capture.py  # 截屏采集 (Atom 2.1)
└── __main__.py        # python -m guandan 入口
tests/
├── test_models.py
├── test_combos.py
├── test_compare.py
├── test_wild.py
├── test_game.py
├── test_cli.py
├── test_strategy.py
├── test_card_detector.py
└── test_screen_capture.py
```

## AI策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `random` | 随机出牌 | 测试/基准 |
| `greedy` | 贪心策略，最小牌先出 | 默认策略 |
| `smart` | 智能策略，考虑手牌结构 | 实战推荐 |

```python
from guandan.strategy import get_strategy

s = get_strategy('smart', aggression=0.7)
play = s.play(hand, last_play, level)
```

## 开发路线图

- [x] **M1**: 纯规则引擎 + CLI验证
- [ ] **M2**: 图像识别层
- [ ] **M3**: UI层集成

## 测试

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT
