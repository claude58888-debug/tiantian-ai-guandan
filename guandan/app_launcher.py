"""One-click application launcher for Guandan AI (天天爱掼蛋 AI助手).

Provides a Tkinter GUI that:
1. Automatically detects the game window (天天爱掼蛋)
2. Identifies which game mode the player selected
3. Loads the appropriate AI strategy for that mode
4. Starts the realtime AI overlay with mode-specific tuning
5. Continuously monitors and adapts to mode changes

Usage:
    python -m guandan.app_launcher
    # Or double-click guandan_app.pyw
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

# Display name
APP_TITLE = "掼蛋AI助手 - 天天爱掼蛋"
APP_VERSION = "1.0.0"

# Game window title patterns
GAME_WINDOW_TITLES = ["天天爱掼蛋", "天天爱摜蛋", "tiantian"]


class GuandanApp:
    """Main application controller.

    Orchestrates window detection, mode detection, strategy loading,
    and the realtime AI overlay pipeline.
    """

    def __init__(self):
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._current_mode = None
        self._game_hwnd = None
        self._status = "等待启动..."

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> None:
        """Start the AI assistant pipeline."""
        if self._running:
            log.warning("Already running")
            return

        self._running = True
        self._status = "正在搜索游戏窗口..."
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="guandan-monitor",
        )
        self._monitor_thread.start()
        log.info("Guandan AI started")

    def stop(self) -> None:
        """Stop the AI assistant."""
        self._running = False
        self._status = "已停止"
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        log.info("Guandan AI stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        from guandan.mode_strategy import GameMode, get_mode_config
        from guandan.mode_detector import ModeTracker

        tracker = ModeTracker(stability_threshold=3)

        while self._running:
            try:
                # Step 1: Find game window
                hwnd = self._find_game_window()
                if not hwnd:
                    self._status = "未检测到天天爱掼蛋，请启动游戏..."
                    time.sleep(2)
                    continue

                self._game_hwnd = hwnd
                self._status = "已检测到游戏窗口，正在识别模式..."

                # Step 2: Capture and detect mode
                mode_result = self._detect_current_mode()
                if mode_result:
                    stable_mode = tracker.update(mode_result)
                    if stable_mode != self._current_mode:
                        self._current_mode = stable_mode
                        cfg = get_mode_config(stable_mode)
                        self._status = (
                            f"模式: {cfg.display_name} | "
                            f"AI策略已加载 | 侵略度: {cfg.aggression}"
                        )
                        log.info("Switched to mode: %s", cfg.display_name)
                else:
                    self._status = "游戏运行中，等待进入对局..."

                time.sleep(1)

            except Exception as e:
                log.error("Monitor error: %s", e)
                self._status = f"错误: {e}"
                time.sleep(3)

    def _find_game_window(self) -> Optional[int]:
        """Find the game window handle."""
        try:
            from guandan.window_finder import find_game_window
            for title in GAME_WINDOW_TITLES:
                result = find_game_window(title)
                if result is not None:
                    return result.hwnd
        except ImportError:
            log.debug("window_finder not available, using fallback")
        except Exception as e:
            log.debug("Window search failed: %s", e)
        return None

    def _detect_current_mode(self):
        """Detect the current game mode from screen."""
        try:
            from guandan.mode_detector import detect_mode_from_screenshot
            from guandan.screen_capture import capture_window

            img = capture_window(self._game_hwnd)
            if img is not None:
                return detect_mode_from_screenshot(img)
        except ImportError:
            log.debug("Screen capture modules not available")
        except Exception as e:
            log.debug("Mode detection failed: %s", e)
        return None


def build_gui(app: GuandanApp):
    """Build and run the Tkinter GUI."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Tkinter not available. Run in CLI mode:")
        print("  python -m guandan.main --realtime")
        return

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("480x600")
    root.resizable(False, False)

    # Try to set icon
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    # ── Header ──
    header = tk.Frame(root, bg="#1a1a2e", height=80)
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(
        header, text="掼蛋AI助手",
        font=("Microsoft YaHei", 20, "bold"),
        fg="white", bg="#1a1a2e",
    ).pack(pady=10)

    tk.Label(
        header, text=f"天天爱掼蛋 智能辅助 v{APP_VERSION}",
        font=("Microsoft YaHei", 10),
        fg="#aaa", bg="#1a1a2e",
    ).pack()

    # ── Status ──
    status_frame = tk.LabelFrame(
        root, text="状态", font=("Microsoft YaHei", 11),
        padx=10, pady=10,
    )
    status_frame.pack(fill="x", padx=15, pady=10)

    status_var = tk.StringVar(value=app.status)
    status_label = tk.Label(
        status_frame, textvariable=status_var,
        font=("Microsoft YaHei", 10),
        wraplength=420, justify="left",
    )
    status_label.pack(anchor="w")

    # ── Mode display ──
    mode_frame = tk.LabelFrame(
        root, text="当前模式", font=("Microsoft YaHei", 11),
        padx=10, pady=10,
    )
    mode_frame.pack(fill="x", padx=15, pady=5)

    mode_var = tk.StringVar(value="等待检测...")
    tk.Label(
        mode_frame, textvariable=mode_var,
        font=("Microsoft YaHei", 14, "bold"),
        fg="#e94560",
    ).pack(anchor="w")

    tips_var = tk.StringVar(value="")
    tk.Label(
        mode_frame, textvariable=tips_var,
        font=("Microsoft YaHei", 9),
        wraplength=420, justify="left", fg="#555",
    ).pack(anchor="w", pady=(5, 0))

    # ── Buttons ──
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=15, pady=10)

    def on_start():
        app.start()
        start_btn.config(state="disabled")
        stop_btn.config(state="normal")

    def on_stop():
        app.stop()
        start_btn.config(state="normal")
        stop_btn.config(state="disabled")

    start_btn = tk.Button(
        btn_frame, text="启动 AI 助手",
        font=("Microsoft YaHei", 12, "bold"),
        bg="#0f3460", fg="white",
        command=on_start,
        width=15, height=2,
    )
    start_btn.pack(side="left", padx=5)

    stop_btn = tk.Button(
        btn_frame, text="停止",
        font=("Microsoft YaHei", 12),
        command=on_stop,
        state="disabled",
        width=10, height=2,
    )
    stop_btn.pack(side="left", padx=5)

    # ── Supported modes list ──
    modes_frame = tk.LabelFrame(
        root, text="支持的模式", font=("Microsoft YaHei", 11),
        padx=10, pady=5,
    )
    modes_frame.pack(fill="x", padx=15, pady=5)

    from guandan.mode_strategy import MODE_CONFIGS
    for cfg in MODE_CONFIGS.values():
        tk.Label(
            modes_frame,
            text=f"  {cfg.display_name}",
            font=("Microsoft YaHei", 9),
            anchor="w",
        ).pack(anchor="w")

    # ── Hotkeys info ──
    tk.Label(
        root,
        text="快捷键: F1=启动  F2=停止  F3=重新校准",
        font=("Microsoft YaHei", 9), fg="#888",
    ).pack(pady=5)

    # ── Status update timer ──
    def update_status():
        status_var.set(app.status)
        if app._current_mode:
            from guandan.mode_strategy import get_mode_config
            cfg = get_mode_config(app._current_mode)
            mode_var.set(cfg.display_name)
            if cfg.tips:
                tips_var.set("\n".join(f"  {t}" for t in cfg.tips[:3]))
        root.after(500, update_status)

    root.after(500, update_status)
    root.mainloop()


def main():
    """Entry point for the app launcher."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = GuandanApp()
    build_gui(app)


if __name__ == "__main__":
    main()
