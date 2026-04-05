from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import tkinter as tk
from tkinter import ttk


@dataclass
class Suggestion:
    cards: list[str]
    combo_type: str
    score: float
    reason: str
    source: str = "rule_based"
    risk_level: str = "SAFE"


@dataclass
class MainWindowViewModel:
    app_status: str = "idle"
    session_id: str = "demo-001"
    game_mode: str = "classic"
    pipeline_status: str = "observe"
    last_refresh_at: str = ""
    phase: str = "running"
    current_player: str = "self"
    last_valid_player: str = "right"
    last_valid_type: str = "pair"
    last_valid_cards: list[str] = field(default_factory=lambda: ["S8", "H8"])
    pass_chain: int = 0
    hand_cards: list[str] = field(
        default_factory=lambda: ["S3", "H3", "C4", "D4", "S5", "H5", "S6", "H6"]
    )
    recognition_confidence: float = 0.92
    auto_detect_enabled: bool = False
    calibration_region: tuple[int, int, int, int] = (0, 0, 1024, 768)
    remaining_counts: dict[str, int] = field(
        default_factory=lambda: {"self": 8, "left": 10, "partner": 6, "right": 3}
    )
    suggestions: list[Suggestion] = field(
        default_factory=lambda: [
            Suggestion(["S6", "H6"], "pair", 0.87, "低成本跟牌，保留高价值牌组", risk_level="SAFE"),
            Suggestion(["S5", "H5"], "pair", 0.73, "可以压住，但后续衔接较弱", risk_level="MODERATE"),
            Suggestion([], "pass", 0.42, "可选择过牌，观察队友接管", risk_level="SAFE"),
        ]
    )
    logs: list[str] = field(
        default_factory=lambda: [
            "[info] UI initialized",
            "[info] Loaded demo state",
            "[info] Waiting for next refresh",
        ]
    )


class GuandanUI(tk.Tk):
    def __init__(self, vm: MainWindowViewModel) -> None:
        super().__init__()
        self.vm = vm
        self.title("天天爱掼蛋辅助 Agent - M3.1")
        self.geometry("1100x760")
        self.minsize(900, 640)

        self._build_styles()
        self._build_layout()
        self.refresh_view()

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        self.status_bar = ttk.Frame(self, padding=10)
        self.status_bar.grid(row=0, column=0, sticky="ew")
        self.status_bar.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(self.status_bar, text="")
        self.status_label.grid(row=0, column=0, sticky="w")

        self.refresh_button = ttk.Button(
            self.status_bar, text="刷新状态", command=self.handle_refresh
        )
        self.refresh_button.grid(row=0, column=1, padx=6)

        self.pause_button = ttk.Button(
            self.status_bar, text="暂停/继续", command=self.handle_pause
        )
        self.pause_button.grid(row=0, column=2, padx=6)

        self.auto_detect_var = tk.BooleanVar(value=False)
        self.auto_detect_button = ttk.Checkbutton(
            self.status_bar, text="自动识别",
            variable=self.auto_detect_var,
            command=self.handle_auto_detect_toggle,
        )
        self.auto_detect_button.grid(row=0, column=3, padx=6)

        self.calibrate_button = ttk.Button(
            self.status_bar, text="校准区域", command=self.handle_calibrate
        )
        self.calibrate_button.grid(row=0, column=4, padx=6)

        body = ttk.Frame(self, padding=10)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)

        self.game_state_frame = ttk.LabelFrame(left, text="当前牌局状态", padding=10)
        self.game_state_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.game_state_text = tk.Text(self.game_state_frame, height=8, wrap="word")
        self.game_state_text.pack(fill="both", expand=True)

        self.hand_frame = ttk.LabelFrame(left, text="手牌展示区", padding=10)
        self.hand_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        left.rowconfigure(1, weight=1)

        self.hand_text = tk.Text(self.hand_frame, height=10, wrap="word")
        self.hand_text.pack(fill="both", expand=True)

        self.confidence_label = ttk.Label(self.hand_frame, text="")
        self.confidence_label.pack(anchor="w")

        self.suggestion_frame = ttk.LabelFrame(right, text="建议动作", padding=10)
        self.suggestion_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        right.rowconfigure(0, weight=1)

        self.suggestion_text = tk.Text(self.suggestion_frame, height=16, wrap="word")
        self.suggestion_text.pack(fill="both", expand=True)

        self.summary_frame = ttk.LabelFrame(right, text="玩家剩余张数", padding=10)
        self.summary_frame.grid(row=1, column=0, sticky="ew")

        self.summary_text = tk.Text(self.summary_frame, height=6, wrap="word")
        self.summary_text.pack(fill="both", expand=True)

        self.log_frame = ttk.LabelFrame(self, text="日志区", padding=10)
        self.log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.log_frame.columnconfigure(0, weight=1)
        self.log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(self.log_frame, height=10, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def refresh_view(self) -> None:
        self.vm.last_refresh_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        status = (
            f"状态: {self.vm.app_status} | "
            f"模式: {self.vm.game_mode} | "
            f"会话: {self.vm.session_id} | "
            f"阶段: {self.vm.pipeline_status} | "
            f"刷新时间: {self.vm.last_refresh_at}"
        )
        self.status_label.config(text=status)

        self._set_text(
            self.game_state_text,
            "\n".join(
                [
                    f"phase: {self.vm.phase}",
                    f"current_player: {self.vm.current_player}",
                    f"last_valid_player: {self.vm.last_valid_player}",
                    f"last_valid_type: {self.vm.last_valid_type}",
                    f"last_valid_cards: {' '.join(self.vm.last_valid_cards)}",
                    f"pass_chain: {self.vm.pass_chain}",
                    f"recognition_confidence: {self.vm.recognition_confidence:.2f}",
                ]
            ),
        )

        self._set_text(
            self.hand_text,
            f"hand_count: {len(self.vm.hand_cards)}\n\n" + " ".join(self.vm.hand_cards),
        )

        self._update_confidence_indicator()

        suggestion_lines = []
        for i, s in enumerate(self.vm.suggestions, start=1):
            cards = "PASS" if not s.cards else " ".join(s.cards)
            suggestion_lines.append(
                f"{i}. [{s.combo_type}] {cards}\n"
                f"   confidence={s.score:.2f} risk={s.risk_level} source={s.source}\n"
                f"   reason={s.reason}"
            )
        self._set_text(self.suggestion_text, "\n\n".join(suggestion_lines))

        self._set_text(
            self.summary_text,
            "\n".join(
                [
                    f"self: {self.vm.remaining_counts['self']}",
                    f"left: {self.vm.remaining_counts['left']}",
                    f"partner: {self.vm.remaining_counts['partner']}",
                    f"right: {self.vm.remaining_counts['right']}",
                ]
            ),
        )

        self._set_text(self.log_text, "\n".join(self.vm.logs))

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.config(state="disabled")

    def handle_refresh(self) -> None:
        self.vm.logs.append("[info] Manual refresh triggered")
        self.vm.app_status = "synced"
        self.vm.pipeline_status = "decide"
        self.refresh_view()

    def handle_pause(self) -> None:
        if self.vm.app_status == "paused":
            self.vm.app_status = "observing"
            self.vm.logs.append("[info] Observation resumed")
        else:
            self.vm.app_status = "paused"
            self.vm.logs.append("[info] Observation paused")
        self.refresh_view()

    def handle_auto_detect_toggle(self) -> None:
        enabled = self.auto_detect_var.get()
        self.vm.auto_detect_enabled = enabled
        if enabled:
            self.vm.logs.append("[info] Auto detect enabled")
        else:
            self.vm.logs.append("[info] Auto detect disabled")
        self.refresh_view()

    def handle_calibrate(self) -> None:
        dialog = CalibrationDialog(self, self.vm.calibration_region)
        self.wait_window(dialog)
        if dialog.result is not None:
            self.vm.calibration_region = dialog.result
            self.vm.logs.append(
                f"[info] Calibration updated: {dialog.result}"
            )
            self.refresh_view()

    def _update_confidence_indicator(self) -> None:
        conf = self.vm.recognition_confidence
        if conf >= 0.8:
            color = "green"
            label = "高"
        elif conf >= 0.5:
            color = "#cc8800"
            label = "中"
        else:
            color = "red"
            label = "低"
        text = f"识别置信度: {conf:.0%} ({label})"
        self.confidence_label.config(text=text, foreground=color)


class CalibrationDialog(tk.Toplevel):
    """Dialog for selecting the game window capture region.

    The user enters x, y, width, height values to define the area
    of the screen that contains the game window.
    """

    def __init__(
        self,
        parent: tk.Tk,
        current: tuple[int, int, int, int],
    ) -> None:
        super().__init__(parent)
        self.title("校准游戏区域")
        self.geometry("320x200")
        self.resizable(False, False)
        self.result: tuple[int, int, int, int] | None = None

        self._entries: dict[str, tk.Entry] = {}
        labels = ["X", "Y", "宽度", "高度"]
        for i, (label, val) in enumerate(zip(labels, current)):
            ttk.Label(self, text=label).grid(row=i, column=0, padx=10, pady=4, sticky="e")
            entry = ttk.Entry(self, width=12)
            entry.insert(0, str(val))
            entry.grid(row=i, column=1, padx=10, pady=4)
            self._entries[label] = entry

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Button(btn_frame, text="确认", command=self._on_ok).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side="left", padx=8)

        self.transient(parent)
        self.grab_set()

    def _on_ok(self) -> None:
        try:
            values = tuple(
                int(self._entries[k].get())
                for k in ["X", "Y", "宽度", "高度"]
            )
            if len(values) == 4:
                self.result = values  # type: ignore[assignment]
        except ValueError:
            pass
        self.destroy()


def main() -> None:
    vm = MainWindowViewModel(app_status="observing")
    app = GuandanUI(vm)
    app.mainloop()


if __name__ == "__main__":
    main()
