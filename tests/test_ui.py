"""Tests for guandan.ui module (Atom 3.2).

Tests the ViewModel dataclasses and UI logic without requiring
a display (headless-safe). Tkinter widget tests use mock where needed.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from guandan.ui import Suggestion, MainWindowViewModel


class TestSuggestion(unittest.TestCase):
    """Tests for the Suggestion dataclass."""

    def test_create_with_cards(self) -> None:
        s = Suggestion(["S6", "H6"], "pair", 0.87, "low cost")
        self.assertEqual(s.cards, ["S6", "H6"])
        self.assertEqual(s.combo_type, "pair")
        self.assertAlmostEqual(s.score, 0.87)
        self.assertEqual(s.reason, "low cost")
        self.assertEqual(s.source, "rule_based")

    def test_create_pass(self) -> None:
        s = Suggestion([], "pass", 0.42, "observe")
        self.assertEqual(s.cards, [])
        self.assertEqual(s.combo_type, "pass")

    def test_custom_source(self) -> None:
        s = Suggestion(["S3"], "single", 0.5, "r", source="neural")
        self.assertEqual(s.source, "neural")


class TestMainWindowViewModel(unittest.TestCase):
    """Tests for the MainWindowViewModel dataclass."""

    def test_defaults(self) -> None:
        vm = MainWindowViewModel()
        self.assertEqual(vm.app_status, "idle")
        self.assertEqual(vm.session_id, "demo-001")
        self.assertEqual(vm.game_mode, "classic")
        self.assertEqual(vm.pipeline_status, "observe")
        self.assertEqual(vm.phase, "running")
        self.assertEqual(vm.current_player, "self")
        self.assertAlmostEqual(vm.recognition_confidence, 0.92)

    def test_custom_status(self) -> None:
        vm = MainWindowViewModel(app_status="observing")
        self.assertEqual(vm.app_status, "observing")

    def test_hand_cards_default(self) -> None:
        vm = MainWindowViewModel()
        self.assertEqual(len(vm.hand_cards), 8)
        self.assertIn("S3", vm.hand_cards)

    def test_remaining_counts_default(self) -> None:
        vm = MainWindowViewModel()
        self.assertEqual(vm.remaining_counts["self"], 8)
        self.assertEqual(vm.remaining_counts["left"], 10)
        self.assertEqual(vm.remaining_counts["partner"], 6)
        self.assertEqual(vm.remaining_counts["right"], 3)

    def test_suggestions_default(self) -> None:
        vm = MainWindowViewModel()
        self.assertEqual(len(vm.suggestions), 3)
        self.assertEqual(vm.suggestions[0].combo_type, "pair")
        self.assertEqual(vm.suggestions[2].combo_type, "pass")

    def test_logs_default(self) -> None:
        vm = MainWindowViewModel()
        self.assertEqual(len(vm.logs), 3)
        self.assertIn("[info]", vm.logs[0])

    def test_mutable_hand_cards(self) -> None:
        vm = MainWindowViewModel()
        vm.hand_cards.append("RJ")
        self.assertIn("RJ", vm.hand_cards)
        # Ensure separate instances
        vm2 = MainWindowViewModel()
        self.assertNotIn("RJ", vm2.hand_cards)

    def test_mutable_logs(self) -> None:
        vm = MainWindowViewModel()
        vm.logs.append("[info] test")
        self.assertEqual(len(vm.logs), 4)
        vm2 = MainWindowViewModel()
        self.assertEqual(len(vm2.logs), 3)

    def test_mutable_remaining_counts(self) -> None:
        vm = MainWindowViewModel()
        vm.remaining_counts["self"] = 5
        self.assertEqual(vm.remaining_counts["self"], 5)
        vm2 = MainWindowViewModel()
        self.assertEqual(vm2.remaining_counts["self"], 8)

    def test_last_valid_cards(self) -> None:
        vm = MainWindowViewModel()
        self.assertEqual(vm.last_valid_cards, ["S8", "H8"])
        self.assertEqual(vm.last_valid_type, "pair")


class TestGuandanUILogic(unittest.TestCase):
    """Test UI handler logic without creating real Tk windows."""

    @patch("guandan.ui.tk.Tk.__init__", return_value=None)
    def _make_ui(self, mock_init):
        """Create a GuandanUI with mocked Tk."""
        from guandan.ui import GuandanUI
        ui = object.__new__(GuandanUI)
        ui.vm = MainWindowViewModel(app_status="observing")
        return ui

    def test_handle_refresh_updates_status(self) -> None:
        ui = self._make_ui()
        ui.refresh_view = MagicMock()
        ui.handle_refresh()
        self.assertEqual(ui.vm.app_status, "synced")
        self.assertEqual(ui.vm.pipeline_status, "decide")
        self.assertIn("[info] Manual refresh triggered", ui.vm.logs)
        ui.refresh_view.assert_called_once()

    def test_handle_pause_toggles(self) -> None:
        ui = self._make_ui()
        ui.refresh_view = MagicMock()
        # First call: observing -> paused
        ui.handle_pause()
        self.assertEqual(ui.vm.app_status, "paused")
        self.assertIn("[info] Observation paused", ui.vm.logs)
        # Second call: paused -> observing
        ui.handle_pause()
        self.assertEqual(ui.vm.app_status, "observing")
        self.assertIn("[info] Observation resumed", ui.vm.logs)

    def test_handle_pause_from_idle(self) -> None:
        ui = self._make_ui()
        ui.vm.app_status = "idle"
        ui.refresh_view = MagicMock()
        ui.handle_pause()
        self.assertEqual(ui.vm.app_status, "paused")


class TestSetText(unittest.TestCase):
    """Test _set_text helper."""

    @patch("guandan.ui.tk.Tk.__init__", return_value=None)
    def test_set_text_calls_widget_methods(self, mock_init):
        from guandan.ui import GuandanUI
        ui = object.__new__(GuandanUI)
        ui.vm = MainWindowViewModel()

        widget = MagicMock()
        ui._set_text(widget, "hello world")
        widget.config.assert_any_call(state="normal")
        widget.delete.assert_called_once_with("1.0", "end")
        widget.insert.assert_called_once_with("1.0", "hello world")
        widget.config.assert_called_with(state="disabled")


class TestMainFunction(unittest.TestCase):
    """Test the main() entry point."""

    @patch("guandan.ui.GuandanUI")
    def test_main_creates_app(self, mock_ui_cls) -> None:
        from guandan.ui import main
        mock_instance = MagicMock()
        mock_ui_cls.return_value = mock_instance
        main()
        mock_ui_cls.assert_called_once()
        mock_instance.mainloop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
