#!/usr/bin/env pythonw
"""Double-click launcher for Guandan AI Assistant.

This .pyw file runs without a console window on Windows.
Simply double-click to start the AI assistant GUI.

It will:
1. Auto-detect the 天天爱掼蛋 game window
2. Identify the current game mode
3. Load optimized AI strategy
4. Display real-time play suggestions
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guandan.app_launcher import main

if __name__ == "__main__":
    main()
