"""Headless TUI regression tests (no real terminal needed)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolutioner.config import Config  # noqa: E402


def make_config(tmp: str) -> Config:
    return Config(
        enable_rag=False,
        rag_dir=f"{tmp}/memory",
        memories_dir=f"{tmp}/memories",
        session_dir=f"{tmp}/sessions",
        usage_db=f"{tmp}/usage.db",
    )


class TestTUI(unittest.IsolatedAsyncioTestCase):
    async def test_compose_refresh_screenshot(self):
        from textual.app import App  # noqa: F401

        from evolutioner.tui_app import EvolutionerTUI

        with tempfile.TemporaryDirectory() as tmp:
            app = EvolutionerTUI(make_config(tmp))
            async with app.run_test(size=(120, 34)) as pilot:
                app.refresh_telemetry()
                await pilot.pause()
                # core DOM sanity
                for selector in ("#ov_left", "#ov_right", "#mo_left", "#mo_right",
                                 "#us_left", "#us_right", "#agentlog", "#prompt"):
                    self.assertIsNotNone(app.query_one(selector))
                # simulate a typed command without hitting the network
                app.run_query = lambda q: None  # guard: no live call in test
                await pilot.press("r")  # refresh action
                await pilot.pause()
                svg = app.export_screenshot()  # uses the live console driver
                out = Path(tmp) / "tui_preview.svg"
                out.write_text(svg, encoding="utf-8")
                self.assertIn("<svg", svg[:200])


if __name__ == "__main__":
    unittest.main()
