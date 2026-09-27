"""Unit tests for Evolutioner MK1 — no model downloads required."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolutioner.config import Config, VARIANTS  # noqa: E402
from evolutioner.context import VirtualContextEngine  # noqa: E402
from evolutioner.harness import Harness  # noqa: E402
from evolutioner.memory import MemoryStore  # noqa: E402
from evolutioner.mock import MockLLM  # noqa: E402
from evolutioner.search import EvolutionerMCTS, MCTSProcessRewardModel  # noqa: E402
from evolutioner.usage import UsageTracker  # noqa: E402
from evolutioner.verifier import SandboxVerifier  # noqa: E402


def make_config(tmp: str, **kw) -> Config:
    defaults = dict(
        enable_rag=False, rag_dir=f"{tmp}/memory", memories_dir=f"{tmp}/memories",
        session_dir=f"{tmp}/sessions", usage_db=f"{tmp}/usage.db",
        use_bwrap=False, mcts_iterations=2, mcts_depth=2, candidates=2,
    )
    defaults.update(kw)
    return Config(**defaults)


class TestConfig(unittest.TestCase):
    def test_variants_complete(self):
        self.assertEqual(set(VARIANTS), {"1.5B", "2B", "3B", "5B"})
        for spec in VARIANTS.values():
            self.assertTrue(spec.repo_id.startswith("wallpillar-lm/"))
            self.assertTrue(spec.filename.endswith(".gguf"))

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            Config(variant="7B")

    def test_load_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cfg.json"
            p.write_text('{"variant": "2B", "unknown_key": 1, "n_ctx": 512}')
            cfg = Config.load(str(p))
            self.assertEqual(cfg.variant, "2B")
            self.assertEqual(cfg.n_ctx, 512)
            self.assertEqual(cfg.extra.get("unknown_key"), 1)


class TestUsageTracker(unittest.TestCase):
    def test_record_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = UsageTracker(f"{tmp}/usage.db")
            tr.record("mcts", "mock", 10, 5, 0.5)
            tr.record("answer", "mock", 8, 20, 1.0)
            s = tr.summary()
            self.assertEqual(s["total_calls"], 2)
            self.assertEqual(s["prompt_tokens"], 18)
            self.assertEqual(s["gen_tokens"], 25)
            self.assertEqual(s["by_kind"]["answer"]["calls"], 1)
            self.assertEqual(len(tr.recent(5)), 2)
            tr.close()


class TestVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = SandboxVerifier(use_bwrap=False)

    def test_ok(self):
        r = self.v.verify("print(21 * 2)")
        self.assertTrue(r.ok)
        self.assertIn("42", r.stdout)

    def test_syntax_error(self):
        r = self.v.verify("def f(:\n  pass")
        self.assertFalse(r.ok)

    def test_timeout(self):
        r = self.v.verify("while True: pass", )
        r2 = SandboxVerifier(timeout_s=1.0, use_bwrap=False).verify("while True: pass")
        self.assertFalse(r2.ok)
        self.assertIn("timeout", r2.stderr)

    def test_empty(self):
        self.assertFalse(self.v.verify("").ok)

    def test_import_blocked_in_real_sandbox(self):
        real = SandboxVerifier(use_bwrap=None)
        if not real.use_bwrap:
            self.skipTest("bwrap not available")
        r = real.verify("import socket\nprint('leak')")
        self.assertFalse(r.ok)


class TestPrm(unittest.TestCase):
    def test_reward_bounds(self):
        prm = MCTSProcessRewardModel(verifier=None)
        self.assertEqual(prm.evaluate_step("plain text step"), 0.5)
        self.assertEqual(prm.evaluate_step("```python\ndef f(:\n```"), 0.0)
        self.assertGreaterEqual(prm.evaluate_step("```python\nprint(1+1)\n```"), 0.5)

    def test_fenced_extraction(self):
        prm = MCTSProcessRewardModel(verifier=None)
        score = prm.evaluate_step("text\n```python\nx = 1\n```\nmore")
        self.assertGreaterEqual(score, 0.5)  # valid python, no dynamic verdict
        expr = prm.evaluate_step("```python\nprint(2+2)\n```")
        self.assertGreaterEqual(expr, 0.5)


class TestMCTS(unittest.TestCase):
    def test_search_runs_and_backprops(self):
        llm = MockLLM(responses=["```python\nprint(2+2)\n```", "Final Answer: 4"])
        mcts = EvolutionerMCTS(llm, verifier=SandboxVerifier(use_bwrap=False),
                               candidates=2, temperature=0.1)
        res = mcts.search("2+2?", iterations=2, max_depth=2)
        self.assertTrue(res.think.startswith("<think>"))
        self.assertTrue(res.think.endswith("</think>"))
        self.assertEqual(res.iterations, 2)

    def test_terminal_step_stops_trajectory(self):
        llm = MockLLM(default="Final Answer: 42")
        mcts = EvolutionerMCTS(llm, verifier=None, candidates=1)
        res = mcts.search("meaning of life", iterations=3, max_depth=4)
        self.assertEqual(res.steps[-1], "Final Answer: 42")


class TestContext(unittest.TestCase):
    def test_add_search_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = VirtualContextEngine(f"{tmp}/mem", top_k=2)
            ctx.add("the sky is blue")
            ctx.add("gravity pulls apples down")
            hits = ctx.search("why do apples fall")
            self.assertIn("gravity", hits[0]["text"])
            # persistence across instances
            ctx2 = VirtualContextEngine(f"{tmp}/mem", top_k=2)
            self.assertEqual(len(ctx2.entries), 2)
            block = ctx2.build_context("apples")
            self.assertTrue(block.startswith("- "))


class TestMemory(unittest.TestCase):
    def test_crud_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryStore(f"{tmp}/m")
            rec = mem.add("User's favourite language is Python", tags=["pref"])
            mem.add("Meeting at 3pm on Tuesday")
            self.assertEqual(len(mem.all()), 2)
            hits = mem.search("favourite language")
            self.assertIn("Python", hits[0]["text"])
            tag_hits = mem.search("language", tags=["pref"])
            self.assertEqual(len(tag_hits), 1)
            self.assertTrue(mem.delete(rec["id"]))
            self.assertIsNone(mem.get(rec["id"]))

    def test_empty_add_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MemoryStore(f"{tmp}/m").add("  ")


class TestOllamaBackend(unittest.TestCase):
    def test_portfolio_tags_complete(self):
        from evolutioner.ollama_backend import PORTFOLIO_TAGS

        self.assertEqual(set(PORTFOLIO_TAGS), set(VARIANTS))
        for tag in PORTFOLIO_TAGS.values():
            self.assertTrue(tag.startswith("wallpillar-lm/evolutionermk1-"))

    def test_tiered_routing_used_by_mcts(self):
        from evolutioner.ollama_backend import TieredOllamaBackend

        # Construction is offline (no server calls).
        tb = TieredOllamaBackend(main_variant="3B", rollout_variant="1.5B")
        self.assertIn("tiered", tb.name)
        self.assertTrue(hasattr(tb, "generate_rollout"))

        # MCTS must route rollout calls through generate_rollout when present.
        class TieredMock(MockLLM):
            def __init__(self):
                super().__init__()
                self.rollout_calls = 0

            def generate_rollout(self, prompt, **kw):
                self.rollout_calls += 1
                return super().generate(prompt, **kw)

        tmock = TieredMock()
        mcts = EvolutionerMCTS(tmock, verifier=None, candidates=1)
        mcts.search("q", iterations=2, max_depth=2)
        self.assertGreater(tmock.rollout_calls, 0)


class TestHarness(unittest.TestCase):
    def test_full_pipeline_with_selfcorrection(self):
        llm = MockLLM(responses=[
            "Step: compute.\n```python\nprint(1/0)\n```\nFinal Answer: bad",   # fails verify
            "Final Answer: 4",                                                  # corrected
        ])
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            h = Harness(cfg, llm, tracker=UsageTracker(cfg.usage_db),
                        verifier=SandboxVerifier(use_bwrap=False))
            res = h.run("1/0?", use_mcts=False, use_rag=False)
            self.assertEqual(res.rounds, 2)
            self.assertEqual(res.answer, "4")
            self.assertFalse(res.verification["ok"])
            self.assertGreater(len(h.history), 0)
            # session persisted
            self.assertTrue((Path(cfg.session_dir) / f"{res.session_id}.json").exists())

    def test_usage_tracked(self):
        llm = MockLLM(default="Final Answer: ok")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            tr = UsageTracker(cfg.usage_db)
            self.addCleanup(tr.close)
            h = Harness(cfg, llm, tracker=tr, verifier=SandboxVerifier(use_bwrap=False))
            h.run("hello", use_mcts=False, use_rag=False)
            s = tr.summary()
            self.assertGreaterEqual(s["by_kind"]["answer"]["calls"], 1)
            tr.close()


if __name__ == "__main__":
    unittest.main()
