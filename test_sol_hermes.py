"""Tests for sol-hermes. Run: python -m unittest test_sol_hermes -v"""

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import __init__ as plugin


class TestConfig(unittest.TestCase):
    def test_defaults_load(self):
        self.assertTrue(plugin.CFG["action_fusion"])
        self.assertTrue(plugin.CFG["observation_pack"])
        self.assertEqual(plugin.CFG["observation_threshold"], 4000)

    def test_token_counter(self):
        stats = plugin.count_saved("a" * 3000, "b" * 300)
        self.assertEqual(stats["tokens_saved"], 900)
        self.assertEqual(stats["saved_percent"], 90.0)

    def test_config_crash_graceful(self):
        self.assertIsNotNone(plugin.CFG)


class TestEconomicCompactor(unittest.TestCase):
    def test_should_compact_economic(self):
        result = plugin._compactor.should_compact(50000, 100000)
        self.assertIn("should_compact", result)
        self.assertIn("economically_favorable", result)
        self.assertIn("window_pressure", result)

    def test_should_not_compact_low_pressure(self):
        result = plugin._compactor.should_compact(10000, 100000)
        self.assertFalse(result["window_pressure"])

    def test_record_compaction_removed(self):
        # record_compaction was removed as dead code
        self.assertFalse(hasattr(plugin._compactor, 'record_compaction'))


class TestActionFusion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.test_file = os.path.join(self.tmp, "test.py")
        with open(self.test_file, "w") as f:
            f.write("def old():\n    pass\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_patch_and_validate(self):
        result = json.loads(plugin.sol_patch_validate(
            self.test_file, "def old():", "def new():", validate=True
        ))
        self.assertTrue(result["ok"])
        self.assertTrue(result["edit_applied"])
        self.assertIn("validation", result)
        self.assertIn("turns_saved", result)

    def test_patch_no_validate(self):
        result = json.loads(plugin.sol_patch_validate(
            self.test_file, "def old():", "def new():", validate=False
        ))
        self.assertTrue(result["ok"])
        self.assertIsNone(result["validation"])

    def test_patch_string_not_found(self):
        result = json.loads(plugin.sol_patch_validate(
            self.test_file, "nonexistent", "replacement"
        ))
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_empty_old_string_rejected(self):
        result = json.loads(plugin.sol_patch_validate(
            self.test_file, "", "replacement"
        ))
        self.assertFalse(result["ok"])
        self.assertIn("empty", result["error"])

    def test_ambiguous_match_rejected(self):
        with open(self.test_file, "w") as f:
            f.write("x = 1\nx = 2\nx = 3\n")
        result = json.loads(plugin.sol_patch_validate(
            self.test_file, "x = ", "y = ", replace_all=False
        ))
        self.assertFalse(result["ok"])
        self.assertIn("ambiguous", result["error"])

    def test_no_shell_injection(self):
        evil = os.path.join(self.tmp, "test;touch INJECTED;.py")
        with open(evil, "w") as f:
            f.write("x = 1\n")
        result = json.loads(plugin.sol_patch_validate(evil, "x = 1", "x = 2", validate=True))
        self.assertTrue(result["ok"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "INJECTED")))


class TestObservationPack(unittest.TestCase):
    def setUp(self):
        self.orig_dir = plugin._projector.dir
        self.tmp = tempfile.mkdtemp()
        plugin._projector.dir = Path(self.tmp)

    def tearDown(self):
        plugin._projector.dir = self.orig_dir
        shutil.rmtree(self.tmp)

    def test_small_content_inline(self):
        result = json.loads(plugin.context_project("small"))
        self.assertEqual(result["type"], "inline")

    def test_large_content_projection(self):
        big = "x" * 5000
        result = json.loads(plugin.context_project(big))
        self.assertEqual(result["type"], "projection")
        self.assertIn("handle_id", result)
        self.assertIn("token_stats", result)

    def test_recall_exact(self):
        big = "line1\n" * 1000
        packed = json.loads(plugin.context_project(big))
        recalled = json.loads(plugin.context_recall(packed["handle_id"], 0, 12))
        self.assertEqual(recalled["chunk"], "line1\nline1\n")

    def test_session_isolation(self):
        big = "x" * 5000
        p1 = json.loads(plugin.context_project(big, session_id="a"))
        p2 = json.loads(plugin.context_project(big, session_id="b"))
        self.assertNotEqual(os.path.dirname(p1["path"]), os.path.dirname(p2["path"]))

    def test_session_sanitized(self):
        big = "x" * 5000
        result = json.loads(plugin.context_project(big, session_id="../../evil"))
        self.assertNotIn("..", result["path"])

    def test_recall_pagination(self):
        big = "0123456789" * 500
        packed = json.loads(plugin.context_project(big))
        page1 = json.loads(plugin.context_recall(packed["handle_id"], 0, 100))
        page2 = json.loads(plugin.context_recall(packed["handle_id"], 100, 100))
        page_last = json.loads(plugin.context_recall(packed["handle_id"], 4900, 100))
        self.assertTrue(page1["has_more"])
        self.assertEqual(page1["next_offset"], 100)
        self.assertTrue(page2["has_more"])
        self.assertFalse(page_last["has_more"])
        self.assertIsNone(page_last["next_offset"])

    def test_recall_negative_offset(self):
        big = "x" * 5000
        packed = json.loads(plugin.context_project(big))
        result = json.loads(plugin.context_recall(packed["handle_id"], -5, 100))
        self.assertIn("error", result)

    def test_recall_offset_string(self):
        big = "x" * 5000
        packed = json.loads(plugin.context_project(big))
        result = json.loads(plugin.context_recall(packed["handle_id"], "0", 100))
        self.assertIn("error", result)
        self.assertEqual(result["error"], "offset must be an integer")

    def test_recall_limit_string(self):
        big = "x" * 5000
        packed = json.loads(plugin.context_project(big))
        result = json.loads(plugin.context_recall(packed["handle_id"], 0, "100"))
        self.assertIn("error", result)
        self.assertEqual(result["error"], "limit must be an integer")

    def test_recall_bad_limit(self):
        big = "x" * 5000
        packed = json.loads(plugin.context_project(big))
        result = json.loads(plugin.context_recall(packed["handle_id"], 0, 99999))
        self.assertIn("error", result)

    def test_non_string_input(self):
        result = json.loads(plugin.context_project(None))
        self.assertEqual(result["type"], "inline")

    def test_handle_id_path_traversal_blocked(self):
        result = json.loads(plugin.context_recall("../../etc/passwd"))
        self.assertIn("error", result)
        self.assertEqual(result["error"], "invalid handle_id")

    def test_handle_id_sanitization(self):
        result = json.loads(plugin.context_recall("obs_20260921_abc123; rm -rf /"))
        self.assertIn("error", result)

    def test_concurrent_put_same_content(self):
        big = "x" * 5000
        results = []
        errors = []

        def worker():
            try:
                r = json.loads(plugin.context_project(big, session_id="concurrent"))
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        handle_ids = {r["handle_id"] for r in results}
        self.assertEqual(len(handle_ids), 1)

    def test_concurrent_put_different_content(self):
        errors = []

        def worker(i):
            try:
                content = f"content_{i}_" + "x" * 5000
                plugin.context_project(content, session_id="concurrent")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestEvidenceReducer(unittest.TestCase):
    def test_small_log_full(self):
        log = "line1\nline2"
        result = json.loads(plugin.evidence_compress(log))
        self.assertEqual(result["type"], "full")

    def test_error_extraction(self):
        lines = ["INFO: ok"] * 100 + ["ERROR: failed"] + ["INFO: ok"] * 100
        log = "\n".join(lines)
        result = json.loads(plugin.evidence_compress(log))
        self.assertEqual(result["type"], "compressed")
        self.assertGreaterEqual(result["sections"], 1)
        self.assertIn("token_stats", result)

    def test_word_boundary(self):
        lines = ["INFO: errorless operation"] * 100
        log = "\n".join(lines)
        result = json.loads(plugin.evidence_compress(log))
        self.assertEqual(result["sections"], 0)

    def test_verify_quotes(self):
        lines = ["INFO: ok"] * 50 + ["ERROR: test failure"] + ["INFO: ok"] * 50
        log = "\n".join(lines)
        compressed = json.loads(plugin.evidence_compress(log))
        verified = json.loads(plugin.evidence_verify(compressed["content"], log))
        self.assertTrue(verified["all_ok"])
        self.assertGreater(verified["checked"], 0)

    def test_verify_fails_on_forged_quote(self):
        compressed = "--- section 1 (lines 1-3) ---\nforged line\n\n"
        verified = json.loads(plugin.evidence_verify(compressed, "original log without it"))
        self.assertFalse(verified["all_ok"])

    def test_verify_no_sections(self):
        verified = json.loads(plugin.evidence_verify("no sections here", "original"))
        self.assertFalse(verified["all_ok"])
        self.assertIn("error", verified)

    def test_bad_input_types(self):
        result = json.loads(plugin.evidence_compress(None))
        self.assertIn("error", result)
        result = json.loads(plugin.evidence_compress("log", "not_a_number"))
        self.assertEqual(json.loads(plugin.evidence_compress("log", 3))["type"], "full")


class TestTransformHook(unittest.TestCase):
    def setUp(self):
        self.orig_dir = plugin._projector.dir
        self.tmp = tempfile.mkdtemp()
        plugin._projector.dir = Path(self.tmp)

    def tearDown(self):
        plugin._projector.dir = self.orig_dir
        shutil.rmtree(self.tmp)

    def test_small_result_unchanged(self):
        result = plugin._transform_result("terminal", {}, "small output")
        self.assertEqual(result, "small output")

    def test_large_result_replaced(self):
        big = "x" * 5000
        result = plugin._transform_result("terminal", {}, big)
        data = json.loads(result)
        self.assertEqual(data["type"], "projection")
        self.assertLess(len(result), len(big))

    def test_non_string_unchanged(self):
        result = plugin._transform_result("terminal", {}, {"not": "string"})
        self.assertEqual(result, {"not": "string"})

    def test_wrong_tool_unchanged(self):
        result = plugin._transform_result("web_search", {}, "x" * 5000)
        self.assertEqual(result, "x" * 5000)

    def test_hook_signature_matches_hermes(self):
        import inspect
        sig = inspect.signature(plugin._transform_result)
        params = list(sig.parameters.keys())
        self.assertEqual(params[0], "tool_name")
        self.assertEqual(params[1], "args")
        self.assertEqual(params[2], "result")
        self.assertEqual(params[3], "kw")


class TestStorageCleanup(unittest.TestCase):
    def test_cleanup_old(self):
        tmp = tempfile.mkdtemp()
        store = plugin.ContextProjector()
        store.dir = Path(tmp)
        store._cleanup_old()
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
