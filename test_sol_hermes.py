"""Tests for sol-hermes. Run: python -m unittest test_sol_hermes -v"""

import json
import os
import shutil
import sys
import tempfile
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
        self.assertIn("token_stats", result)

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

    def test_no_shell_injection(self):
        # Create a file with a shell metachar in name
        evil = os.path.join(self.tmp, "test;touch INJECTED;.py")
        with open(evil, "w") as f:
            f.write("x = 1\n")
        result = json.loads(plugin.sol_patch_validate(evil, "x = 1", "x = 2", validate=True))
        self.assertTrue(result["ok"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "INJECTED")))


class TestObservationPack(unittest.TestCase):
    def setUp(self):
        self.orig_dir = plugin._store.dir
        self.tmp = tempfile.mkdtemp()
        plugin._store.dir = Path(self.tmp)

    def tearDown(self):
        plugin._store.dir = self.orig_dir
        shutil.rmtree(self.tmp)

    def test_small_content_inline(self):
        result = json.loads(plugin.observation_pack("small"))
        self.assertEqual(result["type"], "inline")

    def test_large_content_handle(self):
        big = "x" * 5000
        result = json.loads(plugin.observation_pack(big))
        self.assertEqual(result["type"], "handle")
        self.assertIn("handle_id", result)
        self.assertIn("token_stats", result)

    def test_recall_exact(self):
        big = "line1\n" * 1000
        packed = json.loads(plugin.observation_pack(big))
        recalled = json.loads(plugin.observation_recall(packed["handle_id"], 0, 12))
        self.assertEqual(recalled["chunk"], "line1\nline1\n")

    def test_session_isolation(self):
        big = "x" * 5000
        p1 = json.loads(plugin.observation_pack(big, session_id="a"))
        p2 = json.loads(plugin.observation_pack(big, session_id="b"))
        self.assertNotEqual(os.path.dirname(p1["path"]), os.path.dirname(p2["path"]))

    def test_session_sanitized(self):
        big = "x" * 5000
        result = json.loads(plugin.observation_pack(big, session_id="../../evil"))
        self.assertNotIn("..", result["path"])

    def test_recall_pagination(self):
        big = "0123456789" * 500
        packed = json.loads(plugin.observation_pack(big))
        page1 = json.loads(plugin.observation_recall(packed["handle_id"], 0, 100))
        page2 = json.loads(plugin.observation_recall(packed["handle_id"], 100, 100))
        page_last = json.loads(plugin.observation_recall(packed["handle_id"], 4900, 100))
        self.assertTrue(page1["has_more"])
        self.assertEqual(page1["next_offset"], 100)
        self.assertTrue(page2["has_more"])
        self.assertFalse(page_last["has_more"])
        self.assertIsNone(page_last["next_offset"])

    def test_recall_negative_offset(self):
        big = "x" * 5000
        packed = json.loads(plugin.observation_pack(big))
        result = json.loads(plugin.observation_recall(packed["handle_id"], -5, 100))
        self.assertIn("error", result)

    def test_recall_bad_limit(self):
        big = "x" * 5000
        packed = json.loads(plugin.observation_pack(big))
        result = json.loads(plugin.observation_recall(packed["handle_id"], 0, 99999))
        self.assertIn("error", result)

    def test_non_string_input(self):
        result = json.loads(plugin.observation_pack(None))
        self.assertEqual(result["type"], "inline")


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
        self.orig_dir = plugin._store.dir
        self.tmp = tempfile.mkdtemp()
        plugin._store.dir = Path(self.tmp)

    def tearDown(self):
        plugin._store.dir = self.orig_dir
        shutil.rmtree(self.tmp)

    def test_small_result_unchanged(self):
        result = plugin._transform_result("terminal", {}, "small output")
        self.assertEqual(result, "small output")

    def test_large_result_replaced(self):
        big = "x" * 5000
        result = plugin._transform_result("terminal", {}, big)
        data = json.loads(result)
        self.assertEqual(data["type"], "handle")
        self.assertLess(len(result), len(big))

    def test_non_string_unchanged(self):
        result = plugin._transform_result("terminal", {}, {"not": "string"})
        self.assertEqual(result, {"not": "string"})

    def test_wrong_tool_unchanged(self):
        result = plugin._transform_result("web_search", {}, "x" * 5000)
        self.assertEqual(result, "x" * 5000)


class TestStorageCleanup(unittest.TestCase):
    def test_cleanup_old(self):
        tmp = tempfile.mkdtemp()
        store = plugin.ObsStore()
        store.dir = Path(tmp)
        store._cleanup_old()
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
