#!/usr/bin/env python3
"""CPU tests for workspace scoring, path sandboxing, and fence-fallback reward."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl.reward import compute_score
from rl.tools import ReadFileTool, WriteFileTool, get_workspace, resolve_tool_path
from rl.workspace import make_rollout_workspace, score_workspace, workspace_is_complete


def _agent(ws: Path):
    return SimpleNamespace(extra_fields={"workspace_dir": str(ws)})


class WorkspaceRewardTests(unittest.TestCase):
    def test_full_workspace_scores_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "docker-compose.yml").write_text(
                "services:\n  web:\n    image: nginx\n", encoding="utf-8"
            )
            (ws / "index.html").write_text("<html></html>", encoding="utf-8")
            (ws / "main.py").write_text("print('ok')\n", encoding="utf-8")
            result = score_workspace(ws)
            self.assertEqual(result["score"], 1.0)
            self.assertTrue(result["has_compose"])
            self.assertTrue(result["compose_ok"])
            self.assertTrue(result["has_frontend"])
            self.assertTrue(result["has_backend"])

    def test_compose_only_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "docker-compose.yml").write_text(
                "services:\n  web:\n    image: nginx\n", encoding="utf-8"
            )
            self.assertFalse(workspace_is_complete(ws))
            (ws / "app.py").write_text("print(1)\n", encoding="utf-8")
            self.assertFalse(workspace_is_complete(ws))
            (ws / "templates").mkdir()
            (ws / "templates" / "index.html").write_text("<html></html>", encoding="utf-8")
            self.assertTrue(workspace_is_complete(ws))

    def test_empty_dir_scores_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = score_workspace(tmp)
            self.assertEqual(result["score"], 0.0)
            self.assertEqual(result["n_files"], 0)

    def test_eval_result_overrides_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "workspace"
            logs = root / "logs"
            ws.mkdir()
            logs.mkdir()
            (ws / "readme.txt").write_text("no compose", encoding="utf-8")
            (logs / "eval_result.json").write_text(
                '{"summary": {"combined_score_critical": 0.75}}', encoding="utf-8"
            )
            result = score_workspace(ws)
            self.assertEqual(result["score"], 0.75)
            self.assertEqual(result["combined_score_critical"], 0.75)

    def test_compute_score_fence_fallback(self):
        blob = (
            "```docker-compose.yml\nservices:\n  api:\n    image: python\n```\n"
            "```index.html\n<html></html>\n```\n"
            "```main.py\nprint(1)\n```\n"
        )
        score = compute_score("visapp", blob, None)
        self.assertEqual(score, 1.0)
        self.assertEqual(compute_score("visapp", "", None), 0.0)

    def test_compute_score_workspace_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "docker-compose.yml").write_text("services:\n  x: {}\n", encoding="utf-8")
            score = compute_score("visapp", "", None, extra_info={"workspace_dir": str(ws)})
            self.assertGreaterEqual(score, 0.6)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            with self.assertRaises(ValueError):
                resolve_tool_path(ws, "../etc/passwd", allow_dotdot=False)
            with self.assertRaises(ValueError):
                resolve_tool_path(ws, "/etc/passwd")
            with self.assertRaises(ValueError):
                resolve_tool_path(ws, "foo/../../etc/passwd", allow_dotdot=False)

    def test_write_read_file_tools(self):
        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp) / "workspace"
                ws.mkdir()
                tool_w = WriteFileTool.__new__(WriteFileTool)
                tool_r = ReadFileTool.__new__(ReadFileTool)
                agent = _agent(ws)
                resp, _, _ = await WriteFileTool.execute(
                    tool_w, "id", {"path": "hello.txt", "content": "hi"}, agent_data=agent
                )
                self.assertIn("wrote", resp.text)
                self.assertEqual((ws / "hello.txt").read_text(), "hi")
                bad, _, _ = await WriteFileTool.execute(
                    tool_w, "id", {"path": "../escape.txt", "content": "no"}, agent_data=agent
                )
                self.assertTrue(bad.text.startswith("Error:"))
                self.assertFalse((Path(tmp) / "escape.txt").exists())
                abs_bad, _, _ = await WriteFileTool.execute(
                    tool_w, "id", {"path": "/tmp/nope.txt", "content": "no"}, agent_data=agent
                )
                self.assertTrue(abs_bad.text.startswith("Error:"))
                ok, _, _ = await ReadFileTool.execute(
                    tool_r, "id", {"path": "hello.txt"}, agent_data=agent
                )
                self.assertEqual(ok.text, "hi")
                trav, _, _ = await ReadFileTool.execute(
                    tool_r, "id", {"path": "../../etc/passwd"}, agent_data=agent
                )
                self.assertTrue(trav.text.startswith("Error:"))

        asyncio.run(_run())

    def test_get_workspace_missing(self):
        with self.assertRaises(Exception):
            get_workspace(SimpleNamespace(extra_fields={}))

    def test_make_rollout_workspace_copies_inputs(self):
        vis = Path(__file__).resolve().parents[1]
        task = vis / "c4" / "1_newsletter"
        if not (task / "description.md").is_file():
            self.skipTest("c4/1_newsletter missing")
        root = make_rollout_workspace(task)
        try:
            self.assertTrue((root / "inputs" / "description.md").is_file())
            self.assertTrue((root / "workspace").is_dir())
            self.assertTrue(any((root / "inputs" / "pages").glob("*_structure-only.json")))
        finally:
            from rl.workspace import cleanup_workspace

            cleanup_workspace(root)


if __name__ == "__main__":
    unittest.main()
