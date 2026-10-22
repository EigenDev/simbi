# =============================================================================
# test_pre_commit.py
#
# exercises the symlink policy against isolated git indexes. fixtures include
# additions, type changes, renames, existing links, and link removal.
# usage: python -m unittest discover -s .githooks/tests -v
# =============================================================================

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HOOK = Path(__file__).resolve().parents[1] / "pre-commit"


class SymlinkPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="simbi-hook-test-")
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.env = os.environ.copy()
        for key in tuple(self.env):
            if key.startswith("GIT_"):
                del self.env[key]
        self.git("init", "-q")
        self.git("config", "user.name", "Hook Test")
        self.git("config", "user.email", "hook-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", str(self.repo / "unused-hooks"))

    def git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.repo, env=self.env, text=True,
            capture_output=True, check=True,
        ).stdout

    def hook(self):
        return subprocess.run(
            ["bash", str(HOOK)], cwd=self.repo, env=self.env, text=True,
            capture_output=True, check=False,
        )

    def seed_link(self):
        (self.repo / "science").symlink_to("external-science")
        self.git("add", "--", "science")
        self.git("commit", "-qm", "seed link for policy test")

    def test_regular_file_is_allowed(self):
        (self.repo / "source.txt").write_text("regular file\n")
        self.git("add", "--", "source.txt")
        self.assertEqual(self.hook().returncode, 0)

    def test_untracked_local_link_is_allowed(self):
        (self.repo / "science").symlink_to("external-science")
        self.assertEqual(self.hook().returncode, 0)

    def test_new_link_is_refused(self):
        (self.repo / "science").symlink_to("external-science")
        self.git("add", "--", "science")
        result = self.hook()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("science", result.stderr)

    def test_link_path_with_spaces_is_reported(self):
        (self.repo / "science link").symlink_to("external-science")
        self.git("add", "--", "science link")
        result = self.hook()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("science\\ link", result.stderr)

    def test_regular_file_replaced_by_link_is_refused(self):
        path = self.repo / "science"
        path.write_text("regular file\n")
        self.git("add", "--", "science")
        self.git("commit", "-qm", "seed regular file")
        path.unlink()
        path.symlink_to("external-science")
        self.git("add", "--", "science")
        self.assertEqual(self.git("diff", "--cached", "--name-status"), "T\tscience\n")
        self.assertNotEqual(self.hook().returncode, 0)

    def test_existing_link_is_refused_on_an_unrelated_commit(self):
        self.seed_link()
        (self.repo / "source.txt").write_text("regular file\n")
        self.git("add", "--", "source.txt")
        self.assertNotEqual(self.hook().returncode, 0)

    def test_renamed_link_is_refused(self):
        self.seed_link()
        self.git("mv", "science", "renamed-science")
        self.assertNotEqual(self.hook().returncode, 0)

    def test_removing_link_from_index_is_allowed_and_preserves_local_link(self):
        self.seed_link()
        self.git("rm", "--cached", "--", "science")
        self.assertEqual(self.hook().returncode, 0)
        self.assertTrue((self.repo / "science").is_symlink())


if __name__ == "__main__":
    unittest.main()
