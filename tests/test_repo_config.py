import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles")


class RepoConfigTests(unittest.TestCase):
    def test_python_version_file_is_present(self) -> None:
        python_version = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()
        self.assertTrue(python_version)

    def test_python_version_file_is_patch_pinned(self) -> None:
        python_version = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()
        self.assertRegex(python_version, r"^\d+\.\d+\.\d+$")

    def test_pyproject_declares_uv_managed_non_package_project(self) -> None:
        config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(config["project"]["requires-python"], ">=3.12")
        self.assertEqual(config["tool"]["uv"]["package"], False)
        self.assertIn("dev", config["dependency-groups"])

    def test_readme_documents_uv_sync_and_uv_run(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("uv sync --locked --all-groups", readme)
        self.assertIn("uv run python -m unittest", readme)
        self.assertIn("uv run python -m generators.pytorch_v1 --list", readme)

    def test_uv_lock_is_checked_in(self) -> None:
        self.assertTrue((REPO_ROOT / "uv.lock").exists())

    def test_gitignore_ignores_uv_virtualenv(self) -> None:
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".venv/", gitignore)
        self.assertIn(".venv31212/", gitignore)

    def test_agents_documents_uv_interpreter_and_expecttest_rules(self) -> None:
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("uv sync --locked --all-groups", agents)
        self.assertIn("3.12.12", agents)
        self.assertIn("expecttest", agents)


if __name__ == "__main__":
    unittest.main()
