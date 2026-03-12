import json
import tempfile
import unittest
from pathlib import Path

from validators import math_registry


class MathRegistryTests(unittest.TestCase):
    def _make_repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmpdir = tempfile.TemporaryDirectory()
        root = Path(tmpdir.name)
        (root / "docs" / "math").mkdir(parents=True)
        (root / "cases").mkdir(parents=True)
        return tmpdir, root

    def _write_registry(self, root: Path, entries: list[dict]) -> None:
        (root / "docs" / "math" / "registry.json").write_text(
            json.dumps({"version": 1, "entries": entries}, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_extract_markdown_anchors_supports_explicit_ids(self) -> None:
        anchors = math_registry.extract_markdown_anchors(
            "\n".join(
                [
                    "<a id=\"family-identity\"></a>",
                    "### `identity`",
                    "",
                    "### `u_abs` {#family-u-abs}",
                ]
            )
        )

        self.assertEqual(anchors, {"family-identity", "family-u-abs"})

    def test_validate_registry_accepts_minimal_valid_repo(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "solve.md").write_text(
            "# Solve\n\n## DB Families\n\n<a id=\"family-identity\"></a>\n",
            encoding="utf-8",
        )
        solve_dir = root / "cases" / "solve"
        solve_dir.mkdir()
        (solve_dir / "identity.jsonl").write_text("{}\n", encoding="utf-8")
        self._write_registry(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note_path": "docs/math/solve.md",
                    "anchor": "family-identity",
                }
            ],
        )

        math_registry.validate_registry(root)

    def test_validate_registry_rejects_duplicate_entries(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "solve.md").write_text(
            "<a id=\"family-identity\"></a>\n",
            encoding="utf-8",
        )
        self._write_registry(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note_path": "docs/math/solve.md",
                    "anchor": "family-identity",
                },
                {
                    "op": "solve",
                    "family": "identity",
                    "note_path": "docs/math/solve.md",
                    "anchor": "family-identity",
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            math_registry.validate_registry(root)

    def test_validate_registry_rejects_missing_note_path(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        self._write_registry(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note_path": "docs/math/missing.md",
                    "anchor": "family-identity",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "note_path"):
            math_registry.validate_registry(root)

    def test_validate_registry_rejects_missing_anchor(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "svd.md").write_text(
            "# SVD\n\n## DB Families\n",
            encoding="utf-8",
        )
        self._write_registry(
            root,
            [
                {
                    "op": "svd",
                    "family": "u_abs",
                    "note_path": "docs/math/svd.md",
                    "anchor": "family-u-abs",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "missing anchor"):
            math_registry.validate_registry(root)

    def test_validate_registry_rejects_missing_case_coverage(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "svd.md").write_text(
            "<a id=\"family-u-abs\"></a>\n",
            encoding="utf-8",
        )
        svd_dir = root / "cases" / "svd"
        svd_dir.mkdir()
        (svd_dir / "u_abs.jsonl").write_text("{}\n", encoding="utf-8")
        self._write_registry(root, [])

        with self.assertRaisesRegex(ValueError, "missing registry entries"):
            math_registry.validate_registry(root)


if __name__ == "__main__":
    unittest.main()
