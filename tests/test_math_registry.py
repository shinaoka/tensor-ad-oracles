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

    def test_repo_contains_core_linalg_math_notes(self) -> None:
        note_dir = Path(__file__).resolve().parents[1] / "docs" / "math"
        expected = {
            "svd.md",
            "solve.md",
            "qr.md",
            "lu.md",
            "cholesky.md",
            "inv.md",
            "det.md",
            "eig.md",
            "eigen.md",
            "pinv.md",
            "lstsq.md",
            "norm.md",
        }

        self.assertTrue(expected.issubset({path.name for path in note_dir.glob("*.md")}))

    def test_repo_svd_note_exposes_family_anchors(self) -> None:
        note_path = Path(__file__).resolve().parents[1] / "docs" / "math" / "svd.md"
        text = note_path.read_text(encoding="utf-8")
        anchors = math_registry.extract_markdown_anchors(text)

        self.assertIn("## DB Families", text)
        self.assertEqual(
            {"family-u-abs", "family-s", "family-vh-abs", "family-uvh-product"} - anchors,
            set(),
        )

    def test_repo_eig_and_eigen_notes_are_distinct(self) -> None:
        note_dir = Path(__file__).resolve().parents[1] / "docs" / "math"
        eig_text = (note_dir / "eig.md").read_text(encoding="utf-8")
        eigen_text = (note_dir / "eigen.md").read_text(encoding="utf-8")

        self.assertIn("General", eig_text)
        self.assertIn("Hermitian", eigen_text)

    def test_repo_contains_remaining_known_rule_notes(self) -> None:
        note_dir = Path(__file__).resolve().parents[1] / "docs" / "math"
        expected = {"matrix_exp.md", "scalar_ops.md", "dyadtensor_reverse.md"}

        self.assertTrue(expected.issubset({path.name for path in note_dir.glob("*.md")}))

    def test_repo_scalar_ops_note_exposes_representative_op_anchors(self) -> None:
        note_path = Path(__file__).resolve().parents[1] / "docs" / "math" / "scalar_ops.md"
        anchors = math_registry.extract_markdown_anchors(note_path.read_text(encoding="utf-8"))

        self.assertEqual({"op-abs", "op-add", "op-sum", "op-var"} - anchors, set())

    def test_repo_matrix_exp_note_marks_db_status(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "matrix_exp.md"
        ).read_text(encoding="utf-8")

        self.assertIn("not yet materialized", text)


if __name__ == "__main__":
    unittest.main()
