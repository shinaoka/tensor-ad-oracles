import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from generators.pytorch_v1 import build_case_spec_index
from validators import complex_support


class ComplexSupportTests(unittest.TestCase):
    REPO_ROOT = Path(__file__).resolve().parents[1]

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

    def _write_ledger(self, root: Path, entries: list[dict]) -> None:
        (root / "docs" / "math" / "complex-support.json").write_text(
            json.dumps({"version": 1, "entries": entries}, indent=2) + "\n",
            encoding="utf-8",
        )

    def _spec_index(self, *rows: tuple[str, str, tuple[str, ...]]) -> dict[tuple[str, str], object]:
        return {
            (op, family): SimpleNamespace(
                op=op,
                family=family,
                supported_dtype_names=dtypes,
            )
            for op, family, dtypes in rows
        }

    def test_validate_complex_support_accepts_minimal_valid_repo(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "solve.md").write_text(
            "# Solve\n\n<a id=\"family-identity\"></a>\n",
            encoding="utf-8",
        )
        (root / "cases" / "solve").mkdir()
        (root / "cases" / "solve" / "identity.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "case_id": "solve_complex128_identity_001",
                            "dtype": "complex128",
                        }
                    ),
                    json.dumps(
                        {
                            "case_id": "solve_complex64_identity_002",
                            "dtype": "complex64",
                        }
                    ),
                ]
            )
            + "\n",
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
                }
            ],
        )
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {
                        "path": "docs/math/solve.md",
                        "anchor": "family-identity",
                        "status": "reviewed",
                    },
                    "db": {"status": "covered"},
                    "unsupported_reason": None,
                }
            ],
        )

        complex_support.validate_complex_support(
            root,
            spec_index=self._spec_index(
                ("solve", "identity", ("float64", "complex128", "complex64"))
            ),
        )

    def test_validate_complex_support_rejects_duplicate_entries(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        self._write_registry(root, [])
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {"path": None, "anchor": None, "status": "not_required"},
                    "db": {"status": "unsupported"},
                    "unsupported_reason": "out of scope",
                },
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {"path": None, "anchor": None, "status": "not_required"},
                    "db": {"status": "unsupported"},
                    "unsupported_reason": "out of scope",
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            complex_support.validate_complex_support(
                root,
                spec_index=self._spec_index(("solve", "identity", ("float64",))),
            )

    def test_validate_complex_support_rejects_reviewed_entry_without_note_target(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        self._write_registry(root, [])
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {"path": None, "anchor": None, "status": "reviewed"},
                    "db": {"status": "unsupported"},
                    "unsupported_reason": "out of scope",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "reviewed note"):
            complex_support.validate_complex_support(
                root,
                spec_index=self._spec_index(("solve", "identity", ("float64",))),
            )

    def test_validate_complex_support_rejects_not_required_entry_with_note_target(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        self._write_registry(root, [])
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {
                        "path": "docs/math/solve.md",
                        "anchor": "family-identity",
                        "status": "not_required",
                    },
                    "db": {"status": "unsupported"},
                    "unsupported_reason": "out of scope",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "not_required"):
            complex_support.validate_complex_support(
                root,
                spec_index=self._spec_index(("solve", "identity", ("float64",))),
            )

    def test_validate_complex_support_rejects_unsupported_entry_without_reason(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        self._write_registry(root, [])
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {"path": None, "anchor": None, "status": "not_required"},
                    "db": {"status": "unsupported"},
                    "unsupported_reason": None,
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "unsupported_reason"):
            complex_support.validate_complex_support(
                root,
                spec_index=self._spec_index(("solve", "identity", ("float64",))),
            )

    def test_validate_complex_support_rejects_unsupported_entry_when_complex_support_exists(
        self,
    ) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "solve.md").write_text(
            "# Solve\n\n<a id=\"family-identity\"></a>\n",
            encoding="utf-8",
        )
        (root / "cases" / "solve").mkdir()
        (root / "cases" / "solve" / "identity.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "solve_c128_identity_001", "dtype": "complex128"}),
                    json.dumps({"case_id": "solve_c64_identity_002", "dtype": "complex64"}),
                ]
            )
            + "\n",
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
                }
            ],
        )
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {"path": None, "anchor": None, "status": "not_required"},
                    "db": {"status": "unsupported"},
                    "unsupported_reason": "stale",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "unsupported.*complex support"):
            complex_support.validate_complex_support(
                root,
                spec_index=self._spec_index(
                    ("solve", "identity", ("float64", "complex128", "complex64"))
                ),
            )

    def test_validate_complex_support_rejects_covered_entry_missing_complex_dtype(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "solve.md").write_text(
            "# Solve\n\n<a id=\"family-identity\"></a>\n",
            encoding="utf-8",
        )
        (root / "cases" / "solve").mkdir()
        (root / "cases" / "solve" / "identity.jsonl").write_text(
            json.dumps({"case_id": "solve_c128_identity_001", "dtype": "complex128"}) + "\n",
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
                }
            ],
        )
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {
                        "path": "docs/math/solve.md",
                        "anchor": "family-identity",
                        "status": "reviewed",
                    },
                    "db": {"status": "covered"},
                    "unsupported_reason": None,
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "missing complex dtypes"):
            complex_support.validate_complex_support(
                root,
                spec_index=self._spec_index(
                    ("solve", "identity", ("float64", "complex128", "complex64"))
                ),
            )

    def test_validate_complex_support_rejects_pending_note_without_existing_target(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "cases" / "solve").mkdir()
        (root / "cases" / "solve" / "identity.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "solve_c128_identity_001", "dtype": "complex128"}),
                    json.dumps({"case_id": "solve_c64_identity_002", "dtype": "complex64"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_registry(root, [])
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {"path": None, "anchor": None, "status": "pending"},
                    "db": {"status": "covered"},
                    "unsupported_reason": None,
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "missing path"):
            complex_support.validate_complex_support(
                root,
                spec_index=self._spec_index(
                    ("solve", "identity", ("float64", "complex128", "complex64"))
                ),
            )

    def test_validate_complex_support_rejects_pending_note_when_db_is_covered(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "solve.md").write_text(
            "# Solve\n\n<a id=\"family-identity\"></a>\n",
            encoding="utf-8",
        )
        (root / "cases" / "solve").mkdir()
        (root / "cases" / "solve" / "identity.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "solve_c128_identity_001", "dtype": "complex128"}),
                    json.dumps({"case_id": "solve_c64_identity_002", "dtype": "complex64"}),
                ]
            )
            + "\n",
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
                }
            ],
        )
        self._write_ledger(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note": {
                        "path": "docs/math/solve.md",
                        "anchor": "family-identity",
                        "status": "pending",
                    },
                    "db": {"status": "covered"},
                    "unsupported_reason": None,
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "covered db entry requires completed note"):
            complex_support.validate_complex_support(
                root,
                spec_index=self._spec_index(
                    ("solve", "identity", ("float64", "complex128", "complex64"))
                ),
            )

    def test_repo_complex_support_ledger_validates_against_current_surface(self) -> None:
        complex_support.validate_complex_support(self.REPO_ROOT)

    def test_repo_complex_support_ledger_covers_every_case_family(self) -> None:
        ledger = complex_support.load_complex_support(self.REPO_ROOT)
        entry_keys = {(entry["op"], entry["family"]) for entry in ledger["entries"]}

        self.assertEqual(entry_keys, set(build_case_spec_index()))

    def test_repo_complex_support_ledger_marks_representative_ready_families(self) -> None:
        ledger = {
            (entry["op"], entry["family"]): entry
            for entry in complex_support.load_complex_support(self.REPO_ROOT)["entries"]
        }

        for key in (
            ("svd", "u_abs"),
            ("solve", "identity"),
            ("eig", "values_vectors_abs"),
            ("sum", "identity"),
        ):
            self.assertEqual(ledger[key]["note"]["status"], "reviewed")
            self.assertEqual(ledger[key]["db"]["status"], "covered")

    def test_repo_complex_support_ledger_marks_representative_db_only_families(self) -> None:
        ledger = {
            (entry["op"], entry["family"]): entry
            for entry in complex_support.load_complex_support(self.REPO_ROOT)["entries"]
        }

        for key in (
            ("lu_factor_ex", "identity"),
            ("solve_ex", "identity"),
            ("inv_ex", "identity"),
        ):
            self.assertEqual(ledger[key]["note"]["status"], "not_required")

    def test_repo_complex_support_ledger_marks_representative_float_only_families_unsupported(
        self,
    ) -> None:
        ledger = {
            (entry["op"], entry["family"]): entry
            for entry in complex_support.load_complex_support(self.REPO_ROOT)["entries"]
        }

        for key in (
            ("atan2", "identity"),
            ("copysign", "identity"),
            ("special_ndtr", "identity"),
        ):
            self.assertEqual(ledger[key]["db"]["status"], "unsupported")
            self.assertTrue(ledger[key]["unsupported_reason"])


if __name__ == "__main__":
    unittest.main()
