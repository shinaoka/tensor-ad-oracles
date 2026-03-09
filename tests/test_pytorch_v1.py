import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from generators import pytorch_v1


class PytorchV1RegistryTests(unittest.TestCase):
    def test_build_case_families_returns_expected_registry(self) -> None:
        registry = pytorch_v1.build_case_families()

        self.assertEqual(
            registry["svd"],
            ("u_abs", "s", "vh_abs", "uvh_product", "gauge_ill_defined"),
        )
        self.assertEqual(registry["solve"], ("identity",))
        self.assertEqual(registry["qr"], ("identity",))

    def test_ensure_runtime_dependencies_raises_clear_error_when_missing(self) -> None:
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name in {"torch", "expecttest"}:
                raise ModuleNotFoundError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "requires optional dependencies"):
                pytorch_v1.ensure_runtime_dependencies()

    def test_ensure_runtime_dependencies_mentions_expecttest(self) -> None:
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "expecttest":
                raise ModuleNotFoundError("No module named 'expecttest'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "expecttest"):
                pytorch_v1.ensure_runtime_dependencies()

    def test_ensure_runtime_dependencies_raises_clear_error_when_import_fails(self) -> None:
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("broken torch import")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "failed to import optional dependency"):
                pytorch_v1.ensure_runtime_dependencies()

    def test_build_case_spec_index_returns_expected_specs(self) -> None:
        index = pytorch_v1.build_case_spec_index()

        self.assertEqual(len(index), 11)
        self.assertEqual(
            index[("svd", "u_abs")].observable_kind,
            "svd_u_abs",
        )
        self.assertEqual(
            index[("eigh", "values_vectors_abs")].gradcheck_wrapper,
            "hermitian_input",
        )
        self.assertEqual(
            index[("qr", "identity")].observable_kind,
            "identity",
        )
        self.assertEqual(
            index[("svd", "gauge_ill_defined")].expected_behavior,
            "error",
        )

    def test_main_list_prints_case_registry(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = pytorch_v1.main(["--list"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("svd: u_abs, s, vh_abs, uvh_product, gauge_ill_defined", output)
        self.assertIn("qr: identity", output)
        self.assertIn("pinv_singular: identity", output)

    def test_main_materialize_solve_identity_writes_file(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = pytorch_v1.main(
                [
                    "--materialize",
                    "solve",
                    "--family",
                    "identity",
                    "--limit",
                    "1",
                    "--cases-root",
                    tmpdir,
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmpdir) / "solve" / "identity.jsonl").exists())

    def test_main_materialize_solve_identity_writes_all_nonempty_records(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = pytorch_v1.main(
                [
                    "--materialize",
                    "solve",
                    "--family",
                    "identity",
                    "--limit",
                    "24",
                    "--cases-root",
                    tmpdir,
                ]
            )

            self.assertEqual(exit_code, 0)
            lines = (Path(tmpdir) / "solve" / "identity.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 24)


if __name__ == "__main__":
    unittest.main()
