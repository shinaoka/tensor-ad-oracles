import json
import unittest
from pathlib import Path

from generators import pytorch_v1


class StructuralGenerationTests(unittest.TestCase):
    def test_generate_where_records_cover_metadata_and_broadcasting(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        records = pytorch_v1.generate_structural_identity_records("where", limit=None)
        spec = pytorch_v1.build_case_spec_index()[("where", "identity")]

        self.assertEqual(len(records), 3 * len(spec.supported_dtype_names))
        self.assertEqual(
            {
                tuple(record["inputs"]["x"]["shape"])
                for record in records
                if record["dtype"] == "float64"
            },
            {(2, 3), (), (2, 3)},
        )
        record = records[0]
        self.assertEqual(record["op"], "where")
        self.assertIn("condition", record["op_kwargs"])
        self.assertNotIn("condition", record["inputs"])
        self.assertIn("hvp", record["probes"][0]["pytorch_ref"])

    def test_generate_cat_and_narrow_records_preserve_axis_metadata(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        cat_records = pytorch_v1.generate_structural_identity_records("cat", limit=None)
        narrow_records = pytorch_v1.generate_structural_identity_records("narrow", limit=None)

        self.assertEqual(
            {
                record["op_kwargs"]["dim"]
                for record in cat_records
                if record["dtype"] == "float64"
            },
            {0, 1, -1},
        )
        self.assertTrue(
            all(
                {"start", "length", "dim"}.issubset(record["op_kwargs"])
                for record in narrow_records
            )
        )
        self.assertTrue(
            all(
                record["probes"][0]["pytorch_ref"]["vjp"]["a"]["shape"]
                == record["inputs"]["a"]["shape"]
                for record in narrow_records
            )
        )

    def test_generate_clamp_records_cover_scalar_and_tensor_bounds(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        records = pytorch_v1.generate_structural_identity_records("clamp", limit=None)
        spec = pytorch_v1.build_case_spec_index()[("clamp", "identity")]

        self.assertEqual(len(records), 3 * len(spec.supported_dtype_names))
        self.assertEqual({record["dtype"] for record in records}, set(spec.supported_dtype_names))
        self.assertIn((), {tuple(record["inputs"]["lo"]["shape"]) for record in records})
        self.assertIn((2, 3), {tuple(record["inputs"]["hi"]["shape"]) for record in records})
        self.assertTrue(all(record["op"] == "clamp" for record in records))

    def test_materialize_structural_family_writes_jsonl(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = pytorch_v1.materialize_case_family(
                "where",
                "identity",
                limit=1,
                cases_root=Path(tmpdir),
            )

            self.assertEqual(out_path, Path(tmpdir) / "where" / "identity.jsonl")
            records = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                len(records),
                len(pytorch_v1.build_case_spec_index()[("where", "identity")].supported_dtype_names),
            )


if __name__ == "__main__":
    unittest.main()
