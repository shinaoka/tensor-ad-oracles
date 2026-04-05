import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from generators import jax_v1, pytorch_v1
from validators.encoding import decode_tensor_map


REPO_ROOT = Path(__file__).resolve().parents[1]


class JaxV1Tests(unittest.TestCase):
    def test_build_case_families_matches_pytorch_registry(self) -> None:
        self.assertEqual(jax_v1.build_case_families(), pytorch_v1.build_case_families())

    def test_select_witness_source_prefers_jax_test_for_unary_elementwise_families(self) -> None:
        self.assertEqual(jax_v1.select_witness_source("abs", "identity"), "jax_test")
        self.assertEqual(
            jax_v1.select_witness_source("abs", "identity", prefer_jax_test=False),
            "torch_aligned",
        )

    def test_build_jax_ref_provenance_preserves_witness_source(self) -> None:
        provenance = jax_v1.build_jax_ref_provenance(
            source_commit="deadbeef",
            seed=17,
            jax_version="0.9.1",
            jaxlib_version="0.9.1",
            witness_source="jax_test",
        )

        self.assertEqual(provenance["source_backend"], "jax")
        self.assertEqual(provenance["witness_source"], "jax_test")
        self.assertEqual(provenance["jax_version"], "0.9.1")
        self.assertEqual(provenance["jaxlib_version"], "0.9.1")

    def test_main_materialize_smoke_witnesses_normalize_computation_and_serialization(self) -> None:
        try:
            import jax.numpy as jnp  # noqa: F401
        except Exception as exc:
            self.skipTest(f"jax runtime unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with redirect_stdout(io.StringIO()):
                exit_code = jax_v1.main(
                    ["--materialize-all", "--limit", "1", "--cases-root", tmpdir]
                )

            self.assertEqual(exit_code, 0)
            for op, expected_linearization in (
                ("abs", 1.0),
                ("exp", math.e),
            ):
                out_path = Path(tmpdir) / op / "identity.jsonl"
                self.assertTrue(out_path.exists())
                record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
                probe = record["probes"][0]

                direction = decode_tensor_map(probe["direction"])
                cotangent = decode_tensor_map(probe["cotangent"])
                jvp = decode_tensor_map(probe["jax_ref"]["jvp"])
                vjp = decode_tensor_map(probe["jax_ref"]["vjp"])
                linearization = decode_tensor_map(probe["jax_ref"]["linearization"])
                transpose = decode_tensor_map(probe["jax_ref"]["transpose"])
                pytorch_jvp = decode_tensor_map(probe["pytorch_ref"]["jvp"])
                fd_jvp = decode_tensor_map(probe["fd_ref"]["jvp"])

                self.assertEqual(direction["x"].tolist(), [1.0])
                self.assertEqual(cotangent["value"].tolist(), [1.0])
                self.assertAlmostEqual(float(linearization["value"].item()), expected_linearization, places=12)
                self.assertAlmostEqual(float(jvp["value"].item()), expected_linearization, places=12)
                self.assertAlmostEqual(float(pytorch_jvp["value"].item()), expected_linearization, places=8)
                self.assertAlmostEqual(float(fd_jvp["value"].item()), expected_linearization, places=6)
                self.assertAlmostEqual(float(vjp["x"].item()), float(transpose["x"].item()), places=12)

                lhs = float((cotangent["value"] * jvp["value"]).sum().item())
                rhs = float((transpose["x"] * direction["x"]).sum().item())
                self.assertAlmostEqual(lhs, rhs, places=12)
                self.assertEqual(probe["jax_ref"]["provenance"]["witness_source"], "jax_test")
                self.assertEqual(
                    record["provenance"]["source_file"],
                    "jax/_src/internal_test_util/test_harnesses.py",
                )
                self.assertEqual(
                    record["provenance"]["source_function"],
                    "_make_unary_elementwise_harness",
                )
                self.assertEqual(record["provenance"]["seed"], 17)
                self.assertIn("harness_fullname=", record["provenance"]["comment"])

    def test_torch_aligned_materialization_preserves_published_inputs(self) -> None:
        source_path = REPO_ROOT / "cases" / "abs" / "identity.jsonl"
        source_case = json.loads(source_path.read_text(encoding="utf-8").splitlines()[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = jax_v1.materialize_torch_aligned_case_family(
                "abs",
                "identity",
                cases_root=Path(tmpdir),
            )

            generated = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(generated["inputs"], source_case["inputs"])
        self.assertEqual(
            generated["probes"][0]["jax_ref"]["provenance"]["witness_source"],
            "torch_aligned",
        )
        self.assertEqual(generated["provenance"]["source_file"], source_case["provenance"]["source_file"])
        self.assertEqual(
            generated["provenance"]["source_function"],
            source_case["provenance"]["source_function"],
        )


if __name__ == "__main__":
    unittest.main()
