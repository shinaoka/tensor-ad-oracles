import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from generators import jax_v1, pytorch_v1


class JaxV1Tests(unittest.TestCase):
    def test_build_case_families_matches_pytorch_registry(self) -> None:
        self.assertEqual(jax_v1.build_case_families(), pytorch_v1.build_case_families())

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

    def test_main_materialize_abs_identity_writes_jax_ref_case(self) -> None:
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp  # noqa: F401
        except Exception as exc:
            self.skipTest(f"jax runtime unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with redirect_stdout(io.StringIO()):
                exit_code = jax_v1.main(
                    [
                        "--materialize",
                        "abs",
                        "--family",
                        "identity",
                        "--limit",
                        "1",
                        "--cases-root",
                        tmpdir,
                    ]
                )

            self.assertEqual(exit_code, 0)
            out_path = Path(tmpdir) / "abs" / "identity.jsonl"
            self.assertTrue(out_path.exists())
            record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("jax_ref", record["probes"][0])
            self.assertEqual(record["probes"][0]["jax_ref"]["provenance"]["witness_source"], "jax_test")


if __name__ == "__main__":
    unittest.main()
