import json
import tempfile
import unittest
from pathlib import Path

from generators import incremental_householder_qr, pytorch_v1
from validators.replay import replay_case_file


class IncrementalHouseholderQrGenerationTests(unittest.TestCase):
    def test_registry_contains_all_families_and_dtypes(self) -> None:
        index = pytorch_v1.build_case_spec_index()
        specs = [
            index[("incremental_householder_qr", family)]
            for family in incremental_householder_qr.FAMILIES
        ]

        self.assertEqual(
            tuple(spec.family for spec in specs),
            incremental_householder_qr.FAMILIES,
        )
        self.assertTrue(
            all(
                spec.supported_dtype_names == ("float64", "complex128", "float32", "complex64")
                for spec in specs
            )
        )
        self.assertIn(
            {"shape": (2, 4), "start": 1, "end": 2},
            incremental_householder_qr.sample_specs("selected_q_columns"),
        )

    def test_all_sample_shapes_are_full_rank(self) -> None:
        import torch

        for family in incremental_householder_qr.FAMILIES:
            for sample_spec in incremental_householder_qr.sample_specs(family):
                inputs = incremental_householder_qr.make_inputs(
                    torch,
                    family=family,
                    dtype=torch.complex128,
                    sample_spec=sample_spec,
                )
                kwargs = incremental_householder_qr.metadata(
                    family=family,
                    sample_spec=sample_spec,
                )
                output = incremental_householder_qr.observable(
                    torch,
                    family=family,
                    inputs=inputs,
                    op_kwargs=kwargs,
                )
                if "r" in output:
                    diagonal = torch.diagonal(output["r"])
                    self.assertTrue(torch.all(diagonal.real > 0))
                    self.assertTrue(torch.allclose(diagonal.imag, torch.zeros_like(diagonal.imag)))

    def test_factor_import_direction_stays_upper_trapezoidal(self) -> None:
        import torch

        direction = {
            "q": torch.ones((4, 2), dtype=torch.float64),
            "r": torch.ones((2, 4), dtype=torch.float64),
        }
        projected = incremental_householder_qr.project_direction(
            torch,
            family="from_factors_qr",
            direction=direction,
        )

        self.assertTrue(torch.equal(projected["r"], torch.triu(projected["r"])))
        self.assertTrue(torch.equal(projected["q"], direction["q"]))

    def test_materialized_families_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_root = Path(tmpdir)
            for family in incremental_householder_qr.FAMILIES:
                path = pytorch_v1.materialize_case_family(
                    "incremental_householder_qr",
                    family,
                    limit=1,
                    cases_root=cases_root,
                )
                records = [
                    json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                ]
                spec = pytorch_v1.build_case_spec_index()[("incremental_householder_qr", family)]
                self.assertEqual(len(records), len(spec.supported_dtype_names))
                self.assertTrue(
                    all(record["op_kwargs"]["rank_status"] == "full_rank" for record in records)
                )
                result = replay_case_file(path)
                self.assertEqual(result.failures, [])
                self.assertEqual(result.checked, len(records))


if __name__ == "__main__":
    unittest.main()
