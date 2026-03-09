import unittest
from pathlib import Path

from validators import replay
from validators.case_loader import iter_case_files, load_case_file


class DbReplayTests(unittest.TestCase):
    def test_replay_solve_identity_case_matches_stored_references(self) -> None:
        result = replay.replay_case_file(
            Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/solve/identity.jsonl"),
            limit=1,
        )

        self.assertEqual(result.checked, 1)
        self.assertEqual(result.failures, [])

    def test_replay_one_case_from_each_success_family(self) -> None:
        case_paths = [
            Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/svd/u_abs.jsonl"),
            Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/svd/s.jsonl"),
            Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/svd/vh_abs.jsonl"),
            Path(
                "/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/svd/uvh_product.jsonl"
            ),
            Path(
                "/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/eigh/values_vectors_abs.jsonl"
            ),
            Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/solve/identity.jsonl"),
            Path(
                "/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/cholesky/identity.jsonl"
            ),
            Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/qr/identity.jsonl"),
            Path(
                "/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/pinv_singular/identity.jsonl"
            ),
        ]

        for case_path in case_paths:
            with self.subTest(case_path=f"{case_path.parent.name}/{case_path.name}"):
                result = replay.replay_case_file(case_path, limit=1)
                self.assertEqual(result.checked, 1)
                self.assertEqual(result.failures, [])

    def test_replay_gauge_ill_defined_cases_raise_expected_error(self) -> None:
        case_paths = [
            Path(
                "/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/svd/gauge_ill_defined.jsonl"
            ),
            Path(
                "/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/eigh/gauge_ill_defined.jsonl"
            ),
        ]

        for case_path in case_paths:
            with self.subTest(case_path=f"{case_path.parent.name}/{case_path.name}"):
                result = replay.replay_case_file(case_path, limit=1)
                self.assertEqual(result.checked, 1)
                self.assertEqual(result.failures, [])

    def test_replay_entire_published_case_tree(self) -> None:
        cases_root = Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases")
        total_records = sum(len(load_case_file(path)) for path in iter_case_files(cases_root))

        result = replay.replay_case_tree(cases_root)

        self.assertEqual(result.checked, total_records)
        self.assertEqual(result.failures, [])
