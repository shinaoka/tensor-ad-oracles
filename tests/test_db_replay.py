import unittest
from pathlib import Path

from validators import replay


class DbReplayTests(unittest.TestCase):
    def test_replay_solve_identity_case_matches_stored_references(self) -> None:
        result = replay.replay_case_file(
            Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/cases/solve/identity.jsonl"),
            limit=1,
        )

        self.assertEqual(result.checked, 1)
        self.assertEqual(result.failures, [])
