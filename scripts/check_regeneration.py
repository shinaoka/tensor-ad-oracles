"""Regenerate the published case tree and require byte-for-byte equality."""

from __future__ import annotations

import filecmp
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.pytorch_v1 import materialize_all_case_families


def _relative_case_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*.jsonl")}


def compare_case_trees(expected_root: Path, actual_root: Path) -> None:
    """Raise when the two case trees differ in files or content."""
    expected_files = _relative_case_files(expected_root)
    actual_files = _relative_case_files(actual_root)
    if expected_files != actual_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        extra = sorted(str(path) for path in actual_files - expected_files)
        raise ValueError(f"file set mismatch: missing={missing}, extra={extra}")

    _, mismatch, errors = filecmp.cmpfiles(
        expected_root,
        actual_root,
        sorted(str(path) for path in expected_files),
        shallow=False,
    )
    if mismatch:
        raise ValueError(f"content mismatch for: {mismatch}")
    if errors:
        raise ValueError(f"comparison error for: {errors}")


def check_regeneration(cases_root: Path = CASES_ROOT) -> int:
    """Regenerate the full case tree and require equality with `cases/`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        regenerated_root = Path(tmpdir) / "cases"
        materialize_all_case_families(limit=None, cases_root=regenerated_root)
        compare_case_trees(cases_root, regenerated_root)
    return len(_relative_case_files(cases_root))


def main() -> int:
    compared = check_regeneration()
    print(f"regeneration_checked_files={compared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
