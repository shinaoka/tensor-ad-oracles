"""Validate the complex-support ledger against the repository tree."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.pytorch_v1 import build_case_spec_index
from validators.complex_support import validate_complex_support


def _default_spec_index() -> dict[tuple[str, str], object]:
    return build_case_spec_index()


def main() -> int:
    try:
        validate_complex_support(REPO_ROOT, spec_index=_default_spec_index())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("complex_support_ok=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
