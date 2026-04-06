"""Replay the published JSON database and require zero failures."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators import jax_v1
from validators.replay import replay_case_tree


def _replay_jax_smoke_case_tree() -> object:
    with tempfile.TemporaryDirectory() as tmpdir:
        smoke_root = Path(tmpdir)
        jax_v1.materialize_case_family("abs", "identity", limit=1, cases_root=smoke_root)
        jax_v1.materialize_case_family("exp", "identity", limit=1, cases_root=smoke_root)
        return replay_case_tree(smoke_root)


def main() -> int:
    published = replay_case_tree(CASES_ROOT)
    if published.failures:
        joined = "\n".join(published.failures)
        raise SystemExit(f"replay failed:\n{joined}")

    jax_smoke = _replay_jax_smoke_case_tree()
    if jax_smoke.failures:
        joined = "\n".join(jax_smoke.failures)
        raise SystemExit(f"jax smoke replay failed:\n{joined}")

    print(f"replay_checked={published.checked}")
    print(f"jax_replay_checked={jax_smoke.checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
