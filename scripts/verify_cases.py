"""Helpers for checking case-file hygiene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl_records(path: Path) -> list[dict]:
    """Load all JSON objects from a JSONL file."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def find_duplicate_case_ids(paths: Iterable[Path]) -> None:
    """Raise when the same case id appears in more than one record."""
    seen: dict[str, Path] = {}
    for path in paths:
        for record in load_jsonl_records(path):
            case_id = record["case_id"]
            previous = seen.get(case_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate case_id {case_id!r} found in {previous} and {path}"
                )
            seen[case_id] = path
