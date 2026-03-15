"""Validation helpers for the complex-support ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from generators.pytorch_v1 import build_case_spec_index

from . import math_registry


COMPLEX_SUPPORT_PATH = Path("docs/math/complex-support.json")
_ALLOWED_NOTE_STATUS = {"reviewed", "pending", "not_required"}
_ALLOWED_DB_STATUS = {"covered", "pending", "unsupported"}


def load_complex_support(root: Path) -> dict:
    """Load the complex-support ledger from the repository root."""
    path = root / COMPLEX_SUPPORT_PATH
    if not path.exists():
        raise ValueError(f"complex support ledger not found: {COMPLEX_SUPPORT_PATH}")
    return json.loads(path.read_text(encoding="utf-8"))


def published_complex_dtype_index(cases_root: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    """Return published complex dtypes for every materialized `(op, family)` pair."""
    dtype_index: dict[tuple[str, str], set[str]] = {}
    if not cases_root.exists():
        return {}
    for path in sorted(cases_root.glob("*/*.jsonl")):
        key = (path.parent.name, path.stem)
        bucket = dtype_index.setdefault(key, set())
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            dtype_name = json.loads(line).get("dtype")
            if isinstance(dtype_name, str) and dtype_name.startswith("complex"):
                bucket.add(dtype_name)
    return {key: tuple(sorted(names)) for key, names in dtype_index.items()}


def _default_spec_index() -> dict[tuple[str, str], object]:
    return build_case_spec_index()


def _validate_note_target(root: Path, note_path: str, anchor: str) -> None:
    resolved_note_path = math_registry._resolve_note_path(root, note_path)
    if not resolved_note_path.exists():
        raise ValueError(f"reviewed note_path not found: {note_path}")
    anchors = math_registry.extract_markdown_anchors(
        resolved_note_path.read_text(encoding="utf-8")
    )
    if anchor not in anchors:
        raise ValueError(f"reviewed note anchor not found: {note_path}#{anchor}")


def validate_complex_support(
    root: Path,
    *,
    spec_index: Mapping[tuple[str, str], object] | None = None,
) -> None:
    """Validate the complex-support ledger against notes, registry, and case data."""
    root = root.resolve()
    ledger = load_complex_support(root)
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("complex support entries must be a list")

    expected_index = dict(spec_index or _default_spec_index())
    expected_keys = set(expected_index)
    registry = math_registry.load_registry(root)
    registry_index = {
        (entry["op"], entry["family"]): (entry["note_path"], entry["anchor"])
        for entry in registry.get("entries", [])
    }
    published_complex = published_complex_dtype_index(root / "cases")

    seen: set[tuple[str, str]] = set()
    for entry in entries:
        op = entry.get("op")
        family = entry.get("family")
        note = entry.get("note")
        db = entry.get("db")
        unsupported_reason = entry.get("unsupported_reason")

        if not isinstance(op, str) or not isinstance(family, str):
            raise ValueError(f"invalid complex support entry: {entry!r}")
        if not isinstance(note, dict) or not isinstance(db, dict):
            raise ValueError(f"invalid complex support entry: {entry!r}")

        key = (op, family)
        if key in seen:
            raise ValueError(f"duplicate complex support entry for {op}/{family}")
        if key not in expected_keys:
            raise ValueError(f"unexpected complex support entry for {op}/{family}")
        seen.add(key)

        note_status = note.get("status")
        note_path = note.get("path")
        note_anchor = note.get("anchor")
        if note_status not in _ALLOWED_NOTE_STATUS:
            raise ValueError(f"invalid note status for {op}/{family}: {note_status!r}")

        if note_status == "not_required":
            if note_path is not None or note_anchor is not None:
                raise ValueError(
                    f"not_required note must not declare path/anchor for {op}/{family}"
                )
        else:
            if not isinstance(note_path, str) or not note_path:
                raise ValueError(f"reviewed note target missing path for {op}/{family}")
            if not isinstance(note_anchor, str) or not note_anchor:
                raise ValueError(f"reviewed note target missing anchor for {op}/{family}")
            _validate_note_target(root, note_path, note_anchor)
            registry_target = registry_index.get(key)
            if registry_target is not None and registry_target != (note_path, note_anchor):
                raise ValueError(
                    f"reviewed note target disagrees with registry for {op}/{family}"
                )

        db_status = db.get("status")
        if db_status not in _ALLOWED_DB_STATUS:
            raise ValueError(f"invalid db status for {op}/{family}: {db_status!r}")

        expected_complex = {
            dtype_name
            for dtype_name in getattr(expected_index[key], "supported_dtype_names", ())
            if isinstance(dtype_name, str) and dtype_name.startswith("complex")
        }
        published = set(published_complex.get(key, ()))

        if db_status == "covered":
            missing = sorted(expected_complex - published)
            if missing:
                raise ValueError(
                    f"missing complex dtypes for {op}/{family}: {', '.join(missing)}"
                )
        elif db_status == "unsupported":
            if not isinstance(unsupported_reason, str) or not unsupported_reason.strip():
                raise ValueError(f"unsupported_reason required for {op}/{family}")
        else:
            if unsupported_reason not in (None, ""):
                raise ValueError(f"pending db entry must not set unsupported_reason for {op}/{family}")

        if db_status != "unsupported" and unsupported_reason not in (None, ""):
            raise ValueError(f"unsupported_reason only valid for unsupported db status: {op}/{family}")

    missing = sorted(expected_keys - seen)
    if missing:
        formatted = ", ".join(f"{op}/{family}" for op, family in missing)
        raise ValueError(f"missing complex support entries for: {formatted}")
