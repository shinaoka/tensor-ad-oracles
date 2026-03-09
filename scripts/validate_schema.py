"""Schema validation helpers for tensor-ad-oracles."""

from __future__ import annotations


def require_jsonschema():
    """Return the jsonschema module or raise a clear runtime error."""
    try:
        import jsonschema
    except ModuleNotFoundError as exc:
        raise RuntimeError("jsonschema is required to validate tensor-ad-oracles cases") from exc
    return jsonschema
