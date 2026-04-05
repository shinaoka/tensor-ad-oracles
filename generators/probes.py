"""Probe construction and normalization helpers."""

from __future__ import annotations

import math


def _flat_square_sum(data) -> float:
    total = 0.0
    for item in data:
        if isinstance(item, list):
            if len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                total += float(item[0]) ** 2 + float(item[1]) ** 2
            else:
                total += _flat_square_sum(item)
        else:
            total += float(item) ** 2
    return total


def tensor_norm(tensor: dict) -> float:
    """Compute the Frobenius / Euclidean norm of an encoded tensor."""
    return math.sqrt(_flat_square_sum(tensor["data"]))


def _scale_data(data, scale: float):
    scaled = []
    for item in data:
        if isinstance(item, list):
            if len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                scaled.append([float(item[0]) * scale, float(item[1]) * scale])
            else:
                scaled.append(_scale_data(item, scale))
        else:
            scaled.append(float(item) * scale)
    return scaled


def normalize_tensor(tensor: dict) -> dict:
    """Return a unit-norm copy of an encoded tensor."""
    norm = tensor_norm(tensor)
    if norm == 0.0:
        return dict(tensor)
    normalized = dict(tensor)
    normalized["data"] = _scale_data(tensor["data"], 1.0 / norm)
    return normalized


def normalize_tensor_map(tensor_map: dict[str, dict]) -> dict[str, dict]:
    """Normalize each tensor in a named tensor map independently."""
    return {name: normalize_tensor(tensor) for name, tensor in tensor_map.items()}


def make_probe_record(
    *,
    probe_id: str,
    direction: dict[str, dict],
    cotangent: dict[str, dict],
    pytorch_jvp: dict[str, dict],
    pytorch_vjp: dict[str, dict],
    pytorch_hvp: dict[str, dict] | None = None,
    fd_step: float,
    fd_jvp: dict[str, dict],
    fd_hvp: dict[str, dict] | None = None,
    jax_jvp: dict[str, dict] | None = None,
    jax_vjp: dict[str, dict] | None = None,
    jax_linearization: dict[str, dict] | None = None,
    jax_raw_output_cotangent: dict[str, dict] | None = None,
    jax_transpose: dict[str, dict] | None = None,
    jax_adjoint_check: dict[str, float] | None = None,
    jax_provenance: dict[str, object] | None = None,
) -> dict:
    """Assemble one paired derivative probe record."""
    jax_values = {
        "jvp": jax_jvp,
        "vjp": jax_vjp,
        "linearization": jax_linearization,
        "raw_output_cotangent": jax_raw_output_cotangent,
        "transpose": jax_transpose,
        "adjoint_check": jax_adjoint_check,
        "provenance": jax_provenance,
    }
    provided_jax_fields = [name for name, value in jax_values.items() if value is not None]
    if provided_jax_fields and len(provided_jax_fields) != len(jax_values):
        missing = [name for name, value in jax_values.items() if value is None]
        raise ValueError(
            "jax_ref requires all JAX witness fields when any JAX payload is provided; "
            f"missing: {', '.join(missing)}"
        )

    pytorch_ref = {
        "jvp": pytorch_jvp,
        "vjp": pytorch_vjp,
    }
    if pytorch_hvp is not None:
        pytorch_ref["hvp"] = pytorch_hvp

    fd_ref = {
        "method": "central_difference",
        "stencil_order": 2,
        "step": fd_step,
        "jvp": fd_jvp,
    }
    if fd_hvp is not None:
        fd_ref["hvp"] = fd_hvp

    probe = {
        "probe_id": probe_id,
        "direction": direction,
        "cotangent": cotangent,
        "pytorch_ref": pytorch_ref,
        "fd_ref": fd_ref,
    }
    jax_ref = {}
    if provided_jax_fields:
        jax_ref = {
            "jvp": jax_jvp,
            "vjp": jax_vjp,
            "linearization": jax_linearization,
            "raw_output_cotangent": jax_raw_output_cotangent,
            "transpose": jax_transpose,
            "adjoint_check": jax_adjoint_check,
            "provenance": jax_provenance,
        }
        probe["jax_ref"] = jax_ref

    return probe
