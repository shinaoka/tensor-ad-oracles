"""Shared JAX runtime helpers for v1 case generation."""

from __future__ import annotations

import numpy as np

PINNED_JAX_VERSION = "0.9.1"


def normalize_jax_version(version: str) -> str:
    """Strip local build metadata from a JAX version string."""
    return version.split("+", 1)[0]


def ensure_pinned_jax_version(jax, jaxlib) -> None:
    """Raise when the active JAX runtime does not match the repository pin."""
    actual_jax = normalize_jax_version(jax.__version__)
    actual_jaxlib = normalize_jax_version(jaxlib.__version__)
    if actual_jax != PINNED_JAX_VERSION or actual_jaxlib != PINNED_JAX_VERSION:
        raise RuntimeError(
            f"tensor-ad-oracles requires jax=={PINNED_JAX_VERSION} and jaxlib=={PINNED_JAX_VERSION}, "
            f"got jax=={jax.__version__} and jaxlib=={jaxlib.__version__}"
        )


def import_generation_runtime():
    import jax
    import jax.numpy as jnp
    import jaxlib

    jax.config.update("jax_enable_x64", True)
    ensure_pinned_jax_version(jax, jaxlib)
    return jax, jnp


def dtype_name(dtype) -> str:
    name = getattr(dtype, "name", None)
    if isinstance(name, str) and name:
        return name
    name = getattr(dtype, "__name__", None)
    if isinstance(name, str) and name:
        return name
    text = str(dtype)
    for prefix in ("<class 'jax.numpy.", "jax.numpy.", "numpy."):
        if text.startswith(prefix):
            return text.removeprefix(prefix).rstrip("'>")
    return text


def _raw_tensor_norm(jnp, tensor) -> float:
    if tensor.size == 0:
        return 0.0
    return float(jnp.linalg.norm(tensor).item())


def normalize_raw_tensor(jnp, tensor):
    array = jnp.asarray(tensor)
    norm = _raw_tensor_norm(jnp, array)
    if norm == 0.0:
        return array
    return array / norm


def normalize_raw_tensor_map(tensor_map: dict[str, object]) -> dict[str, object]:
    jax, jnp = import_generation_runtime()
    del jax
    return {name: normalize_raw_tensor(jnp, tensor) for name, tensor in tensor_map.items()}


def tensor_map_inner_product(left: dict[str, object], right: dict[str, object]):
    _, jnp = import_generation_runtime()
    left_items = list(left.items())
    if not left_items:
        return jnp.asarray(0.0)
    total = None
    for name, left_tensor in left_items:
        right_tensor = right[name]
        if left_tensor.size == 0:
            continue
        term = jnp.vdot(left_tensor.reshape(-1), right_tensor.reshape(-1))
        total = term if total is None else total + term
    if total is None:
        return jnp.asarray(0.0)
    return jnp.real(total)


def _flatten_real_data(value) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_real_data(item))
        return flattened
    return [float(value)]


def _flatten_complex_data(value) -> list[list[float]]:
    if isinstance(value, list):
        flattened: list[list[float]] = []
        for item in value:
            flattened.extend(_flatten_complex_data(item))
        return flattened
    return [[float(value.real), float(value.imag)]]


def encode_tensor(tensor) -> dict:
    _, jnp = import_generation_runtime()
    materialized = np.asarray(jnp.asarray(tensor))
    dtype = dtype_name(materialized.dtype)
    shape = list(materialized.shape)
    raw = materialized.tolist()
    data = (
        _flatten_complex_data(raw)
        if np.iscomplexobj(materialized)
        else _flatten_real_data(raw)
    )
    return {
        "dtype": dtype,
        "shape": shape,
        "order": "row_major",
        "data": data,
    }


def encode_tensor_map(tensors: dict[str, object]) -> dict[str, dict]:
    return {name: encode_tensor(tensor) for name, tensor in tensors.items()}


def compute_jax_jvp(observable_fn, inputs, direction):
    jax, _ = import_generation_runtime()
    _, tangent = jax.jvp(observable_fn, (inputs,), (direction,))
    return tangent


def compute_jax_vjp(observable_fn, inputs, cotangent):
    jax, _ = import_generation_runtime()
    _, pullback = jax.vjp(observable_fn, inputs)
    result = pullback(cotangent)
    if isinstance(result, tuple) and len(result) == 1:
        return result[0]
    return result


def compute_jax_linearization(raw_output_fn, inputs, direction):
    """Return the raw-output-space linearization and the cached linear map."""
    jax, _ = import_generation_runtime()
    _, linear_fn = jax.linearize(raw_output_fn, inputs)
    return linear_fn(direction), linear_fn


def compute_jax_transpose(linear_fn, inputs, cotangent):
    jax, _ = import_generation_runtime()
    result = jax.linear_transpose(linear_fn, inputs)(cotangent)
    if isinstance(result, tuple) and len(result) == 1:
        return result[0]
    return result


def compute_jax_adjoint_check(lhs, rhs) -> dict[str, float]:
    lhs_value = float(lhs.item() if hasattr(lhs, "item") else lhs)
    rhs_value = float(rhs.item() if hasattr(rhs, "item") else rhs)
    abs_err = abs(lhs_value - rhs_value)
    rel_denom = max(abs(lhs_value), abs(rhs_value), 1e-300)
    rel_err = abs_err / rel_denom
    return {
        "lhs": lhs_value,
        "rhs": rhs_value,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }
