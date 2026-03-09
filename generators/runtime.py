"""Shared PyTorch runtime helpers for v1 case generation."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from types import SimpleNamespace

from . import observables


def import_generation_runtime():
    import torch
    from torch.testing._internal.opinfo.definitions import linalg

    return torch, linalg


def dtype_name(torch, dtype) -> str:
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.complex128:
        return "complex128"
    raise ValueError(f"unsupported torch dtype for v1 generation: {dtype}")


def raw_tensor_norm(torch, tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(tensor).item())


def normalize_raw_tensor(torch, tensor):
    norm = raw_tensor_norm(torch, tensor)
    if norm == 0.0:
        return tensor.clone()
    return tensor / norm


def normalize_raw_tensor_map(torch, tensor_map: dict[str, object]) -> dict[str, object]:
    return {name: normalize_raw_tensor(torch, tensor) for name, tensor in tensor_map.items()}


def combined_input_norm(torch, tensor_map: dict[str, object]) -> float:
    square_sum = 0.0
    for tensor in tensor_map.values():
        norm = raw_tensor_norm(torch, tensor)
        square_sum += norm * norm
    return math.sqrt(square_sum)


def randn_like(torch, tensor, *, generator):
    return torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    )


def tensor_map_inner_product(torch, left: dict[str, object], right: dict[str, object]):
    left_items = list(left.items())
    if not left_items:
        return torch.tensor(0.0, dtype=torch.float64)
    total = None
    for name, left_tensor in left_items:
        right_tensor = right[name]
        if left_tensor.numel() == 0:
            continue
        term = torch.vdot(left_tensor.reshape(-1), right_tensor.reshape(-1))
        total = term if total is None else total + term
    if total is not None:
        return total
    first = left_items[0][1]
    return torch.zeros((), dtype=first.dtype, device=first.device)


def sample_inputs_for_spec(torch, linalg, spec, *, seed: int):
    rng_state = torch.random.get_rng_state()
    py_state = random.getstate()
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        if spec.op == "svd":
            return list(
                linalg.sample_inputs_svd(
                    SimpleNamespace(name="linalg.svd"),
                    "cpu",
                    torch.float64,
                    requires_grad=True,
                )
            )
        if spec.op == "eigh":
            return list(
                linalg.sample_inputs_linalg_eigh(
                    None,
                    "cpu",
                    torch.float64,
                    requires_grad=True,
                )
            )
        if spec.op == "solve":
            return list(
                linalg.sample_inputs_linalg_solve(
                    None,
                    "cpu",
                    torch.float64,
                    requires_grad=True,
                )
            )
        if spec.op == "cholesky":
            return list(
                linalg.sample_inputs_linalg_cholesky(
                    None,
                    "cpu",
                    torch.float64,
                    requires_grad=True,
                )
            )
        if spec.op == "qr":
            return list(
                linalg.sample_inputs_linalg_qr_geqrf(
                    None,
                    "cpu",
                    torch.float64,
                    requires_grad=True,
                )
            )
        if spec.op == "pinv_singular":
            return list(
                linalg.sample_inputs_linalg_pinv_singular(
                    None,
                    "cpu",
                    torch.float64,
                    requires_grad=True,
                )
            )
    finally:
        torch.random.set_rng_state(rng_state)
        random.setstate(py_state)
    raise ValueError(f"unsupported sample-input op: {spec.op}")


def tensor_map_to_tuple(tensor_map: dict[str, object]) -> tuple[object, ...]:
    return tuple(tensor_map.values())


def tuple_to_tensor_map(keys: Iterable[str], values) -> dict[str, object]:
    if isinstance(values, tuple):
        values_tuple = values
    else:
        values_tuple = (values,)
    return dict(zip(keys, values_tuple, strict=True))


def apply_spec_observable(torch, spec, sample, inputs: dict[str, object]) -> dict[str, object]:
    if spec.op == "svd":
        result = torch.linalg.svd(inputs["a"], **sample.kwargs)
    elif spec.op == "eigh":
        result = torch.linalg.eigh(inputs["a"], **sample.kwargs)
    elif spec.op == "solve":
        result = torch.linalg.solve(inputs["a"], inputs["b"], **sample.kwargs)
    elif spec.op == "cholesky":
        result = torch.linalg.cholesky(inputs["a"], **sample.kwargs)
    elif spec.op == "qr":
        result = torch.linalg.qr(inputs["a"], **sample.kwargs)
    elif spec.op == "pinv_singular":
        result = torch.linalg.pinv(inputs["a"] @ inputs["b"].mT, **sample.kwargs)
    else:
        raise ValueError(f"unsupported op for observable materialization: {spec.op}")
    return observables.apply_observable(spec.observable_kind, result)


def zeros_like_input_map(torch, inputs: dict[str, object], grads) -> dict[str, object]:
    out: dict[str, object] = {}
    for (name, tensor), grad in zip(inputs.items(), grads, strict=True):
        out[name] = torch.zeros_like(tensor) if grad is None else grad
    return out


def map_allclose(torch, expected: dict[str, object], actual: dict[str, object], *, rtol: float, atol: float) -> bool:
    if expected.keys() != actual.keys():
        return False
    return all(
        torch.allclose(expected[name], actual[name], rtol=rtol, atol=atol)
        for name in expected
    )


def structured_input_tensor(torch, spec, tensor):
    cloned = tensor.detach().clone()
    if spec.op == "eigh":
        cloned = cloned + cloned.mH
    return cloned.requires_grad_(True)


def structured_direction_tensor(torch, spec, tensor, *, generator):
    direction = randn_like(torch, tensor, generator=generator)
    if spec.op in {"eigh", "cholesky"} and tensor.ndim >= 2:
        direction = direction + direction.mH
    return direction


def build_input_map(torch, spec, sample) -> dict[str, object]:
    if spec.op in {"svd", "eigh", "cholesky", "qr"}:
        return {"a": structured_input_tensor(torch, spec, sample.input)}
    if spec.op in {"solve", "pinv_singular"}:
        return {
            "a": structured_input_tensor(torch, spec, sample.input),
            "b": structured_input_tensor(torch, spec, sample.args[0]),
        }
    raise ValueError(f"unsupported op for input materialization: {spec.op}")


def build_direction_map(torch, spec, inputs: dict[str, object], *, generator) -> dict[str, object]:
    return normalize_raw_tensor_map(
        torch,
        {
            name: structured_direction_tensor(torch, spec, tensor, generator=generator)
            for name, tensor in inputs.items()
        },
    )


def build_observable_function(torch, spec, sample, input_names: tuple[str, ...]) -> Callable[..., tuple[object, ...]]:
    def observable_fn(*args):
        inputs = dict(zip(input_names, args, strict=True))
        output = apply_spec_observable(torch, spec, sample, inputs)
        return tensor_map_to_tuple(output)

    return observable_fn
