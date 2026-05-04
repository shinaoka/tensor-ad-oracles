"""Local structural AD reference helpers."""

from __future__ import annotations

from copy import deepcopy


STRUCTURAL_OPS = ("where", "cat", "narrow", "clamp")


def supported_dtype_names(op: str) -> tuple[str, ...]:
    if op == "clamp":
        return ("float64", "float32")
    if op in {"where", "cat", "narrow"}:
        return ("float64", "complex128", "float32", "complex64")
    raise ValueError(f"unsupported structural op: {op}")


def source_function(op: str) -> str:
    return {
        "where": "torch.where",
        "cat": "torch.cat",
        "narrow": "Tensor.narrow",
        "clamp": "torch.clamp",
    }[op]


def _tensor(torch, values, *, dtype):
    real = torch.tensor(values, dtype=torch.float64, device="cpu")
    if dtype.is_complex:
        imag = torch.flip(real, dims=tuple(range(real.ndim))) * 0.05 if real.ndim else real * 0.05
        return (real + 1j * imag).to(dtype=dtype)
    return real.to(dtype=dtype)


def _clone_case(case: dict) -> dict:
    return deepcopy(case)


def sample_cases(torch, op: str, *, dtype) -> list[dict[str, object]]:
    """Return deterministic differentiable inputs plus nondifferentiable metadata."""
    if op == "where":
        return [
            {
                "inputs": {
                    "x": _tensor(torch, [[-2.0, -1.0, 0.5], [1.0, 2.0, 3.0]], dtype=dtype),
                    "y": _tensor(torch, [[4.0, 5.0, 6.0], [-4.0, -5.0, -6.0]], dtype=dtype),
                },
                "op_kwargs": {
                    "condition": [[True, False, True], [False, True, False]],
                },
            },
            {
                "inputs": {
                    "x": _tensor(torch, 2.5, dtype=dtype),
                    "y": _tensor(torch, [[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=dtype),
                },
                "op_kwargs": {
                    "condition": [[True, False, False], [True, True, False]],
                },
            },
            {
                "inputs": {
                    "x": _tensor(torch, [[-3.0, -2.0, -1.0], [1.0, 2.0, 3.0]], dtype=dtype),
                    "y": _tensor(torch, -7.0, dtype=dtype),
                },
                "op_kwargs": {
                    "condition": [[False, True, False], [True, False, True]],
                },
            },
        ]

    if op == "cat":
        return [
            {
                "inputs": {
                    "a": _tensor(torch, [[1.0, 2.0, 3.0]], dtype=dtype),
                    "b": _tensor(torch, [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=dtype),
                },
                "op_kwargs": {"dim": 0},
            },
            {
                "inputs": {
                    "a": _tensor(torch, [[1.0], [2.0]], dtype=dtype),
                    "b": _tensor(torch, [[3.0, 4.0], [5.0, 6.0]], dtype=dtype),
                    "c": _tensor(torch, [[7.0], [8.0]], dtype=dtype),
                },
                "op_kwargs": {"dim": 1},
            },
            {
                "inputs": {
                    "a": _tensor(torch, [[[1.0], [2.0]]], dtype=dtype),
                    "b": _tensor(torch, [[[3.0, 4.0], [5.0, 6.0]]], dtype=dtype),
                },
                "op_kwargs": {"dim": -1},
            },
        ]

    if op == "narrow":
        return [
            {
                "inputs": {
                    "a": _tensor(
                        torch,
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                        dtype=dtype,
                    ),
                },
                "op_kwargs": {"dim": 0, "start": 1, "length": 2},
            },
            {
                "inputs": {
                    "a": _tensor(
                        torch,
                        [[-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]],
                        dtype=dtype,
                    ),
                },
                "op_kwargs": {"dim": 1, "start": 1, "length": 2},
            },
            {
                "inputs": {
                    "a": _tensor(
                        torch,
                        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                        dtype=dtype,
                    ),
                },
                "op_kwargs": {"dim": -1, "start": 0, "length": 1},
            },
        ]

    if op == "clamp":
        return [
            {
                "inputs": {
                    "x": _tensor(torch, [[-2.0, -0.25, 0.75], [1.25, 2.0, 3.0]], dtype=dtype),
                    "lo": _tensor(torch, -0.5, dtype=dtype),
                    "hi": _tensor(torch, 1.5, dtype=dtype),
                },
                "op_kwargs": {},
            },
            {
                "inputs": {
                    "x": _tensor(torch, [[-2.0, -0.25, 0.75], [1.25, 2.0, 3.0]], dtype=dtype),
                    "lo": _tensor(torch, [[-1.0, -0.5, 0.0], [0.5, 1.0, 1.5]], dtype=dtype),
                    "hi": _tensor(torch, [[0.0, 0.5, 1.0], [1.5, 2.5, 3.5]], dtype=dtype),
                },
                "op_kwargs": {},
            },
            {
                "inputs": {
                    "x": _tensor(torch, [[-3.0, -0.25, 0.75], [1.25, 2.0, 4.0]], dtype=dtype),
                    "lo": _tensor(torch, [[-1.0, -0.5, 0.0]], dtype=dtype),
                    "hi": _tensor(torch, [[1.0], [2.5]], dtype=dtype),
                },
                "op_kwargs": {},
            },
        ]

    raise ValueError(f"unsupported structural op: {op}")


def apply(torch, op: str, inputs: dict[str, object], op_kwargs: dict[str, object]):
    """Apply one structural op and return a single-output tensor map."""
    metadata = _clone_case(op_kwargs)
    if op == "where":
        condition = torch.tensor(
            metadata["condition"],
            dtype=torch.bool,
            device=inputs["x"].device,
        )
        return {"value": torch.where(condition, inputs["x"], inputs["y"])}

    if op == "cat":
        dim = int(metadata["dim"])
        return {"value": torch.cat([inputs[name] for name in inputs], dim=dim)}

    if op == "narrow":
        return {
            "value": inputs["a"].narrow(
                int(metadata["dim"]),
                int(metadata["start"]),
                int(metadata["length"]),
            )
        }

    if op == "clamp":
        return {"value": torch.clamp(inputs["x"], min=inputs["lo"], max=inputs["hi"])}

    raise ValueError(f"unsupported structural op: {op}")


def apply_tuple(
    torch,
    op: str,
    input_names: tuple[str, ...],
    op_kwargs: dict[str, object],
    *args,
):
    """Tuple-returning adapter for `torch.func` transforms."""
    inputs = dict(zip(input_names, args, strict=True))
    return (apply(torch, op, inputs, op_kwargs)["value"],)
