"""Replay published JSON cases and verify stored references are reproducible."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from generators.pytorch_v1 import build_case_spec_index
from generators.runtime import (
    apply_spec_observable,
    map_allclose,
    tensor_map_to_tuple,
    tuple_to_tensor_map,
    zeros_like_input_map,
)

from .case_loader import load_case_file
from .encoding import decode_tensor_map


@dataclass
class ReplayResult:
    checked: int = 0
    failures: list[str] = field(default_factory=list)


def _replay_solve_identity_case(record: dict) -> None:
    import torch

    spec = build_case_spec_index()[("solve", "identity")]
    sample = SimpleNamespace(kwargs={})
    inputs = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in decode_tensor_map(record["inputs"]).items()
    }
    probe = record["probes"][0]
    direction = decode_tensor_map(probe["direction"])
    cotangent = decode_tensor_map(probe["cotangent"])
    stored_pytorch_jvp = decode_tensor_map(probe["pytorch_ref"]["jvp"])
    stored_pytorch_vjp = decode_tensor_map(probe["pytorch_ref"]["vjp"])
    stored_fd_jvp = decode_tensor_map(probe["fd_ref"]["jvp"])
    fd_step = probe["fd_ref"]["step"]
    comparison = record["comparison"]

    input_names = tuple(inputs.keys())

    def observable_fn(*args):
        replay_inputs = dict(zip(input_names, args, strict=True))
        observable = apply_spec_observable(torch, spec, sample, replay_inputs)
        return tensor_map_to_tuple(observable)

    observable = apply_spec_observable(torch, spec, sample, inputs)
    output_names = tuple(observable.keys())
    _, jvp_tuple = torch.func.jvp(
        observable_fn,
        tensor_map_to_tuple(inputs),
        tensor_map_to_tuple(direction),
    )
    pytorch_jvp = tuple_to_tensor_map(output_names, jvp_tuple)
    grads = torch.autograd.grad(
        tensor_map_to_tuple(observable),
        tensor_map_to_tuple(inputs),
        grad_outputs=tensor_map_to_tuple(cotangent),
        allow_unused=True,
    )
    pytorch_vjp = zeros_like_input_map(torch, inputs, grads)
    plus_inputs = {name: tensor + fd_step * direction[name] for name, tensor in inputs.items()}
    minus_inputs = {name: tensor - fd_step * direction[name] for name, tensor in inputs.items()}
    plus_output = apply_spec_observable(torch, spec, sample, plus_inputs)
    minus_output = apply_spec_observable(torch, spec, sample, minus_inputs)
    fd_jvp = {
        name: (plus_output[name] - minus_output[name]) / (2.0 * fd_step)
        for name in output_names
    }

    if not map_allclose(
        torch,
        stored_pytorch_jvp,
        pytorch_jvp,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    ):
        raise ValueError("stored and replayed PyTorch JVP disagree")
    if not map_allclose(
        torch,
        stored_pytorch_vjp,
        pytorch_vjp,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    ):
        raise ValueError("stored and replayed PyTorch VJP disagree")
    if not map_allclose(
        torch,
        stored_fd_jvp,
        fd_jvp,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    ):
        raise ValueError("stored and replayed FD-JVP disagree")


def replay_case_file(path: Path, *, limit: int | None = None) -> ReplayResult:
    """Replay one JSONL case file and report verification failures."""
    result = ReplayResult()
    records = load_case_file(path)
    for record in records[:limit]:
        try:
            if (record["op"], record["family"], record["expected_behavior"]) != (
                "solve",
                "identity",
                "success",
            ):
                raise ValueError("replay not implemented for this case family")
            _replay_solve_identity_case(record)
        except Exception as exc:
            result.failures.append(f"{record['case_id']}: {exc}")
        else:
            result.checked += 1
    return result
