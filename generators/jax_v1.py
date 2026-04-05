"""JAX-backed v1 case materialization entrypoint."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

from . import probes, pytorch_v1, runtime_jax
from validators.encoding import decode_tensor_map


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schema" / "case.schema.json"


@dataclass(frozen=True)
class SimpleWitnessSpec:
    op: str
    family: str
    input_name: str
    input_value: float
    direction_value: float
    cotangent_value: float
    raw_output_function: object
    comment: str


@dataclass(frozen=True)
class HarnessWitnessSpec:
    source_file: str
    source_function: str
    harness_fullname: str


def _simple_torch_observable(torch, spec: pytorch_v1.CaseFamilySpec, x):
    if spec.op == "abs":
        return torch.abs(x)
    if spec.op == "exp":
        return torch.exp(x)
    raise ValueError(f"unsupported simple JAX smoke op: {spec.op}")


def _simple_raw_output_function(op: str, input_name: str):
    _, jnp = runtime_jax.import_generation_runtime()
    if op == "abs":
        return lambda inputs: {"value": jnp.abs(inputs[input_name])}
    if op == "exp":
        return lambda inputs: {"value": jnp.exp(inputs[input_name])}
    raise ValueError(f"unsupported simple JAX smoke op: {op}")


def _encode_torch_tensor_map(torch, tensor_map: dict[str, object]) -> dict[str, dict]:
    return runtime_jax.encode_tensor_map(
        {name: tensor.detach().clone().cpu() for name, tensor in tensor_map.items()}
    )


def _normalize_torch_tensor_map(torch, tensor_map: dict[str, object]) -> dict[str, object]:
    normalized = {}
    for name, tensor in tensor_map.items():
        norm = float(torch.linalg.vector_norm(tensor).item())
        normalized[name] = tensor.clone() if norm == 0.0 else tensor / norm
    return normalized


def _jax_test_harness_specs() -> dict[tuple[str, str], HarnessWitnessSpec]:
    return {
        ("abs", "identity"): HarnessWitnessSpec(
            source_file="tests/lax_numpy_test.py",
            source_function="testAbs",
            harness_fullname="jax/tests/lax_numpy_test.py::LaxNumpyTest.testAbs",
        ),
        ("exp", "identity"): HarnessWitnessSpec(
            source_file="tests/lax_numpy_test.py",
            source_function="testExp",
            harness_fullname="jax/tests/lax_numpy_test.py::LaxNumpyTest.testExp",
        ),
    }


def _published_torch_source_case_paths() -> dict[tuple[str, str], Path]:
    return {
        ("abs", "identity"): REPO_ROOT / "cases" / "abs" / "identity.jsonl",
        ("exp", "identity"): REPO_ROOT / "cases" / "exp" / "identity.jsonl",
    }


def select_witness_source(op: str, family: str, *, prefer_jax_test: bool = True) -> str:
    """Choose the preferred witness source for one JAX v1 case family."""
    key = (op, family)
    if prefer_jax_test and key in _jax_test_harness_specs():
        return "jax_test"
    if key in _published_torch_source_case_paths():
        return "torch_aligned"
    if key in _jax_test_harness_specs():
        return "jax_test"
    raise ValueError(f"unsupported JAX witness source selection key: {op}/{family}")


def _load_jsonl_case(path: Path, *, index: int = 0) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if index >= len(lines):
        raise IndexError(f"no case at line {index + 1} in {path}")
    return json.loads(lines[index])


def _decode_tensor_map_to_jax(jnp, encoded: dict[str, dict]) -> dict[str, object]:
    return {
        name: jnp.asarray(tensor.detach().cpu().numpy()) for name, tensor in decode_tensor_map(encoded).items()
    }


def build_case_families() -> dict[str, tuple[str, ...]]:
    """Return the fixed JAX v1 op/family registry."""
    return pytorch_v1.build_case_families()


def build_case_spec_index():
    """Reuse the PyTorch v1 case index for JAX materialization."""
    return pytorch_v1.build_case_spec_index()


def build_provenance(
    *,
    source_commit: str,
    seed: int,
    source_repo: str = "jax",
    source_file: str = "generators/jax_v1.py",
    source_function: str = "materialize_case_family",
    comment: str | None = None,
    generator: str = "python-jax-v1",
) -> dict:
    """Build the common provenance block for a JAX materialized case record."""
    return {
        "source_repo": source_repo,
        "source_file": source_file,
        "source_function": source_function,
        "source_commit": source_commit,
        "generator": generator,
        "seed": seed,
        "torch_version": pytorch_v1.PINNED_TORCH_VERSION,
        "fd_policy_version": pytorch_v1.FD_POLICY_VERSION,
        **({"comment": comment} if comment is not None else {}),
    }


def build_jax_ref_provenance(
    *,
    source_commit: str,
    seed: int,
    jax_version: str,
    jaxlib_version: str,
    witness_source: str,
    backend: str = "cpu",
    enable_x64: bool = True,
) -> dict:
    """Build the probe-level JAX witness provenance block."""
    del source_commit, seed
    return {
        "source_backend": "jax",
        "witness_source": witness_source,
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "backend": backend,
        "enable_x64": enable_x64,
    }


def case_output_path(spec: pytorch_v1.CaseFamilySpec, *, cases_root: Path | None = None) -> Path:
    root = cases_root if cases_root is not None else REPO_ROOT / "cases"
    return root / spec.op / f"{spec.family}.jsonl"


def write_case_records(
    spec: pytorch_v1.CaseFamilySpec,
    records: list[dict],
    *,
    cases_root: Path | None = None,
) -> Path:
    out_path = case_output_path(spec, cases_root=cases_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    return out_path


def make_success_case(
    spec: pytorch_v1.CaseFamilySpec,
    *,
    case_id: str,
    dtype: str,
    inputs: dict,
    comparison: dict,
    probes_: list[dict],
    provenance: dict,
) -> dict:
    case = {
        "schema_version": 1,
        "case_id": case_id,
        "op": spec.op,
        "dtype": dtype,
        "family": spec.family,
        "expected_behavior": "success",
        "inputs": inputs,
        "observable": {"kind": spec.observable_kind},
        "comparison": comparison,
        "probes": probes_,
        "provenance": provenance,
    }
    return case


def _simple_witness_specs() -> dict[tuple[str, str], SimpleWitnessSpec]:
    _, jnp = runtime_jax.import_generation_runtime()
    return {
        ("abs", "identity"): SimpleWitnessSpec(
            op="abs",
            family="identity",
            input_name="x",
            input_value=3.0,
            direction_value=1.0,
            cotangent_value=2.0,
            raw_output_function=lambda inputs: {"value": jnp.abs(inputs["x"])},
            comment="jax-native abs/identity smoke witness",
        ),
        ("exp", "identity"): SimpleWitnessSpec(
            op="exp",
            family="identity",
            input_name="x",
            input_value=1.0,
            direction_value=2.0,
            cotangent_value=3.0,
            raw_output_function=lambda inputs: {"value": jnp.exp(inputs["x"])},
            comment="jax-native exp/identity smoke witness",
        ),
    }


def _jax_test_comment(base_comment: str, harness_fullname: str) -> str:
    return f"{base_comment}; harness_fullname={harness_fullname}"


def _supported_materialization_keys() -> tuple[tuple[str, str], ...]:
    return tuple(_simple_witness_specs())


def _case_id(spec: pytorch_v1.CaseFamilySpec, *, dtype: str, index: int) -> str:
    dtype_tag = {
        "float32": "f32",
        "float64": "f64",
        "complex64": "c64",
        "complex128": "c128",
    }[dtype]
    return f"{spec.op}_{dtype_tag}_{spec.family}_{index:03d}"


def _materialize_simple_success_case(
    spec: pytorch_v1.CaseFamilySpec,
    witness_spec: SimpleWitnessSpec,
    *,
    seed: int,
    index: int,
    source_file: str | None = None,
    source_function: str | None = None,
    harness_fullname: str | None = None,
) -> dict:
    jax, jnp = runtime_jax.import_generation_runtime()
    import jaxlib
    import torch

    dtype = jnp.float64
    inputs = {witness_spec.input_name: jnp.array([witness_spec.input_value], dtype=dtype)}
    raw_direction = {witness_spec.input_name: jnp.array([witness_spec.direction_value], dtype=dtype)}
    raw_cotangent = {"value": jnp.array([witness_spec.cotangent_value], dtype=dtype)}
    direction = runtime_jax.normalize_raw_tensor_map(raw_direction)
    cotangent = runtime_jax.normalize_raw_tensor_map(raw_cotangent)

    torch_input = torch.tensor(
        [witness_spec.input_value], dtype=torch.float64, requires_grad=True
    )
    torch_direction = _normalize_torch_tensor_map(
        torch,
        {"x": torch.tensor([witness_spec.direction_value], dtype=torch.float64)},
    )["x"]
    torch_cotangent = _normalize_torch_tensor_map(
        torch,
        {"value": torch.tensor([witness_spec.cotangent_value], dtype=torch.float64)},
    )["value"]

    def torch_observable(x):
        return _simple_torch_observable(torch, spec, x)

    torch_output = torch_observable(torch_input)
    _, torch_jvp = torch.func.jvp(
        torch_observable,
        (torch_input,),
        (torch_direction,),
    )
    torch_vjp = torch.autograd.grad(
        torch_output,
        torch_input,
        grad_outputs=torch_cotangent,
        allow_unused=False,
    )[0]
    fd_step = 1e-6
    fd_jvp = (
        torch_observable(torch_input + fd_step * torch_direction)
        - torch_observable(torch_input - fd_step * torch_direction)
    ) / (2.0 * fd_step)

    jvp = runtime_jax.compute_jax_jvp(witness_spec.raw_output_function, inputs, direction)
    vjp = runtime_jax.compute_jax_vjp(witness_spec.raw_output_function, inputs, cotangent)
    linearization, linear_fn = runtime_jax.compute_jax_linearization(
        witness_spec.raw_output_function,
        inputs,
        direction,
    )
    transpose = runtime_jax.compute_jax_transpose(linear_fn, inputs, cotangent)

    adjoint_check = runtime_jax.compute_jax_adjoint_check(
        runtime_jax.tensor_map_inner_product(cotangent, jvp),
        runtime_jax.tensor_map_inner_product(transpose, direction),
    )

    probe = probes.make_probe_record(
        probe_id="p0",
        direction=runtime_jax.encode_tensor_map(direction),
        cotangent=runtime_jax.encode_tensor_map(cotangent),
        pytorch_jvp=_encode_torch_tensor_map(torch, {"value": torch_jvp}),
        pytorch_vjp=_encode_torch_tensor_map(torch, {witness_spec.input_name: torch_vjp}),
        fd_step=fd_step,
        fd_jvp=_encode_torch_tensor_map(torch, {"value": fd_jvp}),
        jax_jvp=runtime_jax.encode_tensor_map(jvp),
        jax_vjp=runtime_jax.encode_tensor_map(vjp),
        jax_linearization=runtime_jax.encode_tensor_map(linearization),
        jax_raw_output_cotangent=runtime_jax.encode_tensor_map(cotangent),
        jax_transpose=runtime_jax.encode_tensor_map(transpose),
        jax_adjoint_check=adjoint_check,
        jax_provenance=build_jax_ref_provenance(
            source_commit=runtime_jax.normalize_jax_version(jax.__version__),
            seed=seed,
            jax_version=runtime_jax.normalize_jax_version(jax.__version__),
            jaxlib_version=runtime_jax.normalize_jax_version(jaxlib.__version__),
            witness_source="jax_test",
            enable_x64=True,
        ),
    )

    comparison = {
        "first_order": {
            "kind": "allclose",
            "rtol": 1e-12,
            "atol": 1e-12,
        }
    }
    return make_success_case(
        spec,
        case_id=_case_id(spec, dtype="float64", index=index),
        dtype="float64",
        inputs=runtime_jax.encode_tensor_map(inputs),
        comparison=comparison,
        probes_=[probe],
        provenance=build_provenance(
            source_commit=runtime_jax.normalize_jax_version(jax.__version__),
            seed=seed,
            source_file=source_file or "generators/jax_v1.py",
            source_function=source_function or "materialize_case_family",
            comment=(
                _jax_test_comment(witness_spec.comment, harness_fullname)
                if harness_fullname is not None
                else witness_spec.comment
            ),
        ),
    )


def enrich_torch_aligned_case_record(
    spec: pytorch_v1.CaseFamilySpec,
    source_case: dict,
    *,
    seed: int,
) -> dict:
    """Attach a JAX witness to a published PyTorch case without changing its inputs."""
    jax, jnp = runtime_jax.import_generation_runtime()
    import jaxlib

    enriched = copy.deepcopy(source_case)

    source_inputs = enriched["inputs"]
    input_name = next(iter(source_inputs))
    inputs = _decode_tensor_map_to_jax(jnp, source_inputs)
    probe = enriched["probes"][0]
    direction = _decode_tensor_map_to_jax(jnp, probe["direction"])
    cotangent = _decode_tensor_map_to_jax(jnp, probe["cotangent"])
    raw_output_function = _simple_raw_output_function(spec.op, input_name)
    jvp = runtime_jax.compute_jax_jvp(raw_output_function, inputs, direction)
    vjp = runtime_jax.compute_jax_vjp(raw_output_function, inputs, cotangent)
    linearization, linear_fn = runtime_jax.compute_jax_linearization(
        raw_output_function,
        inputs,
        direction,
    )
    transpose = runtime_jax.compute_jax_transpose(linear_fn, inputs, cotangent)
    adjoint_check = runtime_jax.compute_jax_adjoint_check(
        runtime_jax.tensor_map_inner_product(cotangent, jvp),
        runtime_jax.tensor_map_inner_product(transpose, direction),
    )

    probe["jax_ref"] = {
        "jvp": runtime_jax.encode_tensor_map(jvp),
        "vjp": runtime_jax.encode_tensor_map(vjp),
        "linearization": runtime_jax.encode_tensor_map(linearization),
        "raw_output_cotangent": runtime_jax.encode_tensor_map(cotangent),
        "transpose": runtime_jax.encode_tensor_map(transpose),
        "adjoint_check": adjoint_check,
        "provenance": build_jax_ref_provenance(
            source_commit=runtime_jax.normalize_jax_version(jax.__version__),
            seed=seed,
            jax_version=runtime_jax.normalize_jax_version(jax.__version__),
            jaxlib_version=runtime_jax.normalize_jax_version(jaxlib.__version__),
            witness_source="torch_aligned",
            enable_x64=True,
        ),
    }

    enriched["provenance"] = build_provenance(
        source_repo=source_case["provenance"]["source_repo"],
        source_file=source_case["provenance"]["source_file"],
        source_function=source_case["provenance"]["source_function"],
        source_commit=source_case["provenance"]["source_commit"],
        seed=seed,
        comment=source_case["provenance"].get("comment"),
        generator="python-jax-v1",
    )
    return enriched


def materialize_torch_aligned_case_family(
    op: str,
    family: str,
    *,
    limit: int = 1,
    cases_root: Path | None = None,
    source_case_path: Path | None = None,
) -> Path:
    """Materialize a JAX witness from a published PyTorch source case."""
    spec = build_case_spec_index()[(op, family)]
    key = (op, family)
    if key not in _published_torch_source_case_paths():
        raise ValueError(
            f"JAX v1 only has published torch-aligned source cases for: {op}/{family}"
        )
    source_path = source_case_path or _published_torch_source_case_paths()[key]
    source_case = _load_jsonl_case(source_path)
    records = [
        enrich_torch_aligned_case_record(
            spec,
            source_case,
            seed=17 + index,
        )
        for index in range(limit)
    ]
    return write_case_records(spec, records, cases_root=cases_root)


def materialize_case_family(
    op: str,
    family: str,
    *,
    limit: int = 1,
    cases_root: Path | None = None,
) -> Path:
    spec = build_case_spec_index()[(op, family)]
    key = (op, family)
    supported = _simple_witness_specs()
    if key not in supported:
        raise ValueError(f"JAX v1 only materializes simple smoke witnesses in this task: {op}/{family}")
    witness_source = select_witness_source(op, family)
    if witness_source == "torch_aligned":
        return materialize_torch_aligned_case_family(
            op,
            family,
            limit=limit,
            cases_root=cases_root,
        )
    harness_spec = _jax_test_harness_specs().get(key)
    records = [
        _materialize_simple_success_case(
            spec,
            supported[key],
            seed=17 + index,
            index=index + 1,
            source_file=harness_spec.source_file if harness_spec is not None else None,
            source_function=harness_spec.source_function if harness_spec is not None else None,
            harness_fullname=harness_spec.harness_fullname if harness_spec is not None else None,
        )
        for index in range(limit)
    ]
    return write_case_records(spec, records, cases_root=cases_root)


def materialize_all_case_families(
    *,
    limit: int = 1,
    cases_root: Path | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for op, family in _supported_materialization_keys():
        paths.append(
            materialize_case_family(
                op,
                family,
                limit=limit,
                cases_root=cases_root,
            )
        )
    return paths


def _iter_registry_lines() -> list[str]:
    case_families = build_case_families()
    return [f"{op}: {', '.join(case_families[op])}" for op in case_families]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print the fixed v1 op/family registry and exit.")
    parser.add_argument("--materialize", choices=tuple(build_case_families()), help="Materialize one supported op family into JSONL.")
    parser.add_argument("--materialize-all", action="store_true", help="Materialize the supported smoke witnesses into JSONL files.")
    parser.add_argument("--family", help="Case family to materialize for the selected op.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of records to materialize.")
    parser.add_argument("--cases-root", type=Path, default=None, help="Optional output root for generated cases.")
    args = parser.parse_args(argv)

    if args.list:
        for line in _iter_registry_lines():
            print(line)
        return 0

    if args.materialize:
        if not args.family:
            raise SystemExit("--family is required with --materialize")
        out_path = materialize_case_family(
            args.materialize,
            args.family,
            limit=args.limit,
            cases_root=args.cases_root,
        )
        print(out_path)
        return 0

    if args.materialize_all:
        for path in materialize_all_case_families(limit=args.limit, cases_root=args.cases_root):
            print(path)
        return 0

    raise SystemExit("JAX v1 case generation is not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
