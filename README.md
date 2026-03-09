# tensor-ad-oracles

`tensor-ad-oracles` is a machine-readable JSON database for derivative-correctness validation of tensor and linear algebra operations.

Version 1 targets the PyTorch-aligned case families for:

- `svd`
- `eigh`
- `solve`
- `cholesky`
- `qr`
- `pinv_singular`

## Environment

The repository is `uv`-managed. Use the checked-in patch-pinned Python version and lockfile.

```bash
uv sync --locked --all-groups
```

Typical commands:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m generators.pytorch_v1 --list
uv run python -m generators.pytorch_v1 --materialize solve --family identity --limit 1
```

Repository-managed environment files:

- `.python-version`
- `pyproject.toml`
- `uv.lock`

## What Counts As a Case

A case is defined by:

- materialized inputs
- an `observable`
- one or more paired derivative probes

The database does not require raw decomposition outputs to be the comparison target. For spectral operations, the observable may be a processed output such as `U.abs()`, `S`, `Vh.abs()`, or `U @ Vh`, following the same derivative-relevant observables used by PyTorch AD tests.

## Oracle Policy

Every `success` case must provide both:

- `pytorch_ref`
- `fd_ref`

Every published `success` case must satisfy:

- `Jv_torch ~= Jv_fd`
- `<bar_y, Jv_fd> ~= <J*bar_y_torch, v>`

`error` cases do not require numeric references. They encode expected failure behavior with a machine-readable reason code.

## License

This repository is dual-licensed under either of:

- Apache License, Version 2.0, in [LICENSE-APACHE](LICENSE-APACHE)
- MIT license, in [LICENSE-MIT](LICENSE-MIT)

At your option, you may use this repository under either license.

## PyTorch Provenance

Version 1 uses the same AD-relevant case families as PyTorch. Each case stores upstream provenance, including the source file, source function, and source commit used to generate the record.

## Repository Layout

```text
README.md
schema/
  case.schema.json
generators/
  __init__.py
  pytorch_v1.py
cases/
  svd/
  eigh/
  solve/
  cholesky/
  qr/
  pinv_singular/
```

Version 1 materializes these family files:

- `cases/svd/u_abs.jsonl`
- `cases/svd/s.jsonl`
- `cases/svd/vh_abs.jsonl`
- `cases/svd/uvh_product.jsonl`
- `cases/svd/gauge_ill_defined.jsonl`
- `cases/eigh/values_vectors_abs.jsonl`
- `cases/eigh/gauge_ill_defined.jsonl`
- `cases/solve/identity.jsonl`
- `cases/cholesky/identity.jsonl`
- `cases/qr/identity.jsonl`
- `cases/pinv_singular/identity.jsonl`

## Verification Contract

For every paired probe in every `success` case:

- compare `pytorch_ref.jvp` with `fd_ref.jvp`
- check adjoint consistency with `<bar_y, Jv_fd> ~= <J*bar_y_torch, v>`

All probe directions and cotangents are normalized to unit Frobenius norm after any required structure-preserving projection.
