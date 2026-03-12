# Dyadtensor Reverse Wiring Notes

## Purpose

This note records how higher-level dyadtensor APIs attach reverse-mode pullbacks
to user-visible tensor values without exposing tape internals.

## Reverse Registration Model

Builder or eager entrypoints such as:

- `einsum_ad(...).run()`
- `svd_ad(...).run()`
- `qr_ad(...).run()`
- `lu_ad(...).run()`
- `eigen_ad(...).run()`
- `lstsq_ad(...).run()`
- `solve_ad(...).run()`
- `inv_ad(...).run()`
- `det_ad(...).run()`
- `slogdet_ad(...).run()`
- `eig_ad(...).run()`
- `pinv_ad(...).run()`
- `matrix_exp_ad(...).run()`
- `norm_ad(...).run()`

register a local pullback on the tensor-local tape node. User code then asks for
pullbacks through wrapper helpers instead of manipulating tape objects directly.

## Mixed-Type Pullbacks

Most pullbacks stay in one scalar domain. A small number of operators, notably
general `eig`, need a mixed-domain bridge because the primal input and output
scalar domains differ.

## Why This Note Exists Here

The math corpus needs one place that explains how raw operator rules become
higher-level tensor-facing reverse APIs. This note is the bridge between:

- raw operator notes such as [eig.md](./eig.md)
- shared wrapper formulas in [scalar_ops.md](./scalar_ops.md)
- eventual downstream DB replay or frontend documentation

## DB Status

This note documents implementation wiring rather than a published `(op, family)`
oracle family, so it does not currently appear in `docs/math/registry.json`.
