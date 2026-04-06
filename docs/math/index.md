# Math Notes

`tensor-ad-oracles` publishes two first-class artifacts:

- mathematical AD notes
- the machine-readable oracle database

The mathematical notes under `docs/math/` are the human-facing source of truth
for known AD rules in this repository. They are maintained to preserve the full
derivation detail migrated from `tenferro-rs/docs/AD/` while adding explicit
correspondence to PyTorch's manual autograd formulas where relevant.

Standalone linalg operations are documented as one note per operation, while
shared scalar and wrapper formulas are grouped where that keeps the corpus
easier to maintain.

Published DB families are linked to the note corpus through
`docs/math/registry.json`, which maps each materialized `(op, family)` pair to a
stable note anchor.

For the human-facing explanation of that linkage, see
[math-registry.md](../math-registry.md).

## Unified Note Model

Each published operator note now starts with the same six sections:

- `## Forward`
- `## Linearization`
- `## JVP`
- `## Transpose`
- `## VJP (JAX convention)`
- `## VJP (PyTorch convention)`

The intent is:

- `Forward` states the raw mathematical operator.
- `Linearization` and `Transpose` describe the raw-output-space differential and
  adjoint before any DB observable is applied.
- `JVP` records the tangent-evaluation view of the same raw rule.
- `VJP (JAX convention)` and `VJP (PyTorch convention)` explain how the same
  raw cotangent map is surfaced by each framework.

Unless a note says otherwise, `Transpose` means the adjoint under the real
Frobenius inner product

$$
\langle X, Y \rangle_{\mathbb{R}} = \operatorname{Re}\operatorname{tr}(X^\dagger Y).
$$

Later sections keep the full migrated derivations, case splits, and family
anchors intact.

## Core Linalg Notes

- [svd.md](./svd.md)
- [solve.md](./solve.md)
- [qr.md](./qr.md)
- [lu.md](./lu.md)
- [cholesky.md](./cholesky.md)
- [inv.md](./inv.md)
- [det.md](./det.md)
- [eig.md](./eig.md)
- [eigen.md](./eigen.md)
- [pinv.md](./pinv.md)
- [lstsq.md](./lstsq.md)
- [norm.md](./norm.md)

## Shared And Cross-Cutting Notes

- [scalar_ops.md](./scalar_ops.md)
- [matrix_exp.md](./matrix_exp.md)
- [dyadtensor_reverse.md](./dyadtensor_reverse.md)
