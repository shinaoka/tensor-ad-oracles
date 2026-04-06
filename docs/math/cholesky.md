# Cholesky AD Notes

## Conventions

Unless noted otherwise, `Linearization` and `Transpose` are written for the
raw-output-space Cholesky map before any DB observable projection. For complex
tensors, `Transpose` means the adjoint under the real Frobenius inner product

$$
\langle X, Y \rangle_{\mathbb{R}} = \operatorname{Re}\operatorname{tr}(X^\dagger Y).
$$

## Forward

The raw operator is the lower-triangular factor

$$
A \mapsto L,
\qquad
A = L L^{\mathsf{H}},
\qquad
A = A^{\mathsf{H}} \succ 0.
$$

## Linearization

With the helper

$$
\varphi(X) = \mathrm{tril}(X) - \tfrac{1}{2}\mathrm{diag}(X),
$$

the raw-output-space linearization is

$$
\dot{L} = L \, \varphi\!\bigl(L^{-1}\dot{A}\,L^{-\mathsf{H}}\bigr).
$$

## JVP

The JVP is exactly the same tangent formula:

$$
\operatorname{jvp}(\operatorname{chol})(A;\dot{A})
= L \, \varphi\!\bigl(L^{-1}\dot{A}\,L^{-\mathsf{H}}\bigr).
$$

## Transpose

For a raw output cotangent $\bar{L}$, the transpose map is

$$
\bar{A} =
L^{-\mathsf{H}} \,
\varphi^*\!\bigl(\mathrm{tril}(L^{\mathsf{H}}\bar{L})\bigr)
\, L^{-1}.
$$

## VJP (JAX convention)

JAX reads the same raw transpose map directly as the cotangent rule on the
Cholesky factor.

## VJP (PyTorch convention)

PyTorch uses the same triangular-solve sandwich. For `cholesky_ex`, auxiliary
status outputs are treated as metadata, so the VJP applies only to the factor
output.

## Forward Definition

$$
A = L L^{\mathsf{H}},
\qquad
A \in \mathbb{C}^{N \times N},
\qquad
A = A^{\mathsf{H}},
\qquad
L \text{ lower triangular}
$$

with $A$ Hermitian positive-definite.

## Auxiliary Operator

Define

$$
\varphi(X) = \mathrm{tril}(X) - \tfrac{1}{2}\mathrm{diag}(X),
$$

which extracts the lower triangle and halves the diagonal. Its adjoint is

$$
\varphi^*(X) = \tfrac{1}{2}(X + X^{\mathsf{H}} - \mathrm{diag}(X)).
$$

## Forward Rule

Given a Hermitian tangent $\dot{A}$:

$$
\dot{L} = L \, \varphi\!\bigl(L^{-1}\dot{A}\,L^{-\mathsf{H}}\bigr).
$$

Differentiate $A = L L^{\mathsf{H}}$:

$$
\dot{A} = \dot{L} L^{\mathsf{H}} + L \dot{L}^{\mathsf{H}}.
$$

Left-multiplying by $L^{-1}$ and right-multiplying by $L^{-\mathsf{H}}$ gives

$$
L^{-1}\dot{A}\,L^{-\mathsf{H}} =
L^{-1}\dot{L} + (L^{-1}\dot{L})^{\mathsf{H}}.
$$

Since $L^{-1}\dot{L}$ is lower triangular, $\varphi$ inverts this
symmetrization.

## Reverse Rule

Given a cotangent $\bar{L}$:

$$
\bar{A} =
L^{-\mathsf{H}} \,
\varphi^*\!\bigl(\mathrm{tril}(L^{\mathsf{H}}\bar{L})\bigr)
\, L^{-1}.
$$

This is the adjoint of the JVP map and keeps $\bar{A}$ Hermitian.

Never form $L^{-1}$ explicitly; use triangular solves on the left and right.

## Verification

### Forward reconstruction

$$
\|A - L L^{\mathsf{H}}\|_F < \varepsilon.
$$

### Backward checks

- compare JVP/VJP against finite differences on Hermitian perturbations
- confirm failure outside the positive-definite domain

## References

1. S. P. Smith, "Differentiation of the Cholesky Algorithm," 1995.
2. I. Murray, "Differentiation of the Cholesky decomposition," 2016.

## DB Families

<a id="family-cholesky-identity"></a>
### `cholesky/identity`

The DB publishes the differentiable Cholesky factor.

<a id="family-cholesky-ex-identity"></a>
### `cholesky_ex/identity`

The DB validates the factor output while treating auxiliary status outputs as
metadata.
