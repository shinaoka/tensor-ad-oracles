# Inverse AD Notes

## Conventions

Unless noted otherwise, `Linearization` and `Transpose` are written for the
raw-output-space inverse map. For complex tensors, `Transpose` means the adjoint
under the real Frobenius inner product

$$
\langle X, Y \rangle_{\mathbb{R}} = \operatorname{Re}\operatorname{tr}(X^\dagger Y).
$$

## Forward

The raw operator is

$$
A \mapsto B = A^{-1}.
$$

## Linearization

Differentiating $A B = I$ gives

$$
\dot{B} = -B\,\dot{A}\,B.
$$

## JVP

The JVP is the same linearization evaluated at $\dot{A}$:

$$
\operatorname{jvp}(\operatorname{inv})(A;\dot{A}) = -B\,\dot{A}\,B.
$$

## Transpose

For a raw output cotangent $\bar{B}$, the transpose map is

$$
\bar{A} = -B^{\mathsf{H}}\,\bar{B}\,B^{\mathsf{H}}.
$$

## VJP (JAX convention)

JAX exposes the same transpose rule directly on the inverse output cotangent.

## VJP (PyTorch convention)

PyTorch uses the same raw adjoint, typically via the corresponding `solve`
formula. Real inputs remain in the real domain after the final cotangent
projection.

## Forward Definition

$$
B = A^{-1},
\qquad
A \in \mathbb{C}^{N \times N}
$$

## Forward Rule

Differentiate $A B = I$:

$$
\dot{A} B + A \dot{B} = 0
$$

so

$$
\dot{B} = -B\,\dot{A}\,B.
$$

## Reverse Rule

Given a cotangent $\bar{B}$:

$$
\bar{A} = -B^{\mathsf{H}}\,\bar{B}\,B^{\mathsf{H}}.
$$

This is the adjoint of the JVP under the Frobenius inner product.

## Relationship to solve

`inv(A)` is the special case of `solve(A, I)`. Reusing the solve notation
immediately recovers

- JVP: $\dot{B} = -B\,\dot{A}\,B$
- VJP: $\bar{A} = -B^{\mathsf{H}}\,\bar{B}\,B^{\mathsf{H}}$

The same relationship is used in PyTorch and downstream libraries to avoid
duplicating logic.

## Implementation Correspondence

- `tenferro-rs/docs/AD/inv.md` writes the inverse rule directly and then points
  back to solve as the conceptual source.
- PyTorch exposes the inverse derivative via solve-style formulas in
  `derivatives.yaml` and related linear-solve kernels.
- For higher-order AD, prefer `solve` over explicit multiplication by a cached
  inverse.

## Verification

### Forward reconstruction

$$
A B \approx I.
$$

### Backward checks

Compare JVP/VJP against finite differences on scalar losses of the inverse.

## References

1. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode AD," 2008.
2. P. S. Dwyer and M. S. Macphail, "Symbolic Matrix Derivatives," 1948.

## DB Families

<a id="family-inv-identity"></a>
### `inv/identity`

The DB publishes the inverse tensor directly.

<a id="family-inv-ex-identity"></a>
### `inv_ex/identity`

The DB validates the inverse output for the extended variant and treats status
metadata as nondifferentiable.

<a id="family-tensorinv-identity"></a>
### `tensorinv/identity`

The tensor inverse family is the index-reshaped analogue of the same inverse
rule.
