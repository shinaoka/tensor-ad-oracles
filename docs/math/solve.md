# Solve AD Notes

## Forward Definition

For a linear system

$$
A X = B,
$$

the primal solution is

$$
X = A^{-1} B.
$$

## Differential

Differentiate the defining equation:

$$
dA \, X + A \, dX = dB,
$$

so

$$
dX = A^{-1}(dB - dA \, X).
$$

This identity is the core forward-mode rule for dense solve families.

## Reverse Rule

Given a cotangent $\bar X$ for the solution,

$$
\bar B = A^{-H}\bar X
$$

and

$$
\bar A = -A^{-H}\bar X X^H.
$$

The same structure appears in triangular and factor-backed solves; only the
primal solver and any structure-preserving projections differ.

## Structured Variants

- `solve_ex` shares the same derivative on the solution output; status outputs
  are nondifferentiable metadata.
- `solve_triangular` uses the same differential identity with a triangular solve
  in place of an unrestricted inverse.
- `lu_solve` reuses the solve cotangent while taking the LU factorization and
  pivots as the primal inputs.
- `tensorsolve` is the tensor-indexed version of the same implicit-system rule.

## Verification

- Primal residual: check $A X \approx B$.
- Forward mode: compare $dX$ against finite differences.
- Reverse mode: compare VJP with finite differences of scalar losses built from
  the solution.

## DB Families

<a id="family-solve-identity"></a>
### `solve/identity`

The DB publishes the solution tensor directly.

<a id="family-solve-ex-identity"></a>
### `solve_ex/identity`

The DB validates the differentiable solution output; auxiliary execution-status
fields are treated as metadata.

<a id="family-solve-triangular-identity"></a>
### `solve_triangular/identity`

The DB applies the same solve differential with the triangular structure
enforced by the primal operator.

<a id="family-lu-solve-identity"></a>
### `lu_solve/identity`

The DB uses the solution observable for factor-backed solves as well.

<a id="family-tensorsolve-identity"></a>
### `tensorsolve/identity`

The DB treats `tensorsolve` as the indexed tensor analogue of linear solve.
