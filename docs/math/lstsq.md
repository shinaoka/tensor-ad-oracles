# Least Squares AD Notes

## Forward Definition

For the least-squares problem

$$
X = \arg\min_Z \|A Z - B\|_F^2,
$$

the primal solution satisfies the normal-equation or QR/SVD-based solve used by
the implementation.

## Reverse Rule

The least-squares adjoint has two pieces:

- the solve-like cotangent through the solution itself
- a residual correction term that accounts for how the minimizer changes as the
  residual subspace rotates

This residual term is what distinguishes `lstsq_rrule` from a plain inverse or
solve rule.

## Forward Rule

Forward mode differentiates the implicit optimality condition of the least
squares problem and then solves for the tangent of the minimizer.

## Numerical Notes

- Rank deficiency and driver choice affect the conditioning of the derivative.
- The DB currently materializes the gradient-oriented upstream variant.

## Verification

- Primal residual: check the least-squares normality condition.
- Derivatives: compare JVP/VJP against finite differences.

## DB Families

<a id="family-lstsq-grad-oriented-identity"></a>
### `lstsq_grad_oriented/identity`

The DB publishes the differentiable least-squares outputs for the
gradient-oriented upstream variant.
