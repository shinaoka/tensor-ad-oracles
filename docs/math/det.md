# Determinant AD Notes

## Forward Definition

For a nonsingular square matrix,

$$
d\,\det(A) = \det(A)\operatorname{tr}(A^{-1} dA).
$$

For the logarithmic determinant,

$$
d\,\log\det(A) = \operatorname{tr}(A^{-1} dA).
$$

## Reverse Rule

If $\bar d$ is the cotangent of `det(A)`, then

$$
\bar A = \bar d \, \det(A)^* A^{-H}.
$$

For `slogdet(A) = (\operatorname{sign}, \log|\det A|)`, the differentiable part
is the log-absolute-determinant term, whose cotangent is proportional to
$A^{-H}$.

## Numerical Notes

- Singular or nearly singular matrices make both primal and derivative unstable.
- `slogdet` is preferred when the primal magnitude would overflow or underflow.

## Verification

- Reconstruction: compare `det(A)` or `slogdet(A)` with direct primal results.
- Derivatives: compare against finite differences of scalar losses.

## DB Families

<a id="family-det-identity"></a>
### `det/identity`

The DB publishes the determinant value directly.

<a id="family-slogdet-identity"></a>
### `slogdet/identity`

The DB publishes the differentiable `slogdet` observable directly.
