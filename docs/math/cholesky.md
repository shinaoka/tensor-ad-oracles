# Cholesky AD Notes

## Forward Definition

For a Hermitian positive-definite matrix

$$
A = L L^H,
$$

with lower-triangular $L$, the differential is

$$
dA = dL \, L^H + L \, dL^H.
$$

## Reverse Rule

The reverse rule first contracts the cotangent with the Cholesky factor to form
a lower-triangular sensitivity, then symmetrizes it back into the Hermitian
input space:

$$
\bar A = L^{-H} \, \Phi(L^H \bar L) \, L^{-1},
$$

where $\Phi$ denotes the structured symmetrization that keeps diagonal terms
once and off-diagonal terms in Hermitian pairs.

## Forward Rule

Forward mode solves a triangular Sylvester-style equation for $dL$ after
projecting the Hermitian perturbation into the valid lower-triangular tangent
space.

## Numerical Notes

- The factorization is only defined on positive-definite inputs.
- In the DB, Hermitian-wrapper semantics matter for the published Cholesky
  families; raw tensor payloads are interpreted through the same structured
  wrapper used upstream.

## Verification

- Reconstruction: check $A \approx L L^H$.
- Positive-definite domain handling: confirm failures occur outside the domain.
- Derivatives: compare JVP/VJP against finite differences.

## DB Families

<a id="family-cholesky-identity"></a>
### `cholesky/identity`

The DB publishes the differentiable Cholesky factor.

<a id="family-cholesky-ex-identity"></a>
### `cholesky_ex/identity`

The DB validates the factor output while treating auxiliary status outputs as
metadata.
