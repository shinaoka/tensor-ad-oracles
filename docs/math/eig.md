# General Eigen AD Notes

## Forward Definition

For a diagonalizable matrix

$$
A V = V \Lambda,
$$

with eigenvalue diagonal $\Lambda$ and right eigenvectors $V$.

## Reverse Rule

The reverse rule separates diagonal and off-diagonal sensitivities:

- eigenvalue sensitivities land on the diagonal
- eigenvector sensitivities are mixed through the spectral-gap matrix

$$
F_{ij} \approx \frac{1}{\lambda_j - \lambda_i}, \qquad F_{ii} = 0.
$$

The adjoint is assembled from the eigenvalue cotangent and the projected
eigenvector cotangent, then mapped back by the inverse eigenvector basis.

## Forward Rule

Forward mode differentiates the eigen equation and solves the same spectral-gap
system for $d\Lambda$ and $dV$.

## Numerical Notes

- The rule assumes separated eigenvalues.
- Non-normal matrices and clustered eigenvalues are numerically fragile.
- Raw eigenvectors are gauge-dependent up to per-column phase, so the DB uses
  gauge-reduced observables.

## Verification

- Reconstruction: check $A V \approx V \Lambda$.
- Derivatives: compare JVP/VJP against finite differences on scalar losses.

## DB Families

<a id="family-values-vectors-abs"></a>
### `values_vectors_abs`

The DB publishes the eigenvalues together with `vectors.abs()` to remove the
phase ambiguity of the general eigenvectors.

<a id="family-eigvals-identity"></a>
### `eigvals/identity`

The eigenvalue-only family reuses the diagonal part of the same general eigen
differential.
