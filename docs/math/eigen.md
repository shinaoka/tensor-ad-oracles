# Hermitian Eigen AD Notes

## Forward Definition

For a Hermitian matrix

$$
A Q = Q \Lambda,
$$

with real eigenvalues collected in the diagonal matrix $\Lambda$ and unitary
eigenvectors $Q$.

## Reverse Rule

Because the eigenbasis is orthonormal, the reverse rule simplifies to

$$
\bar A = Q \, G \, Q^H,
$$

where $G$ contains:

- the diagonal eigenvalue cotangent
- the off-diagonal eigenvector sensitivity mixed by the Hermitian spectral-gap
  matrix

$$
F_{ij} \approx \frac{1}{\lambda_j - \lambda_i}, \qquad F_{ii} = 0.
$$

The Hermitian structure of $A$ is preserved by symmetrizing the assembled inner
matrix before projecting back.

## Forward Rule

Forward mode differentiates the Hermitian eigen equation and solves the same
gap-weighted system for $d\Lambda$ and $dQ$.

## Numerical Notes

- Repeated eigenvalues create gauge freedom in the eigenvectors.
- The DB therefore uses `vectors.abs()` for the published vector observable.
- Gauge-noninvariant losses are intentionally represented by explicit error
  families.

## Verification

- Reconstruction: check $A Q \approx Q \Lambda$.
- Hermitian invariance: ensure the primal input stays in the structured domain.
- Derivatives: compare JVP/VJP against finite differences.

## DB Families

<a id="family-values-vectors-abs"></a>
### `values_vectors_abs`

The DB publishes the real eigenvalues together with `vectors.abs()` to remove
the sign or phase ambiguity of the Hermitian eigenbasis.

<a id="family-eigvalsh-identity"></a>
### `eigvalsh/identity`

The eigenvalue-only Hermitian family reuses the diagonal part of the same rule.

<a id="family-gauge-ill-defined"></a>
### `gauge_ill_defined`

This family records expected failures for losses that are not invariant under
the eigenvector gauge freedom.
