# SVD AD Notes

## Forward Definition

For a complex or real matrix

$$
A = U \Sigma V^\dagger, \qquad \Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_K),
$$

with $K = \min(M, N)$, the thin SVD uses:

- $U \in \mathbb{C}^{M \times K}$ with $U^\dagger U = I$
- $V \in \mathbb{C}^{N \times K}$ with $V^\dagger V = I$
- $\sigma_i \ge 0$ in descending order

## Reverse Rule

Given cotangents $\bar U$, $\bar S$, and $\bar V$, the reverse rule is built
from the stabilized spectral gap matrix

$$
F_{ij} \approx \frac{1}{\sigma_j^2 - \sigma_i^2}, \qquad F_{ii} = 0,
$$

with a small regularizer in the implementation when singular values are nearly
repeated.

Define the inner matrix

$$
\Gamma = \Gamma_{\bar U} + \Gamma_{\bar V} + \Gamma_{\bar S},
$$

where the off-diagonal parts come from $F \odot (U^\dagger \bar U)$ and
$F \odot (V^\dagger \bar V)$ and the diagonal part from $\bar S$.

The core cotangent is

$$
\bar A_{\text{core}} = U \Gamma V^\dagger.
$$

For non-square matrices, extra projector terms recover the parts of $\bar U$ and
$\bar V$ that lie outside the thin column spaces:

$$
\bar A = \bar A_{\text{core}}
  + \mathbf{1}_{M>K} (I - U U^\dagger)\bar U \Sigma^{-1} V^\dagger
  + \mathbf{1}_{N>K} U \Sigma^{-1} (I - V V^\dagger)\bar V^\dagger.
$$

## Forward Rule

The forward rule solves the same coupled differential system for $(dU, dS, dV)$.
The singular-value differential is the diagonal part of $U^\dagger (dA) V$,
while the off-diagonal tangent pieces are determined by the same spectral-gap
matrix $F$ together with orthogonality projections back onto the Stiefel
manifolds.

## Numerical Notes

- Repeated or nearly repeated singular values make the $F$ matrix unstable.
- Raw $U$ and $V$ are gauge-dependent up to sign or complex phase.
- DB observables therefore use gauge-reduced projections such as `abs()` or
  `U @ Vh`.

## Verification

- Reconstruction: check $A \approx U \Sigma V^\dagger$.
- Orthonormality: check $U^\dagger U \approx I$ and $V^\dagger V \approx I$.
- Derivatives: compare JVP/VJP against finite differences on scalar test
  functionals.

## DB Families

<a id="family-u-abs"></a>
### `u_abs`

The DB publishes `U.abs()` rather than raw `U` to remove sign and phase gauge
ambiguity.

<a id="family-s"></a>
### `s`

The DB publishes the singular values directly.

<a id="family-vh-abs"></a>
### `vh_abs`

The DB publishes the pair `(S, Vh.abs())` so that singular values remain paired
with a gauge-stable right singular-vector observable.

<a id="family-uvh-product"></a>
### `uvh_product`

The DB publishes `(U @ Vh, S)`, which preserves the gauge-invariant subspace
information while keeping the singular values explicit.

<a id="family-svdvals-identity"></a>
### `svdvals/identity`

The `svdvals` family is the singular-value-only projection of the same spectral
rule. It reuses the singular-value part of the SVD differential.

<a id="family-gauge-ill-defined"></a>
### `gauge_ill_defined`

This family records expected failure cases where the chosen loss is not
gauge-invariant and derivatives through the decomposition are intentionally ill
defined.
