# QR AD Notes

## Forward Definition

For the reduced QR factorization

$$
A = Q R,
$$

with orthonormal columns in $Q$ and upper-triangular $R$, the differential is

$$
dA = dQ \, R + Q \, dR.
$$

The orthogonality constraint $Q^\dagger Q = I$ implies that $Q^\dagger dQ$ is
skew-Hermitian.

## Reverse Rule

The reverse rule solves for the skew-Hermitian part of $Q^\dagger dQ$ and the
triangular part of $dR$ simultaneously. Implementations usually express the
result in terms of a triangular copy/projection helper such as `copyltu`,
ensuring that the cotangent respects the QR output structure.

The final cotangent $\bar A$ is assembled from the projected $\bar Q$ and
$\bar R$ terms and then corrected for tall and wide cases so that it stays in
the valid tangent space of the factorization.

## Forward Rule

Forward mode uses the same decomposition:

- solve the triangular differential for $dR$
- recover $dQ$ from the residual orthogonal component

## Verification

- Reconstruction: check $A \approx Q R$.
- Orthogonality: check $Q^\dagger Q \approx I$.
- Derivatives: compare JVP/VJP against finite differences.

## DB Families

<a id="family-identity"></a>
### `identity`

The DB publishes the differentiable QR outputs directly.
