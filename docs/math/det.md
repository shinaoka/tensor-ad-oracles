# Determinant AD Notes

## 1. Determinant

### Forward Definition

For

$$
d = \det(A),
\qquad
A \in \mathbb{C}^{N \times N},
$$

Jacobi's formula gives

$$
\dot{d} = \det(A) \cdot \operatorname{tr}(A^{-1}\dot{A}).
$$

### Reverse Rule

Given a cotangent $\bar{d}$:

- real case:

$$
\bar{A} = \bar{d} \cdot \det(A) \cdot A^{-\mathsf{T}}
$$

- complex case:

$$
\bar{A} = \bar{d} \cdot \overline{\det(A)} \cdot A^{-\mathsf{H}}.
$$

## Singular matrix handling

The inverse formula fails at singular matrices, but the adjugate interpretation
still makes sense:

- rank $N-1$: the adjugate is rank 1 and can be reconstructed from an SVD
- rank $\le N-2$: the adjugate vanishes

The rank-$N-1$ adjugate can be reconstructed from the leave-one-out singular
value products together with the orientation/phase factor carried by the
singular vectors.

## 2. `slogdet`

### Forward Definition

$$
(\operatorname{sign}, \operatorname{logabsdet}) = \operatorname{slogdet}(A).
$$

If $w = \operatorname{tr}(A^{-1}\dot{A})$, then in the complex case

$$
\dot{\operatorname{logabsdet}} = \operatorname{Re}(w),
\qquad
\dot{\operatorname{sign}} = i \operatorname{Im}(w)\operatorname{sign}.
$$

### Reverse Rule

Given cotangents $\bar{s}$ for the sign output and $\bar{\ell}$ for the
log-magnitude output:

- real case:

$$
\bar{A} = \bar{\ell} \cdot A^{-\mathsf{T}}
$$

- complex case:

$$
\bar{A} = g \cdot A^{-\mathsf{H}},
\qquad
g = \bar{\ell} - i \operatorname{Im}(\bar{s}^* s),
$$

where $s = \operatorname{sign}(A)$.

`slogdet` is not differentiable at singular matrices because
$\operatorname{logabsdet} = -\infty$ there.

## Verification

- compare primal `det(A)` and `slogdet(A)` with direct evaluation
- compare JVP/VJP against finite differences away from singularity

## References

1. C. G. J. Jacobi, "De formatione et proprietatibus determinantium," 1841.
2. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode AD," 2008.

## DB Families

<a id="family-det-identity"></a>
### `det/identity`

The DB publishes the determinant value directly.

<a id="family-slogdet-identity"></a>
### `slogdet/identity`

The DB publishes the differentiable `slogdet` observable directly.
