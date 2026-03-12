# Matrix Exponential AD Notes

## Forward Definition

For a square matrix

$$
B = \exp(A),
$$

the Fr\'echet derivative in direction $E$ is

$$
L(A, E) = \int_0^1 \exp(sA)\,E\,\exp((1-s)A)\,ds.
$$

## Block-Matrix Formula

Both forward and reverse mode can be expressed through a single exponential of a
block upper-triangular matrix:

$$
\exp\!\begin{pmatrix} A & E \\ 0 & A \end{pmatrix}
= \begin{pmatrix} \exp(A) & L(A, E) \\ 0 & \exp(A) \end{pmatrix}.
$$

This gives a direct JVP by using $E = dA$ and a direct VJP by using
$A^H$ on the diagonal and the output cotangent in the upper-right block.

## Reverse Rule

Given an output cotangent $\bar B$,

$$
\bar A = L(A^H, \bar B).
$$

That is the adjoint of the linear map $E \mapsto L(A, E)$ under the standard
Frobenius inner product.

## Numerical Notes

- The block-matrix construction is simple but more expensive than a dedicated
  scaling-and-squaring Fr\'echet implementation.
- Non-normal matrices remain numerically delicate.

## Verification

- Compare the block-matrix Fr\'echet derivative against finite differences.
- Check JVP/VJP agreement on scalar losses of `matrix_exp(A)`.

## DB Status

`matrix_exp` is documented here as a known rule, but it is **not yet materialized**
in the current published `cases/` tree.
