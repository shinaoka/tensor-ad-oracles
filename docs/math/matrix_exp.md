# Matrix Exponential AD Notes

## Conventions

Unless noted otherwise, `Linearization` and `Transpose` are written for the
raw-output-space matrix exponential map. For complex tensors, `Transpose` means
the adjoint under the real Frobenius inner product

$$
\langle X, Y \rangle_{\mathbb{R}} = \operatorname{Re}\operatorname{tr}(X^\dagger Y).
$$

## Forward

The raw operator is

$$
A \mapsto B = \exp(A).
$$

## Linearization

The raw-output-space linearization is the Fr\'echet derivative

$$
\dot{B} = L(A, \dot{A})
= \int_0^1 \exp(sA)\,\dot{A}\,\exp((1-s)A)\,ds.
$$

## JVP

The JVP is exactly the same Fr\'echet derivative applied to the tangent:

$$
\operatorname{jvp}(\operatorname{matrix\_exp})(A;\dot{A}) = L(A,\dot{A}).
$$

## Transpose

For a raw output cotangent $\bar{B}$, the transpose map is

$$
\bar{A} = L(A^{\mathsf{H}}, \bar{B}).
$$

## VJP (JAX convention)

JAX reads the same Fr\'echet-adjoint map as the VJP or `linear_transpose`
result.

## VJP (PyTorch convention)

PyTorch uses the same raw rule through
`differential_analytic_matrix_function`; the public VJP is the same adjoint map
packaged through the framework's cotangent convention.

## Forward Definition

$$
B = \exp(A),
\qquad
A \in \mathbb{C}^{N \times N}
$$

The Fr\'echet derivative in direction $E$ is

$$
L(A, E) =
\int_0^1 \exp(sA)\,E\,\exp((1-s)A)\,ds.
$$

## Block Matrix Formula (Mathias 1996)

Both JVP and VJP can be written through a single exponential of a
$2N \times 2N$ block upper-triangular matrix:

$$
\exp\!\begin{pmatrix} A & E \\ 0 & A \end{pmatrix}
= \begin{pmatrix} \exp(A) & L(A, E) \\ 0 & \exp(A) \end{pmatrix}.
$$

The upper-right block is the Fr\'echet derivative.

## Forward Rule

Given $\dot{A}$:

$$
\dot{B} = L(A, \dot{A}),
$$

which is the upper-right block of the block exponential above.

## Reverse Rule

Given a cotangent $\bar{B}$:

$$
\bar{A} = L(A^{\mathsf{H}}, \bar{B}),
$$

which is the adjoint of the Fr\'echet derivative map under the Frobenius inner
product.

## Generality

The same block-matrix technique works for any analytic matrix function
$f$, not just the exponential:

$$
f\!\begin{pmatrix} A & E \\ 0 & A \end{pmatrix}
= \begin{pmatrix} f(A) & L_f(A, E) \\ 0 & f(A) \end{pmatrix}.
$$

## Computational cost

| Method | Cost relative to $\exp(A)$ |
|---|---|
| Block matrix ($2N \times 2N$) | about $8\times$ |
| Dedicated Fr\'echet scaling-and-squaring | about $3\times$ |
| Eigendecomposition shortcut | cheaper on paper, but unstable for non-normal $A$ |

## Verification

- compare the block-matrix Fr\'echet derivative against finite differences
- check JVP/VJP agreement on scalar losses of `matrix_exp(A)`

## References

1. R. Mathias, "A Chain Rule for Matrix Functions and Applications," 1996.
2. A. H. Al-Mohy and N. J. Higham, "Computing the Frechet Derivative of the
   Matrix Exponential," 2009.

## DB Families

<a id="family-identity"></a>
### `matrix_exp/identity`

The DB publishes the matrix exponential output directly.
