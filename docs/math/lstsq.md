# Least Squares AD Notes

## Conventions

Unless noted otherwise, `Linearization` and `Transpose` are written for the
raw-output-space least-squares solution map. For complex tensors, `Transpose`
means the adjoint under the real Frobenius inner product

$$
\langle X, Y \rangle_{\mathbb{R}} = \operatorname{Re}\operatorname{tr}(X^\dagger Y).
$$

## Forward

The raw operator is

$$
(A, b) \mapsto x = \arg\min_x \|Ax - b\|_2^2,
\qquad
x = R^{-1}Q^\dagger b
$$

for the thin QR factorization $A = Q R$.

## Linearization

With residual $r = b - A x$, the raw-output-space linearization is

$$
dx = (A^\dagger A)^{-1}(A^\dagger db + dA^\dagger r - A^\dagger dA\,x).
$$

## JVP

The JVP is that same linearization evaluated at $(dA, db)$.

## Transpose

For a raw output cotangent $\bar{x}$, define

$$
y = R^{-\dagger}\bar{x},
\qquad
z = R^{-1}y.
$$

Then

$$
\bar{b} = Q y,
\qquad
\bar{A} = r z^\dagger - \bar{b} x^\dagger.
$$

## VJP (JAX convention)

JAX reads the same raw transpose map on the least-squares solution output.

## VJP (PyTorch convention)

PyTorch currently routes the solution term through `pinv`-style helpers and
adds the residual correction separately, but the resulting raw cotangent map is
the same.

## Forward Definition

For the least-squares problem

$$
x = \arg\min_x \|A x - b\|_2^2,
\qquad
A \in \mathbb{C}^{M \times N},
\qquad
b \in \mathbb{C}^{M},
\qquad
M \geq N,
$$

the solution satisfies the normal equations

$$
A^\dagger A x = A^\dagger b.
$$

Equivalently, if $A = Q R$ is a thin QR decomposition, then

$$
x = R^{-1} Q^\dagger b.
$$

The same formulas extend to multiple right-hand sides by replacing $b$ and $x$
with matrices.

## Reverse Rule

Given a cotangent $\bar{x}$, compute cotangents for $A$ and $b$.

### Step 1: QR decomposition

$$
A = Q R,
$$

where $Q^\dagger Q = I_N$ and $R$ is upper triangular.

### Step 2: Two triangular solves

$$
y = R^{-\dagger} \bar{x},
\qquad
z = R^{-1} y.
$$

Equivalently,

$$
z = (A^\dagger A)^{-1} \bar{x}.
$$

### Step 3: Residual and cotangents

Let the residual be written explicitly as `r = b - Ax`:

$$
r = b - A x.
$$

Then

$$
\bar{b} = Q y,
$$

$$
\bar{A} = r z^\dagger - \bar{b} x^\dagger.
$$

### Complete formulas

$$
\bar{b} = Q R^{-\dagger} \bar{x},
$$

$$
\bar{A} =
(b - A x) (R^{-1} R^{-\dagger} \bar{x})^\dagger
- (Q R^{-\dagger} \bar{x}) x^\dagger.
$$

## Derivation Sketch

Write the residual as $r = b - A x$. The optimality condition is

$$
A^\dagger r = 0.
$$

Differentiating the normal equations gives

$$
A^\dagger A \, dx
= A^\dagger db + dA^\dagger r - A^\dagger dA \, x.
$$

Therefore

$$
dx = (A^\dagger A)^{-1}(A^\dagger db + dA^\dagger r - A^\dagger dA \, x).
$$

Let $z = (A^\dagger A)^{-1} \bar{x}$. Then

$$
\delta \ell
= \langle \bar{x}, dx \rangle
= \langle A z, db \rangle + \langle r z^\dagger, dA \rangle
- \langle A z \, x^\dagger, dA \rangle.
$$

Since $A z = Q y$, the formulas for $\bar{b}$ and $\bar{A}$ follow.

## Implementation Correspondence

- `tenferro-rs/docs/AD/lstsq.md` uses the QR-based derivation above, which makes
  the residual correction term explicit.
- PyTorch's `linalg_lstsq_solution_jvp` and `linalg_lstsq_backward` currently
  route the solution term through `pinv_jvp` / `pinv_backward`, while the
  residual term is added directly. The resulting adjoint matches the same
  least-squares geometry.
- The residual JVP in PyTorch uses Danskin's theorem, treating the minimizer as
  fixed when differentiating the residual objective itself.

## Verification

### Forward check

$$
A^\dagger(A x - b) \approx 0.
$$

### Backward checks

- fix $b$ and perturb $A$
- fix $A$ and perturb $b$
- compare JVP/VJP against finite differences on scalar losses built from $x$

## References

1. BackwardsLinalg.jl, `src/lstsq.jl`.
2. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode automatic differentiation," 2008.

## DB Families

<a id="family-lstsq-grad-oriented-identity"></a>
### `lstsq_grad_oriented/identity`

The DB publishes the differentiable least-squares outputs for the
gradient-oriented upstream variant.
