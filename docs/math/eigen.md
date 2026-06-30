# Hermitian Eigen AD Notes

## Conventions

Unless noted otherwise, `Linearization` and `Transpose` are written for the
raw-output-space Hermitian eigendecomposition before any DB observable such as
`values_vectors_abs` is applied. For complex tensors, `Transpose` means the
adjoint under the real Frobenius inner product

$$
\langle X, Y \rangle_{\mathbb{R}} = \operatorname{Re}\operatorname{tr}(X^\dagger Y).
$$

## Forward

The raw operator is

$$
A \mapsto (E, U),
\qquad
A = U \operatorname{diag}(E) U^\dagger,
\qquad
A = A^\dagger.
$$

## Linearization

With the stabilized inverse-gap matrix $F$ defined later in the note,

$$
dE = \operatorname{diag}(U^\dagger dA U),
$$

$$
dU = U \left(F \odot (U^\dagger dA U - \operatorname{diag}(dE))\right),
$$

with the skew-Hermitian gauge projected away.

## JVP

The JVP is the same linearization evaluated on the Hermitian tangent $dA$,
returning $(dE, dU)$.

## Transpose

For raw output cotangents $(\bar{E}, \bar{U})$, the transpose map uses the
inner matrix

$$
D =
\frac{1}{2}
\left(
F \odot (\bar{U}^\dagger U)
+ (F \odot (\bar{U}^\dagger U))^\dagger
\right)
+ \operatorname{diag}(\bar{E}),
$$

and returns

$$
\bar{A} = U D U^\dagger.
$$

## VJP (JAX convention)

JAX reads the same raw transpose on the Hermitian eigendecomposition outputs.

## VJP (PyTorch convention)

PyTorch routes the Hermitian case through the same raw rule as a structured
specialization of the general eigendecomposition kernels.

## Forward Definition

$$
A = U \operatorname{diag}(E) U^\dagger,
\qquad
A \in \mathbb{C}^{N \times N},
\qquad
A = A^\dagger
$$

- $U \in \mathbb{C}^{N \times N}$ is unitary
- $E \in \mathbb{R}^N$ contains the real eigenvalues

## Reverse Rule

Given cotangents $\bar{E}$ and $\bar{U}$, compute a Hermitian cotangent
$\bar{A}$.

### Step 1: Build the $F$ matrix

$$
F_{ij} =
\begin{cases}
\dfrac{E_i - E_j}{(E_i - E_j)^2 + \eta}
\approx \dfrac{1}{E_i - E_j}, & i \neq j, \\
0, & i = j.
\end{cases}
$$

Regularization $\eta > 0$ avoids division by zero for degenerate eigenvalues.

### Step 2: Build the inner matrix $D$

$$
D =
\frac{1}{2}
\left(
F \odot (\bar{U}^\dagger U)
+ (F \odot (\bar{U}^\dagger U))^\dagger
\right)
+ \operatorname{diag}(\bar{E}).
$$

The symmetrization ensures that $D$ is Hermitian.

### Step 3: Conjugate back to the input basis

$$
\bar{A} = U D U^\dagger.
$$

The diagonal imaginary gauge of $U^\dagger dU$ drops out after symmetrization,
so the final cotangent stays inside the Hermitian tangent space.

## Derivation Sketch

Differentiate

$$
A U = U \operatorname{diag}(E)
$$

to obtain

$$
dA \, U + A \, dU = dU \, \operatorname{diag}(E) + U \, \operatorname{diag}(dE).
$$

Left-multiplying by $U^\dagger$ yields

$$
U^\dagger dA \, U =
U^\dagger dU \, \operatorname{diag}(E)
- \operatorname{diag}(E) U^\dagger dU
+ \operatorname{diag}(dE).
$$

If $\Omega = U^\dagger dU$, then $\Omega$ is skew-Hermitian, the diagonal of
$U^\dagger dA U$ gives $dE$, and the off-diagonal entries are divided by
$E_j - E_i$. Applying the adjoint of that split yields the formula above.

## Forward Rule

The Hermitian forward rule is the simplification of the general eigendecomposition
JVP:

$$
dE = \operatorname{diag}(U^\dagger dA U),
$$

$$
dU = U \left(F \odot (U^\dagger dA U - \operatorname{diag}(dE))\right),
$$

with the understanding that the skew-Hermitian gauge is projected away.

## Eigenvalue-Sum Scalarization

The `grad_sum_eigh_jvp` and `grad_sum_eigh_vjp` benchmark scalarization uses
only the eigenvalue output. The SPD benchmark input is the real symmetric
positive definite subcase of this Hermitian domain:

$$
\phi(A) = \sum_i E_i.
$$

For Hermitian inputs,

$$
\phi(A) = \operatorname{tr}(\operatorname{diag}(E))
= \operatorname{tr}(A),
$$

so the scalarized JVP along a Hermitian tangent $\dot{A}$ is

$$
\dot{\phi}
= \sum_i \dot{E}_i
= \operatorname{tr}(U^\dagger \dot{A} U)
= \operatorname{tr}(\dot{A}).
$$

Equivalently, substituting the raw linearization gives

$$
\dot{E} = \operatorname{diag}(U^\dagger \dot{A} U),
\qquad
\dot{\phi} = \mathbf{1}^\mathsf{T}\dot{E}.
$$

For the scalarized VJP with output cotangent $\bar{\phi}$, the raw output
cotangents are

$$
\bar{E} = \bar{\phi}\,\mathbf{1},
\qquad
\bar{U} = 0.
$$

The eigenvector cotangent path therefore drops out, and the transpose reduces
to the eigenvalue-only term

$$
\bar{A}
= U \operatorname{diag}(\bar{E}) U^\dagger
= \bar{\phi}\, U I U^\dagger
= \bar{\phi}\, I.
$$

For real symmetric inputs this is the same formula with $U^\dagger = U^T$.
For complex Hermitian inputs, the adjoint convention is essential: the fast
eigenvalue-only path is $U \operatorname{diag}(\bar{E}) U^\dagger$, not
$U \operatorname{diag}(\bar{E}) U^T$. The inverse-gap matrix $F$ and the
eigenvector cotangent branch are not needed in this specialization.

## Verification

### Forward reconstruction

$$
\|A - U \operatorname{diag}(E) U^\dagger\|_F < \varepsilon,
\qquad
U^\dagger U \approx I.
$$

### Backward checks

- eigenvalues only: $f(A) = \sum_i E_i$
- eigenvectors only: scalar losses on a column of $U$
- compare JVP/VJP against finite differences on Hermitian perturbations

## References

1. M. Seeger et al., "Auto-Differentiating Linear Algebra," 2018.
2. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode automatic differentiation," 2008.

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
