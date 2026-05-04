# LU AD Notes

## Conventions

Unless noted otherwise, `Linearization` and `Transpose` are written for the
raw-output-space LU factorization before any DB observable projection. For
complex tensors, `Transpose` means the adjoint under the real Frobenius inner
product

$$
\langle X, Y \rangle_{\mathbb{R}} = \operatorname{Re}\operatorname{tr}(X^\dagger Y).
$$

## Forward

The raw operator is

$$
A \mapsto (P, L, U),
\qquad
P A = L U,
$$

where $P$ is discrete metadata and only $(L, U)$ are differentiated.

## Linearization

In the square case, define

$$
\dot{F} = L^{-1} P \dot{A} U^{-1}.
$$

Then

$$
\dot{L} = L \, \mathrm{tril}_-(\dot{F}),
\qquad
\dot{U} = \mathrm{triu}(\dot{F}) \, U.
$$

Wide and tall cases use the same lower/upper triangular split on the leading
square block, with the extra block corrections recorded later in this note.

## JVP

The JVP is the same block-triangular linearization, returned on the raw factor
outputs $(L, U)$ while keeping pivots as metadata.

## Transpose

In the square case, raw output cotangents $(\bar{L}, \bar{U})$ give

$$
\bar{F} = \mathrm{tril}_-(L^\dagger \bar{L}) + \mathrm{triu}(\bar{U} U^\dagger),
$$

$$
\bar{A} = P^T L^{-\dagger} \bar{F} U^{-\dagger}.
$$

Wide and tall cases use the same leading-block triangular adjoints, with the
explicit block formulas retained below.

## VJP (JAX convention)

JAX reads the same raw cotangent map on the differentiable factor outputs.

## VJP (PyTorch convention)

PyTorch uses the same block-triangular adjoint in `linalg_lu_backward`; pivots
and status metadata stay outside the differentiable surface.

## Forward Definition

$$
P A = L U, \qquad A \in \mathbb{C}^{M \times N}, \qquad K = \min(M, N)
$$

- $P \in \mathbb{R}^{M \times M}$ is a permutation matrix
- $L \in \mathbb{C}^{M \times K}$ is unit lower triangular
- $U \in \mathbb{C}^{K \times N}$ is upper triangular

The permutation is discrete metadata and is not differentiated.

## Notation

- $\mathrm{tril}_-(X)$: strictly lower-triangular part of $X$
- $\mathrm{triu}(X)$: upper-triangular part of $X$, including the diagonal

Since $L$ is unit lower triangular, its tangent and cotangent only use the
strictly lower-triangular part.

## Forward Rule

### Square case ($M = N$)

Given a tangent $\dot{A}$, define

$$
\dot{F} = L^{-1} P \dot{A} U^{-1}.
$$

Then

$$
\dot{L} = L \, \mathrm{tril}_-(\dot{F}),
\qquad
\dot{U} = \mathrm{triu}(\dot{F}) \, U.
$$

Differentiating $P A = L U$ gives

$$
P \dot{A} = \dot{L} U + L \dot{U},
$$

and after left-multiplying by $L^{-1}$ and right-multiplying by $U^{-1}$:

$$
\dot{F} = L^{-1} \dot{L} + \dot{U} U^{-1}.
$$

The two terms separate uniquely into strictly lower-triangular and
upper-triangular parts.

### Wide case ($M < N$)

Partition

$$
A = [A_1 \mid A_2],
\qquad
U = [U_1 \mid U_2],
$$

where $A_1, U_1 \in \mathbb{C}^{M \times M}$.
Define

$$
\dot{F} = L^{-1} P \dot{A}_1 U_1^{-1}.
$$

Then

$$
\dot{L} = L \, \mathrm{tril}_-(\dot{F}),
\qquad
\dot{U}_1 = \mathrm{triu}(\dot{F}) \, U_1,
$$

$$
\dot{U}_2 = L^{-1} P \dot{A}_2 - \mathrm{tril}_-(\dot{F}) U_2.
$$

### Tall case ($M > N$)

Partition

$$
L =
\begin{pmatrix}
L_1 \\
L_2
\end{pmatrix},
\qquad
P =
\begin{pmatrix}
P_1 \\
P_2
\end{pmatrix},
$$

with $L_1 \in \mathbb{C}^{N \times N}$ unit lower triangular and
$P_1 A = L_1 U$.
Define

$$
\dot{F} = L_1^{-1} P_1 \dot{A} U^{-1}.
$$

Then

$$
\dot{L}_1 = L_1 \, \mathrm{tril}_-(\dot{F}),
\qquad
\dot{U} = \mathrm{triu}(\dot{F}) \, U,
$$

$$
\dot{L}_2 = P_2 \dot{A} U^{-1} - L_2 \, \mathrm{triu}(\dot{F}).
$$

## Reverse Rule

Given cotangents $\bar{L}$ and $\bar{U}$ of a real scalar loss $\ell$:

### Square case ($M = N$)

Define

$$
\bar{F} = \mathrm{tril}_-(L^\dagger \bar{L}) + \mathrm{triu}(\bar{U} U^\dagger).
$$

Then

$$
\bar{A} = P^T L^{-\dagger} \bar{F} U^{-\dagger}.
$$

This is the adjoint of the triangular split in the forward rule.

### Wide case ($M < N$)

Partition $\bar{U} = [\bar{U}_1 \mid \bar{U}_2]$ and define

$$
\bar{H}_1 =
\left(
\mathrm{tril}_-(L^\dagger \bar{L} - \bar{U}_2 U_2^\dagger)
+ \mathrm{triu}(\bar{U}_1 U_1^\dagger)
\right) U_1^{-\dagger},
$$

$$
\bar{H}_2 = \bar{U}_2.
$$

Then

$$
\bar{A} = P^T L^{-\dagger} [\bar{H}_1 \mid \bar{H}_2].
$$

### Tall case ($M > N$)

Partition

$$
\bar{L} =
\begin{pmatrix}
\bar{L}_1 \\
\bar{L}_2
\end{pmatrix}
$$

and define

$$
\bar{H}_1 =
L_1^{-\dagger}
\left(
\mathrm{tril}_-(L_1^\dagger \bar{L}_1)
+ \mathrm{triu}(\bar{U} U^\dagger - L_2^\dagger \bar{L}_2)
\right),
$$

$$
\bar{H}_2 = \bar{L}_2.
$$

Then

$$
\bar{A} =
P^T
\begin{pmatrix}
\bar{H}_1 \\
\bar{H}_2
\end{pmatrix}
U^{-\dagger}.
$$

All appearances of $L^{-1}X$, $XU^{-1}$, $L^{-\dagger}X$, and $XU^{-\dagger}$
should be interpreted as triangular solves rather than as explicit inverse
formation.

## Full-Pivot Extension

The full-pivot raw operator is

$$
A \mapsto (P, Q, L, U),
\qquad
P A Q = L U,
$$

or, in plain text, P A Q = L U.  The row permutation $P$, column permutation
$Q$, parity, and status outputs are discrete metadata.  Only the factor outputs
$(L, U)$ are differentiated.

For any open region where the selected row and column pivots are fixed, define

$$
B = P A Q,
\qquad
\dot{B} = P \dot{A} Q.
$$

The partial-pivot formulas above apply unchanged to the no-pivot factorization
$B = L U$.  Equivalently, the square full-pivot linearization is

$$
\dot{F} = L^{-1} P \dot{A} Q U^{-1},
$$

$$
\dot{L} = L \, \mathrm{tril}_-(\dot{F}),
\qquad
\dot{U} = \mathrm{triu}(\dot{F}) \, U.
$$

The wide and tall cases use the same block formulas as the partial-pivot note
after replacing $P\dot{A}$ by $P\dot{A}Q$ and partitioning the columns or rows
of $B = P A Q$ rather than of $A$.

For reverse mode, first compute the same raw cotangent $\bar{B}$ for the
factorization $B = L U$.  In the square case,

$$
\bar{B} =
L^{-\dagger}
\left(
\mathrm{tril}_-(L^\dagger \bar{L})
+ \mathrm{triu}(\bar{U} U^\dagger)
\right)
U^{-\dagger},
$$

and the input cotangent is

$$
\bar{A} = P^T \bar{B} Q^T.
$$

The wide and tall reverse rules are the partial-pivot block adjoints interpreted
as cotangents for $B$, followed by the same permutation pullback
$\bar{A}=P^T\bar{B}Q^T$.

## Verification

### Forward reconstruction

$$
\|P A - L U\|_F < \varepsilon
$$

with $L$ unit lower triangular and $U$ upper triangular.

For the full-pivot family, the reconstruction check is

$$
\|P A Q - L U\|_F < \varepsilon.
$$

### Backward checks

Representative scalar tests:

- $dL$ only: $f(A) = \operatorname{Re}(v^\dagger \operatorname{op} \, v)$ with
  $v = L_{:,1}$
- $dU$ only: $f(A) = \operatorname{Re}(v^\dagger \operatorname{op} \, v)$ with
  $v = U_{1,:}$
- mixed: $f(A) = \operatorname{Re}(L_{1,1}^* U_{1,1})$

where $\operatorname{op}$ is a random Hermitian matrix independent of $A$.

## References

1. S. Axen, "Differentiating the LU decomposition," 2021.
2. M. Seeger et al., "Auto-Differentiating Linear Algebra," 2018.

## DB Families

<a id="family-lu-identity"></a>
### `lu/identity`

The DB publishes the differentiable $(P, L, U)$ decomposition, with the
permutation treated as nondifferentiable metadata.

<a id="family-lu-factor-identity"></a>
### `lu_factor/identity`

The DB validates the packed factor tensor while treating pivots as
nondifferentiable metadata.

<a id="family-lu-factor-ex-identity"></a>
### `lu_factor_ex/identity`

The extended factorization uses the same derivative contract on the factor
tensor; status outputs remain metadata.

<a id="family-full-pivot-lu-identity"></a>
### `full_pivot_lu/identity`

The DB publishes the differentiable $(L, U)$ factors for the full-pivot
contract while treating row pivots, column pivots, parity, and status as
nondifferentiable metadata.  Cases cover square, wide, and tall matrices under
fixed-pivot finite-difference probes because pinned PyTorch does not expose an
upstream full-pivot LU OpInfo.
