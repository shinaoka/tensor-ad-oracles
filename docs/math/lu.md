# LU AD Notes

## Forward Definition

For an LU factorization with permutation matrix $P$,

$$
P A = L U,
$$

where $L$ is unit lower triangular and $U$ is upper triangular.

## Differential

Differentiate the factorization:

$$
P \, dA = dL \, U + L \, dU.
$$

The structure constraints are:

- $dL$ has zero diagonal and is strictly lower triangular
- $dU$ is upper triangular

## Reverse Rule

The reverse rule projects cotangents onto the lower- and upper-triangular
subspaces, solves the triangular adjoint systems, and then maps the result back
through the permutation.

The same projected identity is reused for:

- square LU
- tall and wide LU variants
- factor-only outputs such as `lu_factor`
- extended outputs such as `lu_factor_ex`

## Forward Rule

Forward mode computes $(dL, dU)$ by splitting $P dA$ into lower- and
upper-triangular components after left and right multiplication by the inverses
of $L$ and $U$.

## Verification

- Reconstruction: check $P A \approx L U$.
- Triangular structure: confirm strict lower / upper structure is preserved.
- Derivatives: compare against finite differences on scalar losses of the
  factor outputs.

## DB Families

<a id="family-lu-identity"></a>
### `lu/identity`

The DB publishes the differentiable LU outputs directly.

<a id="family-lu-factor-identity"></a>
### `lu_factor/identity`

The DB validates the factor tensor while treating pivot metadata as
nondifferentiable.

<a id="family-lu-factor-ex-identity"></a>
### `lu_factor_ex/identity`

The DB keeps the same derivative contract on the factor output for the extended
variant.
