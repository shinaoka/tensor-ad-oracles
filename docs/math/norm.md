# Norm AD Notes

## Forward Definition

Norm families differentiate scalar functions of vectors or matrices, including:

- vector norms
- matrix norms
- condition-like quantities built from singular values

## Reverse Rule

For smooth vector norms away from zero, the cotangent is the normalized primal
direction scaled by the derivative of the scalar norm function.

Matrix norms split into two broad cases:

- entrywise norms, which behave like scalar reductions
- spectral norms, which differentiate through singular values or singular
  vectors

Condition-number-like quantities combine these ingredients and inherit the same
spectral caveats.

## Forward Rule

Forward mode applies the same scalar derivative to the primal norm expression,
with spectral norms using the tangent of the singular values.

## Numerical Notes

- Nonsmooth points, especially zero inputs, require subgradient conventions.
- The current DB excludes explicit upstream norm families that are classified as
  unsupported subgradient cases.

## Verification

- Compare norm values against direct primal evaluation.
- Compare derivatives against finite differences away from nonsmooth points.

## DB Families

<a id="family-norm-identity"></a>
### `norm/identity`

The DB publishes the chosen norm value directly.

<a id="family-matrix-norm-identity"></a>
### `matrix_norm/identity`

The DB publishes the matrix-norm observable directly.

<a id="family-vector-norm-identity"></a>
### `vector_norm/identity`

The DB publishes the vector-norm observable directly.

<a id="family-cond-identity"></a>
### `cond/identity`

The DB treats condition-number families as scalar spectral observables derived
from the same singular-value sensitivities.
