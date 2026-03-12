# Inverse AD Notes

## Forward Definition

For

$$
Y = A^{-1},
$$

the differential follows from $A Y = I$:

$$
dY = -A^{-1}(dA)A^{-1}.
$$

## Reverse Rule

Given a cotangent $\bar Y$ on the inverse output,

$$
\bar A = -A^{-H}\bar Y A^{-H}.
$$

This identity is reused by explicit inverse families and by operators whose
derivative is most naturally expressed through an inverse solve.

## Verification

- Reconstruction: check $A A^{-1} \approx I$.
- Derivatives: compare JVP/VJP against finite differences on scalar functionals
  of the inverse.

## DB Families

<a id="family-inv-identity"></a>
### `inv/identity`

The DB publishes the inverse tensor directly.

<a id="family-inv-ex-identity"></a>
### `inv_ex/identity`

The DB validates the inverse output for the extended variant and treats status
metadata as nondifferentiable.

<a id="family-tensorinv-identity"></a>
### `tensorinv/identity`

The tensor inverse family is the index-reshaped analogue of the same inverse
rule.
