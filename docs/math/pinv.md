# Pseudoinverse AD Notes

## Forward Definition

For the Moore-Penrose pseudoinverse

$$
A^+,
$$

the differential can be written in terms of projector corrections onto the row
and column spaces of $A$.

## Reverse Rule

The reverse rule combines three terms:

- a core inverse-like contribution
- a row-space projector correction
- a column-space projector correction

This is the stable extension of the inverse rule to rank-deficient or
rectangular matrices.

## Numerical Notes

- Small singular values dominate the sensitivity.
- Hermitian and singular variants differ in the primal convention, but the
  cotangent structure is still governed by the pseudoinverse projectors.

## Verification

- Moore-Penrose identities: check the standard projector equalities.
- Derivatives: compare JVP/VJP against finite differences.

## DB Families

<a id="family-pinv-identity"></a>
### `pinv/identity`

The DB publishes the pseudoinverse tensor directly.

<a id="family-pinv-hermitian-identity"></a>
### `pinv_hermitian/identity`

The DB uses the Hermitian pseudoinverse convention for the primal operator but
the same projector-based derivative structure.

<a id="family-pinv-singular-identity"></a>
### `pinv_singular/identity`

This family captures the singular-input regime explicitly while keeping the same
observable shape.
