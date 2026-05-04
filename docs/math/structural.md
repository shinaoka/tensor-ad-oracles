# Structural AD Notes

## Conventions

These notes cover selection and shape-rearrangement primitives whose integer,
boolean, and axis arguments are metadata.  Tensor inputs are differentiated
under the real Frobenius inner product used by the rest of the corpus.  Unless
noted otherwise, Linearization and Transpose are written in raw-output-space
before any DB observable projection.

## Forward

The covered raw operators are:

- `where(condition, x, y)`, where condition is nondifferentiable metadata
- `cat(inputs, dim)`, where the axis is metadata
- `narrow(a, dim, start, length)`, where start indices and sizes are
  nondifferentiable metadata
- `clamp(x, lo, hi)`, where `x`, `lo`, and `hi` are differentiable tensors or
  scalar tensors

## Linearization

For `where`, with a fixed boolean mask $m$,

$$
\dot{z} = m \odot \dot{x} + (1-m) \odot \dot{y},
$$

using the same broadcasting as the primal operation.  The condition is
nondifferentiable.

For `cat`, if $z=\operatorname{cat}(x_1,\ldots,x_n;\operatorname{dim}=d)$, then

$$
\dot{z} = \operatorname{cat}(\dot{x}_1,\ldots,\dot{x}_n; d).
$$

The axis is metadata.

For `narrow`, the tangent is the same slice of the source tangent:

$$
\dot{z} = \operatorname{narrow}(\dot{a}, d, s, \ell).
$$

The dimension, start indices and sizes are nondifferentiable metadata.

For `clamp`, away from the switching surfaces $x=lo$ and $x=hi$,

$$
\dot{z}_i =
\begin{cases}
\dot{lo}_i, & x_i < lo_i, \\
\dot{x}_i, & lo_i < x_i < hi_i, \\
\dot{hi}_i, & x_i > hi_i.
\end{cases}
$$

Broadcasted bounds use the framework's broadcast reduction in reverse mode.

## JVP

The JVP is the linearization above evaluated at the chosen tangent inputs while
holding masks, axes, starts, and lengths fixed.

## Transpose

The transpose of `where` routes the output cotangent to `x` where the mask is
true and to `y` where the mask is false, summing over broadcasted dimensions.

The transpose of `cat` splits the cotangent along the concatenation axis and
returns one slice per differentiable input.

The transpose of `narrow` scatters the cotangent back into the selected source
region and writes zero outside that region.

The transpose of `clamp` routes cotangents to `lo`, `x`, or `hi` according to
the active region of each element, again summing over broadcasted dimensions.

## VJP (JAX convention)

JAX-style VJPs are the same transposes of the fixed-metadata linear maps.  The
metadata arguments are not part of the differentiable input tuple.

## VJP (PyTorch convention)

PyTorch autograd exposes the same tensor-input VJPs.  Boolean masks, axes, start
indices, and slice sizes are treated as nondifferentiable arguments.

## DB Families

<a id="family-where-identity"></a>
### `where/identity`

The DB publishes direct `torch.where(condition, x, y)` outputs.  Cases include
mixed masks, tensor/tensor branches, scalar `x` broadcast against tensor `y`,
and tensor `x` broadcast against scalar `y`.

<a id="family-cat-identity"></a>
### `cat/identity`

The DB publishes direct `torch.cat(inputs, dim)` outputs.  Cases cover
concatenation on leading, interior, and negative axes; `dim` is metadata.

<a id="family-narrow-identity"></a>
### `narrow/identity`

The DB publishes direct dynamic-slice outputs using `Tensor.narrow`.  Cases
cover positive and negative dimensions.  The dimension, start index, and slice
length are metadata.

<a id="family-clamp-identity"></a>
### `clamp/identity`

The DB publishes direct ternary clamp outputs for real tensors.  Cases cover
scalar bounds, tensor bounds, and broadcasted tensor bounds.  Boundary points
where $x=lo$ or $x=hi$ are nonsmooth and are not used as success probes.
