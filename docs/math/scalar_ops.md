# Scalar And Tensor Wrapper AD Notes

## Scope

This note covers the shared scalar, pointwise, reduction, and small tensor
wrapper rules that do not need one dedicated note file per operation.

## Complex Gradient Convention

For real-valued losses, the rules follow the conjugate-Wirtinger convention used
by PyTorch:

- VJP formulas conjugate the primal factors where required
- real inputs project complex intermediates back to the real domain

## Scalar Basis Rules

Let `g` be the output cotangent.

- `add`: $(dx_1, dx_2) = (g, g)$
- `sub`: $(dx_1, dx_2) = (g, -g)$
- `mul`: $(dx_1, dx_2) = (g \cdot \overline{x_2}, g \cdot \overline{x_1})$
- `div`: quotient rule with conjugated denominator factors
- `conj`: $dx = \overline{g}$
- `sqrt`: $dx = g / (2 \overline{\sqrt{x}})$
- `exp`: $dx = g \cdot \overline{\exp(x)}$
- `log`: $dx = g / \overline{x}$
- `atan2`: use the standard real partial derivatives over $a^2 + b^2$
- `pow`: differentiate both the base and exponent branches when both are
  differentiable

## Tensor Composite Wrappers

Many published DB families are thin wrappers around the scalar basis:

- unary analytic wrappers such as `sin`, `cos`, `tanh`, `asin`, `acos`, `atan`,
  `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- numerically specialized wrappers such as `log1p`, `expm1`, `rsqrt`, `sinc`
- reduction wrappers such as `sum`, `mean`, `var`, `std`, `nanmean`, `nansum`
- broadcasted binary wrappers such as `hypot`, `xlogy`, `maximum`, `minimum`

Some tensor families currently routed here are not purely scalar pointwise
operators, but they still share a compact differentiable contract that does not
yet justify a dedicated note:

- `cross`
- `diagonal`
- `matrix_power`
- `multi_dot`
- `vander`
- `vecdot`
- `householder_product`

## Reduction Rules

- `sum`: every element receives the same cotangent
- `mean`: every element receives the cotangent divided by the number of reduced
  entries
- `var`: differentiate through the centered residual $x - \operatorname{mean}(x)$
- `std`: combine the variance rule with the derivative of `sqrt`

## Published DB Families Using This Note

### Reflected and arithmetic wrappers

- <a id="op-__radd__"></a>`__radd__`
- <a id="op-__rdiv__"></a>`__rdiv__`
- <a id="op-__rmod__"></a>`__rmod__`
- <a id="op-__rmul__"></a>`__rmul__`
- <a id="op-__rpow__"></a>`__rpow__`
- <a id="op-__rsub__"></a>`__rsub__`
- <a id="op-add"></a>`add`
- <a id="op-div_no_rounding_mode"></a>`div_no_rounding_mode`
- <a id="op-float_power"></a>`float_power`
- <a id="op-hypot"></a>`hypot`
- <a id="op-max_binary"></a>`max_binary`
- <a id="op-maximum"></a>`maximum`
- <a id="op-min_binary"></a>`min_binary`
- <a id="op-minimum"></a>`minimum`
- <a id="op-mul"></a>`mul`
- <a id="op-pow"></a>`pow`
- <a id="op-rsub"></a>`rsub`
- <a id="op-sub"></a>`sub`
- <a id="op-true_divide"></a>`true_divide`
- <a id="op-xlogy"></a>`xlogy`

### Unary analytic, sign, rounding, and casts

- <a id="op-abs"></a>`abs`
- <a id="op-acos"></a>`acos`
- <a id="op-acosh"></a>`acosh`
- <a id="op-angle"></a>`angle`
- <a id="op-asin"></a>`asin`
- <a id="op-asinh"></a>`asinh`
- <a id="op-atan"></a>`atan`
- <a id="op-atan2"></a>`atan2`
- <a id="op-atanh"></a>`atanh`
- <a id="op-cdouble"></a>`cdouble`
- <a id="op-ceil"></a>`ceil`
- <a id="op-clamp_max"></a>`clamp_max`
- <a id="op-clamp_min"></a>`clamp_min`
- <a id="op-complex"></a>`complex`
- <a id="op-conj"></a>`conj`
- <a id="op-conj_physical"></a>`conj_physical`
- <a id="op-copysign"></a>`copysign`
- <a id="op-cos"></a>`cos`
- <a id="op-cosh"></a>`cosh`
- <a id="op-deg2rad"></a>`deg2rad`
- <a id="op-digamma"></a>`digamma`
- <a id="op-double"></a>`double`
- <a id="op-erf"></a>`erf`
- <a id="op-erfc"></a>`erfc`
- <a id="op-erfinv"></a>`erfinv`
- <a id="op-exp"></a>`exp`
- <a id="op-exp2"></a>`exp2`
- <a id="op-expm1"></a>`expm1`
- <a id="op-fill"></a>`fill`
- <a id="op-floor"></a>`floor`
- <a id="op-fmax"></a>`fmax`
- <a id="op-fmin"></a>`fmin`
- <a id="op-frac"></a>`frac`
- <a id="op-frexp"></a>`frexp`
- <a id="op-i0"></a>`i0`
- <a id="op-imag"></a>`imag`
- <a id="op-ldexp"></a>`ldexp`
- <a id="op-lgamma"></a>`lgamma`
- <a id="op-log"></a>`log`
- <a id="op-log10"></a>`log10`
- <a id="op-log1p"></a>`log1p`
- <a id="op-log2"></a>`log2`
- <a id="op-logaddexp"></a>`logaddexp`
- <a id="op-logit"></a>`logit`
- <a id="op-nan_to_num"></a>`nan_to_num`
- <a id="op-neg"></a>`neg`
- <a id="op-positive"></a>`positive`
- <a id="op-polar"></a>`polar`
- <a id="op-rad2deg"></a>`rad2deg`
- <a id="op-real"></a>`real`
- <a id="op-reciprocal"></a>`reciprocal`
- <a id="op-round"></a>`round`
- <a id="op-round_decimals_0"></a>`round_decimals_0`
- <a id="op-round_decimals_3"></a>`round_decimals_3`
- <a id="op-round_decimals_neg_3"></a>`round_decimals_neg_3`
- <a id="op-rsqrt"></a>`rsqrt`
- <a id="op-sgn"></a>`sgn`
- <a id="op-sigmoid"></a>`sigmoid`
- <a id="op-sign"></a>`sign`
- <a id="op-sin"></a>`sin`
- <a id="op-sinc"></a>`sinc`
- <a id="op-sinh"></a>`sinh`
- <a id="op-special_entr"></a>`special_entr`
- <a id="op-special_erfcx"></a>`special_erfcx`
- <a id="op-special_i0e"></a>`special_i0e`
- <a id="op-special_i1"></a>`special_i1`
- <a id="op-special_i1e"></a>`special_i1e`
- <a id="op-special_log_ndtr"></a>`special_log_ndtr`
- <a id="op-special_ndtr"></a>`special_ndtr`
- <a id="op-special_ndtri"></a>`special_ndtri`
- <a id="op-special_polygamma_special_polygamma_n_0"></a>`special_polygamma_special_polygamma_n_0`
- <a id="op-special_xlog1py"></a>`special_xlog1py`
- <a id="op-sqrt"></a>`sqrt`
- <a id="op-square"></a>`square`
- <a id="op-tan"></a>`tan`
- <a id="op-tanh"></a>`tanh`
- <a id="op-trunc"></a>`trunc`

### Reductions and statistics

- <a id="op-amax"></a>`amax`
- <a id="op-amin"></a>`amin`
- <a id="op-mean"></a>`mean`
- <a id="op-nanmean"></a>`nanmean`
- <a id="op-nansum"></a>`nansum`
- <a id="op-prod"></a>`prod`
- <a id="op-std"></a>`std`
- <a id="op-std_unbiased"></a>`std_unbiased`
- <a id="op-sum"></a>`sum`
- <a id="op-var"></a>`var`
- <a id="op-var_unbiased"></a>`var_unbiased`

### Neural-network functional wrappers

- <a id="op-nn_functional_celu"></a>`nn_functional_celu`
- <a id="op-nn_functional_elu"></a>`nn_functional_elu`
- <a id="op-nn_functional_hardshrink"></a>`nn_functional_hardshrink`
- <a id="op-nn_functional_hardsigmoid"></a>`nn_functional_hardsigmoid`
- <a id="op-nn_functional_hardtanh"></a>`nn_functional_hardtanh`
- <a id="op-nn_functional_logsigmoid"></a>`nn_functional_logsigmoid`
- <a id="op-nn_functional_mish"></a>`nn_functional_mish`
- <a id="op-nn_functional_prelu"></a>`nn_functional_prelu`
- <a id="op-nn_functional_relu"></a>`nn_functional_relu`
- <a id="op-nn_functional_relu6"></a>`nn_functional_relu6`
- <a id="op-nn_functional_rrelu"></a>`nn_functional_rrelu`
- <a id="op-nn_functional_selu"></a>`nn_functional_selu`
- <a id="op-nn_functional_silu"></a>`nn_functional_silu`
- <a id="op-nn_functional_softplus"></a>`nn_functional_softplus`
- <a id="op-nn_functional_softshrink"></a>`nn_functional_softshrink`
- <a id="op-nn_functional_softsign"></a>`nn_functional_softsign`
- <a id="op-nn_functional_tanhshrink"></a>`nn_functional_tanhshrink`
- <a id="op-nn_functional_threshold"></a>`nn_functional_threshold`

### Special-function parameter families

- <a id="op-mvlgamma_mvlgamma_p_1"></a>`mvlgamma_mvlgamma_p_1`
- <a id="op-mvlgamma_mvlgamma_p_3"></a>`mvlgamma_mvlgamma_p_3`
- <a id="op-mvlgamma_mvlgamma_p_5"></a>`mvlgamma_mvlgamma_p_5`
- <a id="op-polygamma_polygamma_n_0"></a>`polygamma_polygamma_n_0`
- <a id="op-polygamma_polygamma_n_1"></a>`polygamma_polygamma_n_1`
- <a id="op-polygamma_polygamma_n_2"></a>`polygamma_polygamma_n_2`
- <a id="op-polygamma_polygamma_n_3"></a>`polygamma_polygamma_n_3`
- <a id="op-polygamma_polygamma_n_4"></a>`polygamma_polygamma_n_4`

### Small tensor wrappers currently grouped here

- <a id="op-cross"></a>`cross`
- <a id="op-diagonal"></a>`diagonal`
- <a id="op-householder_product"></a>`householder_product`
- <a id="op-matrix_power"></a>`matrix_power`
- <a id="op-multi_dot"></a>`multi_dot`
- <a id="op-vander"></a>`vander`
- <a id="op-vecdot"></a>`vecdot`

## Notes On Future Splits

This shared note is intentionally broad in the first migration pass. Operations
that later grow heavier derivation detail can be split into dedicated note files
without changing the DB schema; only the central registry needs to move.
