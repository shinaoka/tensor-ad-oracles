# Scalar And Tensor Wrapper AD Notes

## Scope

This note records shared scalar AD formulas together with the tensor-level
wrappers built from them.

## Complex Gradient Convention

For real-valued losses:

- gradients follow the conjugate-Wirtinger convention
- VJP formulas include complex conjugation where required
- real inputs project complex intermediates back to the real domain

## Scalar Basis Rules

Let `g` be the output cotangent, `x` the primal input, and `y = f(x)` the
primal output.

### Core arithmetic

- `add`: for `x_1 + \alpha x_2`, $(dx_1, dx_2) = (g, \overline{\alpha}\, g)$
- `sub`: for `x_1 - \alpha x_2`, $(dx_1, dx_2) = (g, -\overline{\alpha}\, g)$
- `mul`: $(dx_1, dx_2) = (g \cdot \overline{x_2}, g \cdot \overline{x_1})$
- `div`:
  - numerator path: $dx_1 = g / \overline{x_2}$
  - denominator path: $dx_2 = -g \cdot \overline{x_1 / x_2^2}$
  - integer-style rounding modes are treated as nondifferentiable branches

### Analytic unary wrappers

- `conj`: $dx = \overline{g}$
- `sqrt`: $dx = g / (2 \overline{\sqrt{x}})$
- `exp`: $dx = g \cdot \overline{y}$
- `log`: $dx = g / \overline{x}$
- `expm1`: $dx = g \cdot \overline{\exp(x)}$
- `log1p`: $dx = g / \overline{(1 + x)}$
- `sin`: $dx = g \cdot \overline{\cos(x)}$
- `cos`: $dx = -g \cdot \overline{\sin(x)}$
- `tanh`: $dx = g \cdot \overline{(1 - y^2)}$

### Parameterized wrappers

- `atan2`: for real inputs $(a, b)$,
  $da = g \, b / (a^2 + b^2)$ and $db = -g \, a / (a^2 + b^2)$, with the
  zero-denominator singularity masked by the implementation convention
- `powf`: for fixed exponent $p$, $dx = g \cdot \overline{(p x^{p-1})}$
- `powi`: integer-exponent specialization of `powf`
- `pow`:
  - base path: $dx = g \cdot \overline{a x^{a-1}}$
  - exponent path: $da = g \cdot \overline{x^a \log(x)}$

## Tensor-Composite Rules

Tensor-level wrappers built on top of the scalar basis include:

- pointwise unary analytic families
- broadcasted binary analytic families
- small tensor wrappers such as `cross`, `diagonal`, `matrix_power`,
  `multi_dot`, `vander`, `vecdot`, and `householder_product`

## Tensor Reduction Wrappers

### `sum_ad`

For a reduction over index set $\mathcal{I}$,

$$
y = \sum_{i \in \mathcal{I}} x_i
\quad \Longrightarrow \quad
\bar{x}_i = \bar{y}
$$

for every reduced element, with the cotangent broadcast back to the input
shape.

### `mean_ad`

If $n$ entries are reduced,

$$
y = \frac{1}{n} \sum_{i \in \mathcal{I}} x_i
\quad \Longrightarrow \quad
\bar{x}_i = \frac{\bar{y}}{n}.
$$

### `var_ad`

Let $\mu = \operatorname{mean}(x)$ over the reduced axes and let `correction`
denote the Bessel-style offset used by the variance operator. Then

$$
\operatorname{var}(x) = \frac{1}{n - \mathrm{correction}} \sum_i |x_i - \mu|^2,
$$

so away from the singular degrees-of-freedom boundary,

$$
\bar{x}
= \frac{2}{n - \mathrm{correction}} \, \bar{v} \, (x - \mu).
$$

At $n - \mathrm{correction} \le 0$, the operator is singular and the derivative
inherits the same NaN / infinity boundary behavior as the primal convention.

### `std_ad`

For $\sigma = \sqrt{v}$ with $v = \operatorname{var}(x)$,

$$
\bar{v} = \frac{\bar{\sigma}}{2 \sigma},
$$

masked at $\sigma = 0$, and then the variance rule is applied to propagate back
to $x$.

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
