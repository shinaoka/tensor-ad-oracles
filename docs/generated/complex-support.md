# Complex Support Report

Generated from the checked-in complex-support ledger and the published `cases/` tree.

## Summary

- Total tracked families: 176
- Ready for downstream: 103
- Unsupported: 73
- Pending note review: 0
- Pending DB coverage: 0

## Full Ledger

| op | family | note status | db status | complex published dtypes | unsupported reason | ready |
| --- | --- | --- | --- | --- | --- | --- |
| __radd__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| __rdiv__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| __rmod__ | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| __rmul__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| __rpow__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| __rsub__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| abs | identity | reviewed | covered | complex128, complex64 | - | yes |
| acos | identity | reviewed | covered | complex128, complex64 | - | yes |
| acosh | identity | reviewed | covered | complex128, complex64 | - | yes |
| add | identity | reviewed | covered | complex128, complex64 | - | yes |
| amax | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| amin | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| angle | identity | reviewed | covered | complex128, complex64 | - | yes |
| asin | identity | reviewed | covered | complex128, complex64 | - | yes |
| asinh | identity | reviewed | covered | complex128, complex64 | - | yes |
| atan | identity | reviewed | covered | complex128, complex64 | - | yes |
| atan2 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| atanh | identity | reviewed | covered | complex128, complex64 | - | yes |
| cdouble | identity | reviewed | covered | complex128, complex64 | - | yes |
| ceil | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| cholesky | identity | reviewed | covered | complex128, complex64 | - | yes |
| cholesky_ex | identity | not_required | covered | complex128, complex64 | - | yes |
| clamp_max | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| clamp_min | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| complex | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| cond | identity | not_required | covered | complex128, complex64 | - | yes |
| conj | identity | reviewed | covered | complex128, complex64 | - | yes |
| conj_physical | identity | reviewed | covered | complex128, complex64 | - | yes |
| copysign | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| cos | identity | reviewed | covered | complex128, complex64 | - | yes |
| cosh | identity | reviewed | covered | complex128, complex64 | - | yes |
| cross | identity | reviewed | covered | complex128, complex64 | - | yes |
| deg2rad | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| det | identity | reviewed | covered | complex128, complex64 | - | yes |
| diagonal | identity | reviewed | covered | complex128, complex64 | - | yes |
| digamma | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| div_no_rounding_mode | identity | reviewed | covered | complex128, complex64 | - | yes |
| double | identity | reviewed | covered | complex128, complex64 | - | yes |
| eig | values_vectors_abs | reviewed | covered | complex128, complex64 | - | yes |
| eigh | gauge_ill_defined | reviewed | covered | complex128 | - | yes |
| eigh | values_vectors_abs | reviewed | covered | complex128, complex64 | - | yes |
| eigvals | identity | not_required | covered | complex128, complex64 | - | yes |
| eigvalsh | identity | not_required | covered | complex128, complex64 | - | yes |
| erf | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| erfc | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| erfinv | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| exp | identity | reviewed | covered | complex128, complex64 | - | yes |
| exp2 | identity | reviewed | covered | complex128, complex64 | - | yes |
| expm1 | identity | reviewed | covered | complex128, complex64 | - | yes |
| fill | identity | reviewed | covered | complex128, complex64 | - | yes |
| float_power | identity | reviewed | covered | complex128, complex64 | - | yes |
| floor | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| fmax | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| fmin | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| frac | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| frexp | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| householder_product | identity | reviewed | covered | complex128, complex64 | - | yes |
| hypot | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| i0 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| imag | identity | reviewed | covered | complex128, complex64 | - | yes |
| inv | identity | reviewed | covered | complex128, complex64 | - | yes |
| inv_ex | identity | not_required | covered | complex128, complex64 | - | yes |
| ldexp | identity | reviewed | covered | complex128, complex64 | - | yes |
| lgamma | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| log | identity | reviewed | covered | complex128, complex64 | - | yes |
| log10 | identity | reviewed | covered | complex128, complex64 | - | yes |
| log1p | identity | reviewed | covered | complex128, complex64 | - | yes |
| log2 | identity | reviewed | covered | complex128, complex64 | - | yes |
| logaddexp | identity | reviewed | covered | complex128, complex64 | - | yes |
| logit | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| lstsq_grad_oriented | identity | reviewed | covered | complex128, complex64 | - | yes |
| lu | identity | reviewed | covered | complex128, complex64 | - | yes |
| lu_factor | identity | not_required | covered | complex128, complex64 | - | yes |
| lu_factor_ex | identity | not_required | covered | complex128, complex64 | - | yes |
| lu_solve | identity | not_required | covered | complex128, complex64 | - | yes |
| matrix_norm | identity | not_required | covered | complex128, complex64 | - | yes |
| matrix_power | identity | reviewed | covered | complex128, complex64 | - | yes |
| max_binary | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| maximum | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| mean | identity | reviewed | covered | complex128, complex64 | - | yes |
| min_binary | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| minimum | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| mul | identity | reviewed | covered | complex128, complex64 | - | yes |
| multi_dot | identity | reviewed | covered | complex128, complex64 | - | yes |
| mvlgamma_mvlgamma_p_1 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| mvlgamma_mvlgamma_p_3 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| mvlgamma_mvlgamma_p_5 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nan_to_num | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nanmean | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nansum | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| neg | identity | reviewed | covered | complex128, complex64 | - | yes |
| nn_functional_celu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_elu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_hardshrink | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_hardsigmoid | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_hardtanh | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_logsigmoid | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_mish | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_prelu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_relu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_relu6 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_rrelu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_selu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_silu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_softplus | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_softshrink | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_softsign | identity | reviewed | covered | complex128, complex64 | - | yes |
| nn_functional_tanhshrink | identity | reviewed | covered | complex128, complex64 | - | yes |
| nn_functional_threshold | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| norm | identity | reviewed | covered | complex128, complex64 | - | yes |
| pinv | identity | reviewed | covered | complex128, complex64 | - | yes |
| pinv_hermitian | identity | not_required | covered | complex128, complex64 | - | yes |
| pinv_singular | identity | not_required | covered | complex128, complex64 | - | yes |
| polar | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_0 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_1 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_2 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_3 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_4 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| positive | identity | reviewed | covered | complex128, complex64 | - | yes |
| pow | identity | reviewed | covered | complex128, complex64 | - | yes |
| prod | identity | reviewed | covered | complex128, complex64 | - | yes |
| qr | identity | reviewed | covered | complex128, complex64 | - | yes |
| rad2deg | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| real | identity | reviewed | covered | complex128, complex64 | - | yes |
| reciprocal | identity | reviewed | covered | complex128, complex64 | - | yes |
| round | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| round_decimals_0 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| round_decimals_3 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| round_decimals_neg_3 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| rsqrt | identity | reviewed | covered | complex128, complex64 | - | yes |
| rsub | identity | reviewed | covered | complex128, complex64 | - | yes |
| sgn | identity | reviewed | covered | complex128, complex64 | - | yes |
| sigmoid | identity | reviewed | covered | complex128, complex64 | - | yes |
| sign | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| sin | identity | reviewed | covered | complex128, complex64 | - | yes |
| sinc | identity | reviewed | covered | complex128, complex64 | - | yes |
| sinh | identity | reviewed | covered | complex128, complex64 | - | yes |
| slogdet | identity | reviewed | covered | complex128, complex64 | - | yes |
| solve | identity | reviewed | covered | complex128, complex64 | - | yes |
| solve_ex | identity | not_required | covered | complex128, complex64 | - | yes |
| solve_triangular | identity | reviewed | covered | complex128, complex64 | - | yes |
| special_entr | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_erfcx | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_i0e | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_i1 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_i1e | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_log_ndtr | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_ndtr | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_ndtri | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_polygamma_special_polygamma_n_0 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_xlog1py | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| sqrt | identity | reviewed | covered | complex128, complex64 | - | yes |
| square | identity | reviewed | covered | complex128, complex64 | - | yes |
| std | identity | reviewed | covered | complex128, complex64 | - | yes |
| std_unbiased | identity | reviewed | covered | complex128, complex64 | - | yes |
| sub | identity | reviewed | covered | complex128, complex64 | - | yes |
| sum | identity | reviewed | covered | complex128, complex64 | - | yes |
| svd | gauge_ill_defined | reviewed | covered | complex128 | - | yes |
| svd | s | reviewed | covered | complex128, complex64 | - | yes |
| svd | u_abs | reviewed | covered | complex128, complex64 | - | yes |
| svd | uvh_product | reviewed | covered | complex128, complex64 | - | yes |
| svd | vh_abs | reviewed | covered | complex128, complex64 | - | yes |
| svdvals | identity | not_required | covered | complex128, complex64 | - | yes |
| tan | identity | reviewed | covered | complex128, complex64 | - | yes |
| tanh | identity | reviewed | covered | complex128, complex64 | - | yes |
| tensorinv | identity | reviewed | covered | complex128, complex64 | - | yes |
| tensorsolve | identity | reviewed | covered | complex128, complex64 | - | yes |
| true_divide | identity | reviewed | covered | complex128, complex64 | - | yes |
| trunc | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| vander | identity | reviewed | covered | complex128, complex64 | - | yes |
| var | identity | reviewed | covered | complex128, complex64 | - | yes |
| var_unbiased | identity | reviewed | covered | complex128, complex64 | - | yes |
| vecdot | identity | reviewed | covered | complex128, complex64 | - | yes |
| vector_norm | identity | not_required | covered | complex128, complex64 | - | yes |
| xlogy | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |

## Ready For Downstream

| op | family | note status | db status | complex published dtypes | unsupported reason | ready |
| --- | --- | --- | --- | --- | --- | --- |
| __radd__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| __rdiv__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| __rmul__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| __rpow__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| __rsub__ | identity | reviewed | covered | complex128, complex64 | - | yes |
| abs | identity | reviewed | covered | complex128, complex64 | - | yes |
| acos | identity | reviewed | covered | complex128, complex64 | - | yes |
| acosh | identity | reviewed | covered | complex128, complex64 | - | yes |
| add | identity | reviewed | covered | complex128, complex64 | - | yes |
| angle | identity | reviewed | covered | complex128, complex64 | - | yes |
| asin | identity | reviewed | covered | complex128, complex64 | - | yes |
| asinh | identity | reviewed | covered | complex128, complex64 | - | yes |
| atan | identity | reviewed | covered | complex128, complex64 | - | yes |
| atanh | identity | reviewed | covered | complex128, complex64 | - | yes |
| cdouble | identity | reviewed | covered | complex128, complex64 | - | yes |
| cholesky | identity | reviewed | covered | complex128, complex64 | - | yes |
| cholesky_ex | identity | not_required | covered | complex128, complex64 | - | yes |
| cond | identity | not_required | covered | complex128, complex64 | - | yes |
| conj | identity | reviewed | covered | complex128, complex64 | - | yes |
| conj_physical | identity | reviewed | covered | complex128, complex64 | - | yes |
| cos | identity | reviewed | covered | complex128, complex64 | - | yes |
| cosh | identity | reviewed | covered | complex128, complex64 | - | yes |
| cross | identity | reviewed | covered | complex128, complex64 | - | yes |
| det | identity | reviewed | covered | complex128, complex64 | - | yes |
| diagonal | identity | reviewed | covered | complex128, complex64 | - | yes |
| div_no_rounding_mode | identity | reviewed | covered | complex128, complex64 | - | yes |
| double | identity | reviewed | covered | complex128, complex64 | - | yes |
| eig | values_vectors_abs | reviewed | covered | complex128, complex64 | - | yes |
| eigh | gauge_ill_defined | reviewed | covered | complex128 | - | yes |
| eigh | values_vectors_abs | reviewed | covered | complex128, complex64 | - | yes |
| eigvals | identity | not_required | covered | complex128, complex64 | - | yes |
| eigvalsh | identity | not_required | covered | complex128, complex64 | - | yes |
| exp | identity | reviewed | covered | complex128, complex64 | - | yes |
| exp2 | identity | reviewed | covered | complex128, complex64 | - | yes |
| expm1 | identity | reviewed | covered | complex128, complex64 | - | yes |
| fill | identity | reviewed | covered | complex128, complex64 | - | yes |
| float_power | identity | reviewed | covered | complex128, complex64 | - | yes |
| householder_product | identity | reviewed | covered | complex128, complex64 | - | yes |
| imag | identity | reviewed | covered | complex128, complex64 | - | yes |
| inv | identity | reviewed | covered | complex128, complex64 | - | yes |
| inv_ex | identity | not_required | covered | complex128, complex64 | - | yes |
| ldexp | identity | reviewed | covered | complex128, complex64 | - | yes |
| log | identity | reviewed | covered | complex128, complex64 | - | yes |
| log10 | identity | reviewed | covered | complex128, complex64 | - | yes |
| log1p | identity | reviewed | covered | complex128, complex64 | - | yes |
| log2 | identity | reviewed | covered | complex128, complex64 | - | yes |
| logaddexp | identity | reviewed | covered | complex128, complex64 | - | yes |
| lstsq_grad_oriented | identity | reviewed | covered | complex128, complex64 | - | yes |
| lu | identity | reviewed | covered | complex128, complex64 | - | yes |
| lu_factor | identity | not_required | covered | complex128, complex64 | - | yes |
| lu_factor_ex | identity | not_required | covered | complex128, complex64 | - | yes |
| lu_solve | identity | not_required | covered | complex128, complex64 | - | yes |
| matrix_norm | identity | not_required | covered | complex128, complex64 | - | yes |
| matrix_power | identity | reviewed | covered | complex128, complex64 | - | yes |
| mean | identity | reviewed | covered | complex128, complex64 | - | yes |
| mul | identity | reviewed | covered | complex128, complex64 | - | yes |
| multi_dot | identity | reviewed | covered | complex128, complex64 | - | yes |
| neg | identity | reviewed | covered | complex128, complex64 | - | yes |
| nn_functional_softsign | identity | reviewed | covered | complex128, complex64 | - | yes |
| nn_functional_tanhshrink | identity | reviewed | covered | complex128, complex64 | - | yes |
| norm | identity | reviewed | covered | complex128, complex64 | - | yes |
| pinv | identity | reviewed | covered | complex128, complex64 | - | yes |
| pinv_hermitian | identity | not_required | covered | complex128, complex64 | - | yes |
| pinv_singular | identity | not_required | covered | complex128, complex64 | - | yes |
| positive | identity | reviewed | covered | complex128, complex64 | - | yes |
| pow | identity | reviewed | covered | complex128, complex64 | - | yes |
| prod | identity | reviewed | covered | complex128, complex64 | - | yes |
| qr | identity | reviewed | covered | complex128, complex64 | - | yes |
| real | identity | reviewed | covered | complex128, complex64 | - | yes |
| reciprocal | identity | reviewed | covered | complex128, complex64 | - | yes |
| rsqrt | identity | reviewed | covered | complex128, complex64 | - | yes |
| rsub | identity | reviewed | covered | complex128, complex64 | - | yes |
| sgn | identity | reviewed | covered | complex128, complex64 | - | yes |
| sigmoid | identity | reviewed | covered | complex128, complex64 | - | yes |
| sin | identity | reviewed | covered | complex128, complex64 | - | yes |
| sinc | identity | reviewed | covered | complex128, complex64 | - | yes |
| sinh | identity | reviewed | covered | complex128, complex64 | - | yes |
| slogdet | identity | reviewed | covered | complex128, complex64 | - | yes |
| solve | identity | reviewed | covered | complex128, complex64 | - | yes |
| solve_ex | identity | not_required | covered | complex128, complex64 | - | yes |
| solve_triangular | identity | reviewed | covered | complex128, complex64 | - | yes |
| sqrt | identity | reviewed | covered | complex128, complex64 | - | yes |
| square | identity | reviewed | covered | complex128, complex64 | - | yes |
| std | identity | reviewed | covered | complex128, complex64 | - | yes |
| std_unbiased | identity | reviewed | covered | complex128, complex64 | - | yes |
| sub | identity | reviewed | covered | complex128, complex64 | - | yes |
| sum | identity | reviewed | covered | complex128, complex64 | - | yes |
| svd | gauge_ill_defined | reviewed | covered | complex128 | - | yes |
| svd | s | reviewed | covered | complex128, complex64 | - | yes |
| svd | u_abs | reviewed | covered | complex128, complex64 | - | yes |
| svd | uvh_product | reviewed | covered | complex128, complex64 | - | yes |
| svd | vh_abs | reviewed | covered | complex128, complex64 | - | yes |
| svdvals | identity | not_required | covered | complex128, complex64 | - | yes |
| tan | identity | reviewed | covered | complex128, complex64 | - | yes |
| tanh | identity | reviewed | covered | complex128, complex64 | - | yes |
| tensorinv | identity | reviewed | covered | complex128, complex64 | - | yes |
| tensorsolve | identity | reviewed | covered | complex128, complex64 | - | yes |
| true_divide | identity | reviewed | covered | complex128, complex64 | - | yes |
| vander | identity | reviewed | covered | complex128, complex64 | - | yes |
| var | identity | reviewed | covered | complex128, complex64 | - | yes |
| var_unbiased | identity | reviewed | covered | complex128, complex64 | - | yes |
| vecdot | identity | reviewed | covered | complex128, complex64 | - | yes |
| vector_norm | identity | not_required | covered | complex128, complex64 | - | yes |

## Unsupported

| op | family | note status | db status | complex published dtypes | unsupported reason | ready |
| --- | --- | --- | --- | --- | --- | --- |
| __rmod__ | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| amax | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| amin | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| atan2 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| ceil | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| clamp_max | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| clamp_min | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| complex | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| copysign | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| deg2rad | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| digamma | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| erf | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| erfc | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| erfinv | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| floor | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| fmax | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| fmin | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| frac | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| frexp | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| hypot | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| i0 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| lgamma | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| logit | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| max_binary | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| maximum | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| min_binary | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| minimum | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| mvlgamma_mvlgamma_p_1 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| mvlgamma_mvlgamma_p_3 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| mvlgamma_mvlgamma_p_5 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nan_to_num | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nanmean | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nansum | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_celu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_elu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_hardshrink | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_hardsigmoid | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_hardtanh | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_logsigmoid | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_mish | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_prelu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_relu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_relu6 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_rrelu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_selu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_silu | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_softplus | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_softshrink | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| nn_functional_threshold | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polar | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_0 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_1 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_2 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_3 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| polygamma_polygamma_n_4 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| rad2deg | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| round | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| round_decimals_0 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| round_decimals_3 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| round_decimals_neg_3 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| sign | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_entr | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_erfcx | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_i0e | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_i1 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_i1e | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_log_ndtr | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_ndtr | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_ndtri | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_polygamma_special_polygamma_n_0 | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| special_xlog1py | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| trunc | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |
| xlogy | identity | not_required | unsupported | - | float-only in pinned PyTorch upstream AD coverage | no |

## Still Pending

None.
