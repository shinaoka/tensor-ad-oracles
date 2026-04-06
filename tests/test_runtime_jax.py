import unittest

from generators import runtime_jax


class RuntimeJaxTests(unittest.TestCase):
    def test_import_generation_runtime_returns_pinned_jax_modules(self) -> None:
        try:
            import jaxlib

            jax, jnp = runtime_jax.import_generation_runtime()
        except Exception as exc:
            self.skipTest(f"jax runtime unavailable: {exc}")

        self.assertEqual(runtime_jax.normalize_jax_version(jax.__version__), "0.9.1")
        self.assertEqual(runtime_jax.normalize_jax_version(jaxlib.__version__), "0.9.1")

    def test_dtype_name_normalizes_jax_dtype_objects(self) -> None:
        try:
            import jax.numpy as jnp
        except Exception as exc:
            self.skipTest(f"jax runtime unavailable: {exc}")

        self.assertEqual(runtime_jax.dtype_name(jnp.float32), "float32")
        self.assertEqual(runtime_jax.dtype_name(jnp.dtype("complex128")), "complex128")

    def test_normalize_raw_tensor_map_scales_each_array_independently(self) -> None:
        try:
            import jax.numpy as jnp
        except Exception as exc:
            self.skipTest(f"jax runtime unavailable: {exc}")

        normalized = runtime_jax.normalize_raw_tensor_map(
            {
                "a": jnp.array([3.0, 4.0]),
                "b": jnp.array([0.0, 0.0]),
            }
        )

        self.assertAlmostEqual(float(jnp.linalg.norm(normalized["a"])), 1.0)
        self.assertEqual(normalized["b"].tolist(), [0.0, 0.0])

    def test_compute_jax_observable_jvp_and_vjp(self) -> None:
        try:
            import jax.numpy as jnp
        except Exception as exc:
            self.skipTest(f"jax runtime unavailable: {exc}")

        def observable_fn(inputs):
            return {"value": inputs["x"] ** 2 + 1.0}

        inputs = {"x": jnp.array([2.0])}
        direction = {"x": jnp.array([3.0])}
        cotangent = {"value": jnp.array([5.0])}

        jvp = runtime_jax.compute_jax_jvp(observable_fn, inputs, direction)
        vjp = runtime_jax.compute_jax_vjp(observable_fn, inputs, cotangent)

        self.assertEqual(jvp["value"].tolist(), [12.0])
        self.assertEqual(vjp["x"].tolist(), [20.0])

    def test_compute_jax_linearization_returns_directional_output(self) -> None:
        try:
            import jax.numpy as jnp
        except Exception as exc:
            self.skipTest(f"jax runtime unavailable: {exc}")

        def raw_output_fn(inputs):
            return {"value": jnp.abs(inputs["x"])}

        inputs = {"x": jnp.array([3.0])}
        direction = {"x": jnp.array([1.0])}

        linearization, _ = runtime_jax.compute_jax_linearization(
            raw_output_fn,
            inputs,
            direction,
        )

        self.assertEqual(linearization["value"].tolist(), [1.0])

    def test_compute_jax_transpose_and_adjoint_check(self) -> None:
        try:
            import jax.numpy as jnp
        except Exception as exc:
            self.skipTest(f"jax runtime unavailable: {exc}")

        def linear_fn(tangent):
            return {"value": tangent["x"] * 12.0}

        inputs = {"x": jnp.array([2.0])}
        cotangent = {"value": jnp.array([5.0])}
        direction = {"x": jnp.array([3.0])}

        transpose = runtime_jax.compute_jax_transpose(linear_fn, inputs, cotangent)
        adjoint_check = runtime_jax.compute_jax_adjoint_check(
            runtime_jax.tensor_map_inner_product(cotangent, linear_fn(direction)),
            runtime_jax.tensor_map_inner_product(transpose, direction),
        )

        self.assertEqual(transpose["x"].tolist(), [60.0])
        self.assertEqual(adjoint_check, {"lhs": 180.0, "rhs": 180.0, "abs_err": 0.0, "rel_err": 0.0})


if __name__ == "__main__":
    unittest.main()
