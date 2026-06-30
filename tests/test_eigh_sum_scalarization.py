import unittest


class EighSumScalarizationTests(unittest.TestCase):
    def _import_torch(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")
        return torch

    def _real_spd_case(self, torch):
        base = torch.tensor(
            [
                [2.0, -0.5, 0.25],
                [0.75, 1.5, -0.25],
                [-0.125, 0.5, 1.0],
            ],
            dtype=torch.float64,
        )
        tangent_seed = torch.tensor(
            [
                [0.25, 0.75, -0.5],
                [0.75, -0.125, 0.375],
                [-0.5, 0.375, 0.625],
            ],
            dtype=torch.float64,
        )
        eye = torch.eye(base.shape[0], dtype=base.dtype)
        return base @ base.mT + 0.5 * eye, tangent_seed

    def _complex_hermitian_case(self, torch):
        real = torch.tensor(
            [
                [4.0, 1.0, 0.2],
                [1.0, 3.0, -0.4],
                [0.2, -0.4, 2.0],
            ],
            dtype=torch.float64,
        )
        imag = torch.tensor(
            [
                [0.0, 0.7, -0.3],
                [-0.7, 0.0, 0.5],
                [0.3, -0.5, 0.0],
            ],
            dtype=torch.float64,
        )
        tangent_real = torch.tensor(
            [
                [0.5, -0.25, 0.125],
                [-0.25, 0.75, 0.375],
                [0.125, 0.375, -0.625],
            ],
            dtype=torch.float64,
        )
        tangent_imag = torch.tensor(
            [
                [0.0, 0.2, -0.4],
                [-0.2, 0.0, 0.1],
                [0.4, -0.1, 0.0],
            ],
            dtype=torch.float64,
        )
        return real + 1j * imag, tangent_real + 1j * tangent_imag

    def test_jvp_matches_tangent_trace(self) -> None:
        torch = self._import_torch()

        for matrix, tangent in (
            self._real_spd_case(torch),
            self._complex_hermitian_case(torch),
        ):
            matrix = matrix.detach().requires_grad_(True)

            def phi(a):
                return torch.linalg.eigh(a).eigenvalues.sum()

            _, jvp = torch.autograd.functional.jvp(phi, matrix, tangent)
            expected = torch.trace(tangent).real

            self.assertTrue(torch.allclose(jvp, expected, rtol=1e-10, atol=1e-10))

            step = 1e-6
            fd_jvp = (phi(matrix + step * tangent) - phi(matrix - step * tangent)) / (
                2 * step
            )
            self.assertTrue(torch.allclose(fd_jvp, expected, rtol=1e-8, atol=1e-8))

    def test_vjp_matches_scalar_cotangent_times_identity(self) -> None:
        torch = self._import_torch()

        for matrix, _ in (
            self._real_spd_case(torch),
            self._complex_hermitian_case(torch),
        ):
            matrix = matrix.detach().requires_grad_(True)
            cotangent = torch.tensor(2.75, dtype=torch.float64)

            phi = torch.linalg.eigh(matrix).eigenvalues.sum()
            (cotangent * phi).backward()

            expected = cotangent.to(matrix.dtype) * torch.eye(
                matrix.shape[0],
                dtype=matrix.dtype,
            )
            self.assertTrue(
                torch.allclose(matrix.grad, expected, rtol=1e-10, atol=1e-10)
            )

    def test_complex_eigenvalue_only_transpose_uses_adjoint(self) -> None:
        torch = self._import_torch()

        matrix, _ = self._complex_hermitian_case(torch)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        cotangent = torch.tensor(2.75, dtype=torch.float64)
        bar_e = torch.full_like(eigenvalues, cotangent)
        diagonal = torch.diag(bar_e.to(matrix.dtype))

        adjoint_path = eigenvectors @ diagonal @ eigenvectors.mH
        transpose_path = eigenvectors @ diagonal @ eigenvectors.T
        expected = cotangent.to(matrix.dtype) * torch.eye(
            matrix.shape[0],
            dtype=matrix.dtype,
        )

        self.assertTrue(torch.allclose(adjoint_path, expected, rtol=1e-10, atol=1e-10))
        self.assertGreater(torch.max(torch.abs(transpose_path - expected)).item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
