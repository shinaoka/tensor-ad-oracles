import unittest


class LuQrSumScalarizationTests(unittest.TestCase):
    def _import_torch(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")
        return torch

    def _right_solve_adjoint(self, torch, rhs, upper):
        return torch.linalg.solve_triangular(upper, rhs.mH, upper=True).mH

    def _left_solve_adjoint_unit_lower(self, torch, lower, rhs):
        return torch.linalg.solve_triangular(
            lower.mH,
            rhs,
            upper=True,
            unitriangular=True,
        )

    def _copyltu(self, torch, value):
        lower = torch.tril(value, diagonal=-1)
        diag = torch.diag_embed(torch.real(torch.diagonal(value)).to(value.dtype))
        return lower + diag + lower.mH

    def _tril_im_inv_adj_skew(self, torch, value):
        out = torch.tril(value - value.mH).clone()
        if value.is_complex():
            idx = torch.arange(out.shape[0], device=out.device)
            out[idx, idx] *= 0.5
        return out

    def _lu_sum_grad_from_rule(self, torch, matrix):
        permutation, lower, upper = torch.linalg.lu(matrix)
        bar_l = torch.ones_like(lower)
        bar_u = torch.ones_like(upper)
        rows, cols = matrix.shape

        # torch.linalg.lu returns A = P L U, while the notes write P A = L U.
        # Therefore the notes' P^T action is left multiplication by this P.
        if rows == cols:
            bar_f = torch.tril(lower.mH @ bar_l, diagonal=-1) + torch.triu(
                bar_u @ upper.mH
            )
            y = self._left_solve_adjoint_unit_lower(torch, lower, bar_f)
            return permutation @ self._right_solve_adjoint(torch, y, upper)

        if rows < cols:
            upper_1 = upper[:, :rows]
            upper_2 = upper[:, rows:]
            bar_u_1 = bar_u[:, :rows]
            bar_u_2 = bar_u[:, rows:]
            bar_h_1 = torch.tril(
                lower.mH @ bar_l - bar_u_2 @ upper_2.mH,
                diagonal=-1,
            ) + torch.triu(bar_u_1 @ upper_1.mH)
            bar_h_1 = self._right_solve_adjoint(torch, bar_h_1, upper_1)
            bar_h = torch.cat([bar_h_1, bar_u_2], dim=1)
            return permutation @ self._left_solve_adjoint_unit_lower(
                torch,
                lower,
                bar_h,
            )

        lower_1 = lower[:cols, :]
        lower_2 = lower[cols:, :]
        bar_l_1 = bar_l[:cols, :]
        bar_l_2 = bar_l[cols:, :]
        bar_h_1 = torch.tril(lower_1.mH @ bar_l_1, diagonal=-1) + torch.triu(
            bar_u @ upper.mH - lower_2.mH @ bar_l_2
        )
        bar_h_1 = self._left_solve_adjoint_unit_lower(torch, lower_1, bar_h_1)
        bar_h = torch.cat([bar_h_1, bar_l_2], dim=0)
        return permutation @ self._right_solve_adjoint(torch, bar_h, upper)

    def _qr_sum_grad_from_rule(self, torch, matrix):
        q_factor, r_factor = torch.linalg.qr(matrix, mode="reduced")
        bar_q = torch.ones_like(q_factor)
        bar_r = torch.ones_like(r_factor)
        rows, cols = matrix.shape

        if rows >= cols:
            w_sum = r_factor @ bar_r.mH - bar_q.mH @ q_factor
            h_sum = self._copyltu(torch, w_sum)
            return self._right_solve_adjoint(
                torch,
                bar_q + q_factor @ h_sum,
                r_factor,
            )

        helper = self._tril_im_inv_adj_skew(
            torch,
            q_factor.mH @ bar_q - bar_r @ r_factor.mH,
        )
        lead = self._right_solve_adjoint(
            torch,
            q_factor @ helper,
            r_factor[:, :rows],
        )
        padded = torch.zeros_like(matrix)
        padded[:, :rows] = lead
        return q_factor @ bar_r + padded

    def _sum_lu_outputs(self, torch, matrix):
        _, lower, upper = torch.linalg.lu(matrix)
        return (lower.sum() + upper.sum()).real

    def _sum_qr_outputs(self, torch, matrix):
        q_factor, r_factor = torch.linalg.qr(matrix, mode="reduced")
        return (q_factor.sum() + r_factor.sum()).real

    def _autograd_grad(self, torch, scalarized, matrix):
        leaf = matrix.detach().clone().requires_grad_(True)
        scalarized(torch, leaf).backward()
        return leaf.grad

    def _tangent_like(self, torch, matrix):
        base = torch.arange(
            1,
            matrix.numel() + 1,
            dtype=torch.float64,
            device=matrix.device,
        ).reshape(matrix.shape)
        base = base / (3 * matrix.numel())
        if matrix.is_complex():
            imag = torch.flip(base, dims=(1,)) / (5 * matrix.numel())
            return base.to(matrix.dtype) + 1j * imag.to(matrix.dtype)
        return base.to(matrix.dtype)

    def _real_inner(self, torch, lhs, rhs):
        return torch.real(torch.sum(torch.conj(lhs) * rhs))

    def _assert_rule_matches_autograd_and_fd(self, torch, scalarized, rule, matrix):
        expected = rule(torch, matrix)
        actual = self._autograd_grad(torch, scalarized, matrix)
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-9, atol=1e-9),
            msg=f"autograd mismatch for shape={tuple(matrix.shape)} dtype={matrix.dtype}",
        )

        tangent = self._tangent_like(torch, matrix)
        step = 1e-6
        fd_jvp = (
            scalarized(torch, matrix + step * tangent)
            - scalarized(torch, matrix - step * tangent)
        ) / (2 * step)
        rule_jvp = self._real_inner(torch, expected, tangent)
        self.assertTrue(
            torch.allclose(fd_jvp, rule_jvp, rtol=1e-6, atol=1e-6),
            msg=f"finite-difference mismatch for shape={tuple(matrix.shape)} dtype={matrix.dtype}",
        )

    def _lu_cases(self, torch):
        square = torch.tensor(
            [
                [3.0, 1.0, 2.0],
                [1.0, 4.0, 0.5],
                [0.25, -1.0, 2.5],
            ],
            dtype=torch.float64,
        )
        square_imag = torch.tensor(
            [
                [0.2, -0.3, 0.5],
                [0.1, 0.4, -0.2],
                [-0.6, 0.25, 0.3],
            ],
            dtype=torch.float64,
        )
        wide = torch.tensor(
            [
                [2.0, -1.0, 3.0, 0.5],
                [1.0, 3.0, -0.25, 2.0],
            ],
            dtype=torch.float64,
        )
        wide_imag = torch.tensor(
            [
                [0.25, 0.5, -0.75, 0.125],
                [-0.5, 0.25, 0.375, -0.25],
            ],
            dtype=torch.float64,
        )
        tall = torch.tensor(
            [
                [2.0, -1.0],
                [1.0, 3.0],
                [0.5, 2.0],
                [-1.0, 0.25],
            ],
            dtype=torch.float64,
        )
        tall_imag = torch.tensor(
            [
                [0.2, -0.4],
                [0.5, 0.1],
                [-0.25, 0.75],
                [0.3, -0.2],
            ],
            dtype=torch.float64,
        )
        return (
            square,
            wide,
            tall,
            square + 1j * square_imag,
            wide + 1j * wide_imag,
            tall + 1j * tall_imag,
        )

    def _qr_cases(self, torch):
        tall = torch.tensor(
            [
                [2.0, -1.0],
                [1.0, 3.0],
                [0.5, 2.0],
                [-1.0, 0.25],
            ],
            dtype=torch.float64,
        )
        tall_imag = torch.tensor(
            [
                [0.2, -0.4],
                [0.5, 0.1],
                [-0.25, 0.75],
                [0.3, -0.2],
            ],
            dtype=torch.float64,
        )
        wide = torch.tensor(
            [
                [2.0, -1.0, 3.0, 0.5],
                [1.0, 3.0, -0.25, 2.0],
            ],
            dtype=torch.float64,
        )
        wide_imag = torch.tensor(
            [
                [0.25, 0.5, -0.75, 0.125],
                [-0.5, 0.25, 0.375, -0.25],
            ],
            dtype=torch.float64,
        )
        return (
            tall,
            wide,
            tall + 1j * tall_imag,
            wide + 1j * wide_imag,
        )

    def test_lu_sum_vjp_matches_autograd_and_fd(self) -> None:
        torch = self._import_torch()

        for matrix in self._lu_cases(torch):
            self._assert_rule_matches_autograd_and_fd(
                torch,
                self._sum_lu_outputs,
                self._lu_sum_grad_from_rule,
                matrix,
            )

    def test_qr_sum_vjp_matches_autograd_and_fd(self) -> None:
        torch = self._import_torch()

        for matrix in self._qr_cases(torch):
            self._assert_rule_matches_autograd_and_fd(
                torch,
                self._sum_qr_outputs,
                self._qr_sum_grad_from_rule,
                matrix,
            )


if __name__ == "__main__":
    unittest.main()
