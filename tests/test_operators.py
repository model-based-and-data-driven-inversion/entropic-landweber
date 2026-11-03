"""Unit tests for Fredholm and Fourier operators and adjoint verification."""

from __future__ import annotations

import unittest
import numpy as np
from entropic_landweber.ground_truth import kernel_1
from entropic_landweber.operators import FourierSamplingOperator, FredholmOperator


class TestOperators(unittest.TestCase):
    def test_fredholm_adjoint_property(self) -> None:
        """Verify duality: <Au, v>_{L^2} == <u, A*v>_{L^2}."""
        x = np.linspace(0, 1, 100)
        op = FredholmOperator(kernel_1, x)

        rng = np.random.default_rng(123)
        u = rng.uniform(0.1, 2.0, len(x))
        v = rng.uniform(-1.0, 1.0, len(x))

        Au = op.apply(u)
        Astar_v = op.apply_adjoint(v)

        # <Au, v>_{L^2} = \int (Au)(x) v(x) dx
        inner_Au_v = np.sum((Au * v) * op.w_x)

        # <u, A*v>_{L^2} = \int u(y) (A*v)(y) dy
        inner_u_Astar_v = np.sum((u * Astar_v) * op.w_y)

        self.assertAlmostEqual(inner_Au_v, inner_u_Astar_v, places=10)

    def test_fourier_adjoint_property(self) -> None:
        r"""Verify duality: sum_j (Au)_j * conj(v_j) == int u(x) (A*v)(x) dx for real u."""
        op = FourierSamplingOperator(n_samples=16, a=10.0, n_grid=2001)

        rng = np.random.default_rng(456)
        u = rng.uniform(0.01, 1.0, op.n_grid)
        v = rng.normal(size=op.n_samples) + 1j * rng.normal(size=op.n_samples)

        Au = op.apply(u)
        Astar_v = op.apply_adjoint(v)

        discrete_inner = np.real(np.sum(Au * np.conj(v)))
        integral_inner = np.sum(u * Astar_v * op.weights)

        self.assertAlmostEqual(discrete_inner, integral_inner, places=10)


if __name__ == "__main__":
    unittest.main()
