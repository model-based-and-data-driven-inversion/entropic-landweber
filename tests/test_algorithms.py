"""Unit tests for iterative algorithms (monotonicity, positivity, convergence)."""

from __future__ import annotations

import unittest
import numpy as np
from entropic_landweber.algorithms import (
    entropic_landweber,
    expectation_maximization,
    projected_landweber,
)
from entropic_landweber.ground_truth import exact_solution_1, kernel_1
from entropic_landweber.operators import FredholmOperator


class TestAlgorithms(unittest.TestCase):
    def test_entropic_landweber_monotonicity_and_positivity(self) -> None:
        """Check Proposition 3.3 (i): residual ||Au_k - y|| decreases monotonically."""
        x = np.linspace(0, 1, 60)
        op = FredholmOperator(kernel_1, x)
        z = exact_solution_1(x, sigma2=0.01, normalize=True)
        y = op.apply(z)

        u0 = np.ones_like(x)
        res = entropic_landweber(op, y, u0, lam=0.5, n_iterations=30, m=1)

        # Check nonnegativity
        self.assertTrue(np.all(res.u_final >= 0))

        # Check unit mass (m = 1)
        mass = np.sum(res.u_final * op.w_y)
        self.assertAlmostEqual(mass, 1.0, places=10)

        # Check monotonic energy decrease: E_{k+1} <= E_k
        diffs = np.diff(res.energies)
        self.assertTrue(np.all(diffs <= 1e-12))

    def test_cumulative_vs_stepwise_equivalence(self) -> None:
        """Verify Section 6.2: recursive cumulative accumulator matches step-wise update."""
        x = np.linspace(0, 1, 50)
        op = FredholmOperator(kernel_1, x)
        z = exact_solution_1(x, sigma2=0.01, normalize=True)
        y = op.apply(z)
        u0 = np.ones_like(x)

        # Run with cumulative form (use_cumulative_form=True)
        res_cum = entropic_landweber(op, y, u0, lam=0.8, n_iterations=25, m=1, use_cumulative_form=True)
        # Run with step-wise form (use_cumulative_form=False)
        res_step = entropic_landweber(op, y, u0, lam=0.8, n_iterations=25, m=1, use_cumulative_form=False)

        np.testing.assert_allclose(res_cum.u_final, res_step.u_final, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_cum.energies, res_step.energies, rtol=1e-12, atol=1e-12)

    def test_projected_landweber_positivity(self) -> None:
        """Check that Projected Landweber preserves nonnegativity."""
        x = np.linspace(0, 1, 60)
        op = FredholmOperator(kernel_1, x)
        z = exact_solution_1(x, sigma2=0.01, normalize=True)
        y = op.apply(z)

        u0 = np.ones_like(x)
        res = projected_landweber(op, y, u0, tau=0.5, n_iterations=20)
        self.assertTrue(np.all(res.u_final >= 0))

    def test_em_algorithm_positivity(self) -> None:
        """Check that EM preserves strict positivity."""
        x = np.linspace(0, 1, 60)
        op = FredholmOperator(kernel_1, x)
        z = exact_solution_1(x, sigma2=0.01, normalize=True)
        y = op.apply(z)

        u0 = np.ones_like(x)
        res = expectation_maximization(op, y, u0, n_iterations=20)
        self.assertTrue(np.all(res.u_final > 0))


if __name__ == "__main__":
    unittest.main()
