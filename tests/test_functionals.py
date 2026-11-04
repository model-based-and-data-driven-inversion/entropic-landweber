"""Unit tests for entropy, Bregman distance, and distance functionals."""

from __future__ import annotations

import unittest
import numpy as np
from entropic_landweber.functionals import (
    kullback_leibler_divergence,
    l1_error,
    l1_norm,
    shannon_entropy,
)


class TestFunctionals(unittest.TestCase):
    def test_shannon_entropy(self) -> None:
        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        u_const = np.ones_like(x)
        # f(1) = \int 1 * ln(1) dx = 0
        self.assertAlmostEqual(shannon_entropy(u_const, dx=dx), 0.0, places=10)

        # Negative value yields infinity
        u_neg = np.array([-1.0, 2.0])
        self.assertEqual(shannon_entropy(u_neg), float("inf"))

    def test_kullback_leibler_divergence(self) -> None:
        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        u = np.exp(-x)
        v = np.exp(-2 * x)

        # d(u, u) == 0
        self.assertAlmostEqual(kullback_leibler_divergence(u, u, dx=dx), 0.0, places=10)

        # Nonnegativity d(v, u) >= 0
        d_vu = kullback_leibler_divergence(v, u, dx=dx)
        self.assertGreaterEqual(d_vu, 0.0)

        # Strict convexity: d(v, u) > 0 for v != u
        self.assertGreater(d_vu, 1e-5)

    def test_l1_norm_and_error(self) -> None:
        from entropic_landweber.operators import trapezoidal_weights

        x = np.linspace(0, 1, 101)
        dx = x[1] - x[0]
        w = trapezoidal_weights(len(x), dx)
        u = np.full_like(x, 2.0)
        z = np.full_like(x, 0.5)

        self.assertAlmostEqual(l1_norm(u, weights=w), 2.0, places=10)
        self.assertAlmostEqual(l1_error(u, z, weights=w), 1.5, places=10)


if __name__ == "__main__":
    unittest.main()
