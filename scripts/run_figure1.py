"""Replicate Figure 1 from Burger, Resmerita, Benning (2020).

L^1 error plotted versus iteration number (0 to 1000) for:
1. Kernel 1: k1(x, y) = exp(x*y), exact solution z1(x) = exp(-x^2 / (2*sigma^2))
2. Kernel 2: k2(x, y) = 3*exp(-(x-y)^2 / 0.04), exact solution z2(x)
3. Kernel 3: k3(x, y) = 1 (x >= y), exact solution z3(x) = exp(-x^2 / (2*sigma^2))

Comparing:
- Entropic Landweber (red)
- Randomized / Stochastic Entropic Landweber (black)
- Expectation-Maximization (blue)
- Projected Landweber (green)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from entropic_landweber.algorithms import (
    entropic_landweber,
    expectation_maximization,
    projected_landweber,
    randomized_entropic_landweber,
)
from entropic_landweber.ground_truth import (
    exact_solution_1,
    exact_solution_2,
    exact_solution_3,
    kernel_1,
    kernel_2,
    kernel_3,
)
from entropic_landweber.operators import FredholmOperator
from entropic_landweber.plotting import plot_figure1


def run_case1(n_grid: int = 100, n_iterations: int = 1000) -> dict[str, np.ndarray]:
    """Execute algorithms on Case 1."""
    x = np.linspace(0.0, 1.0, n_grid)
    op = FredholmOperator(kernel_1, x)
    z1 = exact_solution_1(x, sigma2=0.01, normalize=True)
    y1 = op.apply(z1)

    u0 = np.ones(n_grid)

    res_entropy = entropic_landweber(
        op, y1, u0, lam=0.9, n_iterations=n_iterations, m=1, z_true=z1
    )
    res_rand = randomized_entropic_landweber(
        op, y1, u0, lam=0.9, n_iterations=n_iterations, m=1, z_true=z1, seed=0
    )
    res_em = expectation_maximization(
        op, y1, u0, n_iterations=n_iterations, z_true=z1
    )
    res_proj = projected_landweber(
        op, y1, u0, tau=0.87, n_iterations=n_iterations, z_true=z1
    )

    return {
        "entropy": res_entropy.l1_errors,
        "randomized": res_rand.l1_errors,
        "em": res_em.l1_errors,
        "projection": res_proj.l1_errors,
    }


def run_case2(n_grid: int = 100, n_iterations: int = 1000) -> dict[str, np.ndarray]:
    """Execute algorithms on Case 2."""
    x = np.linspace(0.0, 1.0, n_grid)
    op = FredholmOperator(kernel_2, x)
    z2 = exact_solution_2(x, sigma2=0.01)
    y2 = op.apply(z2)

    u0 = np.full(n_grid, 2.0 / 3.0)

    res_entropy = entropic_landweber(
        op, y2, u0, lam=0.3, n_iterations=n_iterations, m=0, z_true=z2
    )
    res_rand = randomized_entropic_landweber(
        op, y2, u0, lam=0.3, n_iterations=n_iterations, m=0, z_true=z2, seed=4
    )
    res_em = expectation_maximization(
        op, y2, u0, n_iterations=n_iterations, z_true=z2
    )
    res_proj = projected_landweber(
        op, y2, u0, tau=0.3, n_iterations=n_iterations, z_true=z2
    )

    return {
        "entropy": res_entropy.l1_errors,
        "randomized": res_rand.l1_errors,
        "em": res_em.l1_errors,
        "projection": res_proj.l1_errors,
    }


def run_case3(n_grid: int = 100, n_iterations: int = 1000) -> dict[str, np.ndarray]:
    """Execute algorithms on Case 3."""
    x = np.linspace(0.0, 1.0, n_grid)
    op = FredholmOperator(kernel_3, x)
    z3 = exact_solution_3(x, sigma2=0.01, normalize=True)
    y3 = op.apply(z3)

    u0 = np.ones(n_grid)

    res_entropy = entropic_landweber(
        op, y3, u0, lam=1.0, n_iterations=n_iterations, m=1, z_true=z3
    )
    res_rand = randomized_entropic_landweber(
        op, y3, u0, lam=1.0, n_iterations=n_iterations, m=1, z_true=z3, seed=2
    )
    res_em = expectation_maximization(
        op, y3, u0, n_iterations=n_iterations, z_true=z3
    )
    res_proj = projected_landweber(
        op, y3, u0, tau=1.0, n_iterations=n_iterations, z_true=z3
    )

    return {
        "entropy": res_entropy.l1_errors,
        "randomized": res_rand.l1_errors,
        "em": res_em.l1_errors,
        "projection": res_proj.l1_errors,
    }


def main() -> None:
    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Running Case 1 (Kernel k1: exp(xy))...")
    case1 = run_case1()

    print("Running Case 2 (Kernel k2: Gaussian convolution)...")
    case2 = run_case2()

    print("Running Case 3 (Kernel k3: Numerical differentiation)...")
    case3 = run_case3()

    png_path = figures_dir / "figure1.png"
    pdf_path = figures_dir / "figure1.pdf"

    print(f"Generating Figure 1 and saving to {png_path}...")
    fig = plot_figure1(case1, case2, case3, save_path=png_path)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print("Figure 1 generated successfully.")


if __name__ == "__main__":
    main()
