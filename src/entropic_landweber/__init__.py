"""Entropic Landweber package for linear ill-posed problems.

Replication and implementation of algorithms from:
Martin Burger, Elena Resmerita, and Martin Benning.
"An entropic Landweber method for linear ill-posed problems"
Inverse Problems 36 (2020) 015009.
"""

from .algorithms import (
    IterationResult,
    entropic_landweber,
    expectation_maximization,
    projected_landweber,
    randomized_entropic_landweber,
)
from .functionals import (
    d_functional,
    kullback_leibler_divergence,
    l1_error,
    l1_norm,
    shannon_entropy,
)
from .ground_truth import (
    exact_solution_1,
    exact_solution_2,
    exact_solution_3,
    fourier_ground_truth_z1,
    fourier_ground_truth_z2,
    kernel_1,
    kernel_2,
    kernel_3,
)
from .operators import (
    FourierSamplingOperator,
    FredholmOperator,
    trapezoidal_weights,
)
from .plotting import (
    plot_figure1,
    plot_figure2,
    plot_figure3,
    plot_figure4,
    plot_figure5,
)

__all__ = [
    "IterationResult",
    "entropic_landweber",
    "randomized_entropic_landweber",
    "projected_landweber",
    "expectation_maximization",
    "shannon_entropy",
    "kullback_leibler_divergence",
    "d_functional",
    "l1_norm",
    "l1_error",
    "kernel_1",
    "kernel_2",
    "kernel_3",
    "exact_solution_1",
    "exact_solution_2",
    "exact_solution_3",
    "fourier_ground_truth_z1",
    "fourier_ground_truth_z2",
    "FredholmOperator",
    "FourierSamplingOperator",
    "trapezoidal_weights",
    "plot_figure1",
    "plot_figure2",
    "plot_figure3",
    "plot_figure4",
    "plot_figure5",
]
