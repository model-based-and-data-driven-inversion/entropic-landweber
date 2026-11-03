"""Forward and adjoint operators for linear inverse problems.

Includes:
1. Fredholm integral operators on L^1(0, 1) -> L^2(0, 1)
2. Discrete Fourier sampling operators on L^1([-a, a]) -> C^n
"""

from __future__ import annotations

from typing import Callable
import numpy as np


def trapezoidal_weights(n_points: int, dx: float) -> np.ndarray:
    """Compute 1D composite trapezoidal rule weights.

    Parameters
    ----------
    n_points : int
        Number of grid points.
    dx : float
        Grid spacing.

    Returns
    -------
    np.ndarray
        Weight array of length `n_points`.
    """
    w = np.full(n_points, dx, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


class FredholmOperator:
    r"""Linear Fredholm integral operator of the first kind:

    .. math::
        (Au)(x) = \int_\Omega k(x, y) u(y) \, \mathrm{d}y

    between :math:`L^1(\Omega)` and :math:`L^2(\tilde{\Omega})`.
    Discretized on a uniform grid via the trapezoidal rule.

    Adjoint operator:
    .. math::
        (A^* v)(y) = \int_{\tilde{\Omega}} k(x, y) v(x) \, \mathrm{d}x
    """

    def __init__(
        self,
        kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray | None = None,
    ) -> None:
        r"""Initialise the Fredholm operator.

        Parameters
        ----------
        kernel_fn : callable
            Bivariate kernel function k(x, y).
        x_grid : np.ndarray
            Discretization grid in target space \tilde{\Omega}.
        y_grid : np.ndarray | None, optional
            Discretization grid in domain \Omega. If None, uses x_grid.
        """
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.y_grid = self.x_grid if y_grid is None else np.asarray(y_grid, dtype=float)

        self.dx = float(self.x_grid[1] - self.x_grid[0])
        self.dy = float(self.y_grid[1] - self.y_grid[0])

        self.w_x = trapezoidal_weights(len(self.x_grid), self.dx)
        self.w_y = trapezoidal_weights(len(self.y_grid), self.dy)

        # Kernel matrix K[i, j] = k(x_i, y_j)
        self.K = np.asarray(kernel_fn(self.x_grid[:, None], self.y_grid[None, :]), dtype=float)

        # Discretized forward matrix: (A u)_i = sum_j K[i, j] u_j w_y[j]
        self.matrix = self.K * self.w_y[None, :]

        # Spectral norm of discretized operator
        self._norm: float | None = None

    @property
    def operator_norm(self) -> float:
        """Compute or retrieve operator spectral norm ||A||_2."""
        if self._norm is None:
            self._norm = float(np.linalg.norm(self.matrix, 2))
        return self._norm

    def apply(self, u: np.ndarray) -> np.ndarray:
        """Apply forward operator Au."""
        return self.matrix @ u

    def apply_adjoint(self, v: np.ndarray) -> np.ndarray:
        r"""Apply adjoint operator A^* v.

        In the continuous L^2 inner product,
        .. math::
            \langle Au, v \rangle_{L^2} = \int u(y) (A^* v)(y) \, \mathrm{d}y

        with :math:`(A^* v)(y) = \int k(x, y) v(x) \, \mathrm{d}x \approx K^T (W_x v)`.
        """
        return self.K.T @ (self.w_x * v)

    def row_action(self, j: int, u: np.ndarray) -> tuple[float, np.ndarray]:
        r"""Compute evaluation and adjoint for single row block j.

        Parameters
        ----------
        j : int
            Row index in {0, ..., M-1}.
        u : np.ndarray
            Current iterate.

        Returns
        -------
        val : float
            Single measurement prediction (A_j u).
        adj_component : np.ndarray
            Adjoint contribution A_j^*(1).
        """
        val = float(np.dot(self.matrix[j, :], u))
        adj_component = self.K[j, :] * self.w_x[j]
        return val, adj_component


class FourierSamplingOperator:
    r"""Discrete Fourier sampling operator:

    .. math::
        (Au)_j = \frac{1}{\sqrt{2\pi}} \int_{-a}^a u(x) e^{-i x \xi_j} \, \mathrm{d}x, \quad j=1, \dots, n

    with coordinates :math:`\xi_j = \frac{2\pi (j-1)}{n}`.

    Adjoint operator:
    .. math::
        (A^* v)(x) = \operatorname{Re}\left( \frac{1}{\sqrt{2\pi}} \sum_{j=1}^n v_j e^{i x \xi_j} \right)
    """

    def __init__(
        self,
        n_samples: int = 16,
        a: float = 10.0,
        n_grid: int = 4001,
    ) -> None:
        """Initialise Fourier sampling operator.

        Parameters
        ----------
        n_samples : int, optional
            Number of discrete Fourier frequencies n, by default 16.
        a : float, optional
            Domain boundary [-a, a], by default 10.0.
        n_grid : int, optional
            Number of quadrature grid points on [-a, a], by default 4001.
        """
        self.n_samples = n_samples
        self.a = float(a)
        self.n_grid = n_grid

        # Frequency coordinates: xi_j = 2 pi (j-1) / n for j=1..n
        j_indices = np.arange(1, n_samples + 1)
        self.xi = 2.0 * np.pi * (j_indices - 1) / float(n_samples)

        # Spatial grid on [-a, a]
        self.x_grid = np.linspace(-self.a, self.a, n_grid)
        self.dx = float(self.x_grid[1] - self.x_grid[0])
        self.weights = trapezoidal_weights(n_grid, self.dx)

        # Precompute Fourier kernel matrices
        # Shape: (n_samples, n_grid)
        # exp(-i * xi_j * x_k)
        self.exp_neg = np.exp(-1j * np.outer(self.xi, self.x_grid))
        # exp(+i * xi_j * x_k)
        self.exp_pos = np.exp(1j * np.outer(self.xi, self.x_grid))

        # Forward matrix: A_mat @ u
        self.matrix = (1.0 / np.sqrt(2.0 * np.pi)) * self.exp_neg * self.weights[None, :]

    def apply(self, u: np.ndarray) -> np.ndarray:
        """Apply forward operator Au to real function u on the grid."""
        return self.matrix @ u

    def apply_adjoint(self, v: np.ndarray) -> np.ndarray:
        r"""Apply adjoint operator A^* v for complex vector v in C^n.

        .. math::
            (A^* v)(x) = \operatorname{Re}\left( \frac{1}{\sqrt{2\pi}} \sum_{j=1}^n v_j e^{i x \xi_j} \right)
        """
        v_col = np.asarray(v, dtype=complex)[:, None]
        summed = np.sum(v_col * self.exp_pos, axis=0)
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.real(summed)
