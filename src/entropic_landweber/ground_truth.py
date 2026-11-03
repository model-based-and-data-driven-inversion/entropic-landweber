"""Ground truth functions and kernels from the paper.

Contains the analytical definitions of kernels k1, k2, k3 and exact solutions
z1, z2, z3 for Section 6.1 (Integral equations), as well as probability
densities z1 and z2 for Section 6.2 (Fourier sampling).
"""

from __future__ import annotations

import numpy as np
from .operators import FourierSamplingOperator, trapezoidal_weights


# -----------------------------------------------------------------------------
# Section 6.1: Integral Equation Kernels and Exact Solutions
# -----------------------------------------------------------------------------

def kernel_1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Kernel 1:

    .. math::
        k_1(x, y) = e^{xy}
    """
    return np.exp(x * y)


def kernel_2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Kernel 2:

    .. math::
        k_2(x, y) = 3 e^{-\frac{(x-y)^2}{0.04}}
    """
    return 3.0 * np.exp(-((x - y) ** 2) / 0.04)


def kernel_3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Kernel 3 (Volterra operator / integration):

    .. math::
        k_3(x, y) = \begin{cases} 1 & \text{if } x \ge y \\ 0 & \text{else} \end{cases}
    """
    return np.where(x >= y, 1.0, 0.0)


def exact_solution_1(
    x: np.ndarray,
    sigma2: float = 0.01,
    normalize: bool = True,
) -> np.ndarray:
    r"""Exact solution 1:

    .. math::
        z_1(x) = e^{-\frac{x^2}{2\sigma^2}}

    Normalized to unit mass on [0, 1] when `normalize=True` to match Figure 1.
    """
    z = np.exp(-(x ** 2) / (2.0 * sigma2))
    if normalize:
        dx = float(x[1] - x[0])
        w = trapezoidal_weights(len(x), dx)
        z = z / np.sum(z * w)
    return z


def exact_solution_2(
    x: np.ndarray,
    sigma2: float = 0.01,
) -> np.ndarray:
    r"""Exact solution 2:

    .. math::
        z_2(x) = 1 - 0.9 e^{-\frac{(x-0.1)^2}{2\sigma^2}} - 0.3 e^{-\frac{(x-0.3)^2}{2\sigma^2}}
                   - 0.5 e^{-\frac{(x-0.5)^2}{2\sigma^2}} - 0.2 e^{-\frac{(x-0.7)^2}{2\sigma^2}}
                   - 0.7 e^{-\frac{(x-0.9)^2}{2\sigma^2}}
    """
    z = 1.0 - (
        0.9 * np.exp(-((x - 0.1) ** 2) / (2.0 * sigma2))
        + 0.3 * np.exp(-((x - 0.3) ** 2) / (2.0 * sigma2))
        + 0.5 * np.exp(-((x - 0.5) ** 2) / (2.0 * sigma2))
        + 0.2 * np.exp(-((x - 0.7) ** 2) / (2.0 * sigma2))
        + 0.7 * np.exp(-((x - 0.9) ** 2) / (2.0 * sigma2))
    )
    return z


def exact_solution_3(
    x: np.ndarray,
    sigma2: float = 0.01,
    normalize: bool = True,
) -> np.ndarray:
    r"""Exact solution 3:

    .. math::
        z_3(x) = e^{-\frac{x^2}{2\sigma^2}}

    Identical form to z_1, normalized to unit mass when `normalize=True`.
    """
    return exact_solution_1(x, sigma2=sigma2, normalize=normalize)


# -----------------------------------------------------------------------------
# Section 6.2: Discrete Sampling of Continuous Probability Densities
# -----------------------------------------------------------------------------

def gaussian_density(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""Normalized 1D Gaussian density:

    .. math::
        g(x, \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    """
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def gaussian_mixture_ztilde(
    x: np.ndarray,
    weights: tuple[float, float, float] = (0.1, 0.6, 0.3),
    means: tuple[float, float, float] = (0.0, -1.0, 0.5),
    sigmas: tuple[float, float, float] = (1.0, 0.1, 0.25),
) -> np.ndarray:
    r"""Linear combination of three normalised Gaussians:

    .. math::
        \tilde{z}(x) = \sum_{l=1}^3 c_l g(x, \mu_l, \sigma_l)
    """
    z = np.zeros_like(x, dtype=float)
    for c_l, mu_l, sig_l in zip(weights, means, sigmas):
        z += c_l * gaussian_density(x, mu_l, sig_l)
    return z


def fourier_ground_truth_z1(
    op: FourierSamplingOperator,
    weights: tuple[float, float, float] = (0.1, 0.6, 0.3),
    means: tuple[float, float, float] = (0.0, -1.0, 0.5),
    sigmas: tuple[float, float, float] = (1.0, 0.1, 0.25),
) -> np.ndarray:
    r"""Construct ground truth z_1 satisfying the source condition:

    .. math::
        1 + \ln z_1 \in \mathcal{R}(A^*)

    which gives:
    .. math::
        z_1(x) = c \exp\left( (A^* A \tilde{z})(x) \right)
               = c \exp\left( \frac{1}{2\pi} \operatorname{Re}\left( \sum_{j=1}^n \left(\int_\Omega \tilde{z}(t) e^{-it\xi_j}\mathrm{d}t\right) e^{ix\xi_j} \right) \right)

    where c is chosen to ensure :math:`\int_{-10}^{10} z_1(x) \, \mathrm{d}x = 1`.
    """
    # Exact Fourier transform of \tilde{z} on R (tails outside [-10, 10] are < 1e-18):
    F_ztilde = np.zeros(op.n_samples, dtype=complex)
    for c_l, mu_l, sig_l in zip(weights, means, sigmas):
        F_ztilde += c_l * np.exp(-1j * mu_l * op.xi - 0.5 * (sig_l * op.xi) ** 2)

    # Re( sum_{j=1}^n F_ztilde[j] e^{i x \xi_j} )
    # Note (A* A \tilde{z})(x) = (1 / (2*pi)) * Re( sum_{j=1}^n F_ztilde[j] e^{i x \xi_j} )
    re_sum = np.sum(
        F_ztilde[:, None] * op.exp_pos,
        axis=0,
    )
    unnorm = np.exp(np.real(re_sum) / (2.0 * np.pi))
    c_norm = 1.0 / np.sum(unnorm * op.weights)
    return c_norm * unnorm


def fourier_ground_truth_z2(
    op: FourierSamplingOperator,
    weights: tuple[float, float, float] = (0.1, 0.4, 0.5),
    means: tuple[float, float, float] = (0.0, -1.0, 0.5),
    sigmas: tuple[float, float, float] = (1.0, 0.1, 0.25),
) -> np.ndarray:
    r"""Construct ground truth z_2 = \tilde{z} violating the source condition.

    Uses weights (0.1, 0.4, 0.5) with identical means and standard deviations.
    """
    z2 = gaussian_mixture_ztilde(op.x_grid, weights=weights, means=means, sigmas=sigmas)
    # Ensure numerical unit integral on the quadrature grid
    c_norm = 1.0 / np.sum(z2 * op.weights)
    return c_norm * z2
