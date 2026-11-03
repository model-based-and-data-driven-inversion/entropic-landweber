"""Functionals and distances for entropic regularization.

Includes the Boltzmann-Shannon entropy functional, the Kullback-Leibler
divergence (Bregman distance), the non-linear distance functional D(u, v),
and L^1 norms and errors.
"""

from __future__ import annotations

import numpy as np


def shannon_entropy(
    u: np.ndarray,
    weights: np.ndarray | None = None,
    dx: float | None = None,
) -> float:
    r"""Compute the negative Boltzmann-Shannon entropy functional:

    .. math::
        f(u) = \int_\Omega u(t) \ln u(t) \, \mathrm{d}t

    Parameters
    ----------
    u : np.ndarray
        Non-negative function values on the grid.
    weights : np.ndarray | None, optional
        Quadrature weights. If None and dx is provided, uniform weights are used.
    dx : float | None, optional
        Grid spacing if weights is None.

    Returns
    -------
    float
        Entropy value f(u). Returns +inf if any elements are strictly negative.
    """
    if np.any(u < 0):
        return float("inf")

    # Convention: 0 * ln(0) = 0
    safe_u = np.where(u > 0, u, 1.0)
    log_u = np.where(u > 0, np.log(safe_u), 0.0)
    integrand = u * log_u

    if weights is not None:
        return float(np.sum(integrand * weights))
    if dx is not None:
        return float(np.sum(integrand) * dx)
    return float(np.sum(integrand))


def kullback_leibler_divergence(
    v: np.ndarray,
    u: np.ndarray,
    weights: np.ndarray | None = None,
    dx: float | None = None,
) -> float:
    r"""Compute the Kullback-Leibler divergence (Bregman distance of entropy):

    .. math::
        d(v, u) = \int_\Omega \left( v(t) \ln\frac{v(t)}{u(t)} - v(t) + u(t) \right) \mathrm{d}t

    Parameters
    ----------
    v : np.ndarray
        First function argument (target / reference).
    u : np.ndarray
        Second function argument (base iterate).
    weights : np.ndarray | None, optional
        Quadrature weights.
    dx : float | None, optional
        Grid spacing if weights is None.

    Returns
    -------
    float
        Bregman distance d(v, u) >= 0.
    """
    if np.any(u <= 0):
        return float("inf")
    if np.any(v < 0):
        return float("inf")

    safe_v = np.where(v > 0, v, 1.0)
    log_ratio = np.where(v > 0, np.log(safe_v / u), 0.0)
    integrand = v * log_ratio - v + u

    if weights is not None:
        return float(np.sum(integrand * weights))
    if dx is not None:
        return float(np.sum(integrand) * dx)
    return float(np.sum(integrand))


def d_functional(
    u: np.ndarray,
    v: np.ndarray,
    Au: np.ndarray,
    Av: np.ndarray,
    c: float,
    weights: np.ndarray | None = None,
    dx: float | None = None,
) -> float:
    r"""Compute the non-linear functional D(u, v):

    .. math::
        D(u, v) = c \, d(u, v) - \frac{1}{2} \|Au - Av\|^2

    Parameters
    ----------
    u : np.ndarray
        First function argument.
    v : np.ndarray
        Second function argument.
    Au : np.ndarray
        Forward evaluation Au.
    Av : np.ndarray
        Forward evaluation Av.
    c : float
        Regularization penalty parameter (c = 1 / lambda).
    weights : np.ndarray | None, optional
        Quadrature weights for d(u, v).
    dx : float | None, optional
        Grid spacing for d(u, v).

    Returns
    -------
    float
        Functional value D(u, v).
    """
    kl = kullback_leibler_divergence(u, v, weights=weights, dx=dx)
    residual_diff_sq = np.sum(np.abs(Au - Av) ** 2)
    return float(c * kl - 0.5 * residual_diff_sq)


def l1_norm(
    u: np.ndarray,
    weights: np.ndarray | None = None,
    dx: float | None = None,
) -> float:
    r"""Compute the L^1 norm:

    .. math::
        \|u\|_{L^1} = \int_\Omega |u(t)| \, \mathrm{d}t

    Parameters
    ----------
    u : np.ndarray
        Function values on the grid.
    weights : np.ndarray | None, optional
        Quadrature weights.
    dx : float | None, optional
        Grid spacing if weights is None.

    Returns
    -------
    float
        L^1 norm.
    """
    abs_u = np.abs(u)
    if weights is not None:
        return float(np.sum(abs_u * weights))
    if dx is not None:
        return float(np.sum(abs_u) * dx)
    return float(np.sum(abs_u))


def l1_error(
    u: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray | None = None,
    dx: float | None = None,
) -> float:
    r"""Compute the L^1 error between iterate u and ground truth z:

    .. math::
        \|u - z\|_{L^1} = \int_\Omega |u(t) - z(t)| \, \mathrm{d}t
    """
    return l1_norm(u - z, weights=weights, dx=dx)
