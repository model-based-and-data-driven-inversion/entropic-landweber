"""Iterative regularization algorithms for linear ill-posed problems.

Includes:
1. Entropic Landweber method (Proposition 3.1, Eq. 14)
2. Randomized / Stochastic Entropic Landweber method
3. Projected Landweber method
4. Expectation-Maximization (EM / Richardson-Lucy) algorithm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .functionals import l1_error, kullback_leibler_divergence


@dataclass
class IterationResult:
    """Container for iteration outputs and convergence history."""

    u_final: np.ndarray
    k_star: int
    energies: np.ndarray
    l1_errors: np.ndarray | None = None
    kl_divergences: np.ndarray | None = None
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    stopped_by_discrepancy: bool = False
    delta: float | None = None


def entropic_landweber(
    op: Any,
    y: np.ndarray,
    u0: np.ndarray,
    lam: float,
    n_iterations: int = 1000,
    m: int = 0,
    delta: float | None = None,
    tau_discrepancy: float = 1.0,
    z_true: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    track_history: bool = True,
    use_cumulative_form: bool = True,
) -> IterationResult:
    r"""Run the Entropic Landweber iteration (Proposition 3.1, Eq. 14 / Section 6.2):

    When `use_cumulative_form=True` (Section 6.2, page 17):
    .. math::
        \tilde{y}_{k+1} = \tilde{y}_k + A u_k, \quad \tilde{y}_0 = 0
        u_{k+1}(x) = c_k^m \cdots c_0^m u_0(x) \exp\left( \lambda A^*((k+1)y - \tilde{y}_{k+1})(x) \right)

    This formulation keeps the marginal computational cost for evaluating the
    forward operator constant per iteration while computing the iterate directly
    from :math:`u_0`.

    When `use_cumulative_form=False` (Eq. 14):
    .. math::
        u_{k+1} = u_k c_k^m \exp\left( \lambda A^*(y - A u_k) \right)

    where:
    .. math::
        c_k^m = \begin{cases} 1 & \text{if } m = 0 \\
        \frac{1}{\int_\Omega u_k \exp\left( \lambda A^*(y - A u_k) \right) \mathrm{d}t} & \text{if } m = 1 \end{cases}

    Parameters
    ----------
    op : Operator
        Linear operator providing `apply(u)` and `apply_adjoint(v)`.
    y : np.ndarray
        Measurement vector.
    u0 : np.ndarray
        Initial non-negative starting function (bounded away from 0).
    lam : float
        Step size parameter \lambda = 1/c.
    n_iterations : int, optional
        Maximum number of iterations, by default 1000.
    m : int, optional
        Constraint index: 0 (unconstrained mass) or 1 (probability density).
    delta : float | None, optional
        Discrepancy principle threshold: stops when energy <= delta.
    tau_discrepancy : float, optional
        Discrepancy tolerance multiplier, default 1.0.
    z_true : np.ndarray | None, optional
        Ground truth function for tracking L^1 errors.
    weights : np.ndarray | None, optional
        Quadrature weights. If None, retrieved from operator if available.
    track_history : bool, optional
        Whether to record energy and error at every iteration.
    use_cumulative_form : bool, optional
        Whether to use the recursive accumulator \tilde{y}_{k+1} = \tilde{y}_k + A u_k
        from Section 6.2, by default True.

    Returns
    -------
    IterationResult
        Execution history and final iterate.
    """
    if weights is None and hasattr(op, "weights"):
        weights = op.weights
    elif weights is None and hasattr(op, "w_y"):
        weights = op.w_y

    u = np.asarray(u0, dtype=float).copy()
    if m == 1 and weights is not None:
        u = u / np.sum(u * weights)
    u_init = u.copy()

    # Recursive accumulator \tilde{y}_k = \sum_{l=0}^{k-1} A u_l (Section 6.2, page 17)
    tilde_y = np.zeros_like(y, dtype=complex if np.iscomplexobj(y) else float)

    energies = []
    l1_errs = [] if z_true is not None else None
    kl_divs = [] if z_true is not None else None

    stopped_by_discrepancy = False
    k_star = n_iterations

    # Threshold for discrepancy principle: energy <= delta or ||Au - y|| < sqrt(tau)*delta
    # In Section 6.2, paper defines delta = 0.5 * ||eta||^2 and stops when energy <= delta
    discrepancy_threshold = delta * tau_discrepancy if delta is not None else None

    for k in range(n_iterations):
        Au = op.apply(u)
        res = y - Au
        energy = 0.5 * float(np.sum(np.abs(res) ** 2))

        if track_history:
            energies.append(energy)
            if z_true is not None:
                l1_errs.append(l1_error(u, z_true, weights=weights))
                kl_divs.append(kullback_leibler_divergence(z_true, u, weights=weights))

        # Check stopping rule (discrepancy principle)
        if discrepancy_threshold is not None and energy <= discrepancy_threshold:
            stopped_by_discrepancy = True
            k_star = k
            break

        if use_cumulative_form:
            # Recursive accumulator: \tilde{y}_{k+1} = \tilde{y}_k + A u_k
            tilde_y = tilde_y + Au
            cum_res = (k + 1) * y - tilde_y
            cum_grad = op.apply_adjoint(cum_res)
            u_next = u_init * np.exp(lam * cum_grad)
        else:
            # Step-wise multiplicative update: u_{k+1} = u_k * exp(\lambda A^*(y - A u_k))
            grad = op.apply_adjoint(res)
            u_next = u * np.exp(lam * grad)

        if m == 1:
            total_mass = np.sum(u_next * weights) if weights is not None else np.sum(u_next)
            u = u_next / total_mass
        else:
            u = u_next

    # If completed without early stopping, compute final energy
    if not stopped_by_discrepancy:
        Au = op.apply(u)
        res = y - Au
        energy = 0.5 * float(np.sum(np.abs(res) ** 2))
        energies.append(energy)
        if z_true is not None:
            l1_errs.append(l1_error(u, z_true, weights=weights))
            kl_divs.append(kullback_leibler_divergence(z_true, u, weights=weights))

    return IterationResult(
        u_final=u,
        k_star=k_star,
        energies=np.array(energies, dtype=float),
        l1_errors=np.array(l1_errs, dtype=float) if l1_errs is not None else None,
        kl_divergences=np.array(kl_divs, dtype=float) if kl_divs is not None else None,
        stopped_by_discrepancy=stopped_by_discrepancy,
        delta=delta,
    )


def randomized_entropic_landweber(
    op: Any,
    y: np.ndarray,
    u0: np.ndarray,
    lam: float,
    n_iterations: int = 1000,
    m: int = 0,
    z_true: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    seed: int | None = 42,
) -> IterationResult:
    r"""Run the Randomized / Stochastic Entropic Landweber method (Section 6.1):

    .. math::
        u_{k+1} = u_k c_k^m \exp\left( \lambda M A_{J(k)}^* (y_{J(k)} - A_{J(k)} u_k) \right)

    where J(k) is chosen uniformly at random from {1, ..., M}.

    Parameters
    ----------
    op : FredholmOperator
        Linear operator providing `row_action(j, u)`.
    y : np.ndarray
        Measurement vector.
    u0 : np.ndarray
        Initial non-negative starting function.
    lam : float
        Step size parameter \lambda.
    n_iterations : int, optional
        Number of stochastic iterations, by default 1000.
    m : int, optional
        Constraint index: 0 or 1.
    z_true : np.ndarray | None, optional
        Ground truth function for tracking L^1 errors.
    weights : np.ndarray | None, optional
        Quadrature weights.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    IterationResult
        Execution history and final iterate.
    """
    if weights is None and hasattr(op, "weights"):
        weights = op.weights
    elif weights is None and hasattr(op, "w_y"):
        weights = op.w_y

    rng = np.random.default_rng(seed)
    u = np.asarray(u0, dtype=float).copy()
    if m == 1 and weights is not None:
        u = u / np.sum(u * weights)

    M = len(y)
    energies = []
    l1_errs = [] if z_true is not None else None

    for k in range(n_iterations):
        if z_true is not None:
            l1_errs.append(l1_error(u, z_true, weights=weights))

        # Sample equation index uniformly from {0, ..., M-1}
        j = rng.integers(0, M)
        val_j, adj_j = op.row_action(j, u)
        res_j = y[j] - val_j

        # Update step: \lambda M A_j^*(res_j)
        grad_j = adj_j * res_j
        u_next = u * np.exp(lam * float(M) * grad_j)

        if m == 1:
            total_mass = np.sum(u_next * weights) if weights is not None else np.sum(u_next)
            u = u_next / total_mass
        else:
            u = u_next

    if z_true is not None:
        l1_errs.append(l1_error(u, z_true, weights=weights))

    return IterationResult(
        u_final=u,
        k_star=n_iterations,
        energies=np.array(energies, dtype=float),
        l1_errors=np.array(l1_errs, dtype=float) if l1_errs is not None else None,
    )


def projected_landweber(
    op: Any,
    y: np.ndarray,
    u0: np.ndarray,
    tau: float,
    n_iterations: int = 1000,
    z_true: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> IterationResult:
    r"""Run the Projected Landweber iteration (Section 6.1):

    .. math::
        u_{k+1} = \left( u_k - \tau A^*(A u_k - y) \right)_+
                = \max\left( 0, u_k + \tau A^*(y - A u_k) \right)

    Parameters
    ----------
    op : Operator
        Linear operator.
    y : np.ndarray
        Measurement vector.
    u0 : np.ndarray
        Initial non-negative starting function.
    tau : float
        Landweber step size parameter.
    n_iterations : int, optional
        Maximum number of iterations, by default 1000.
    z_true : np.ndarray | None, optional
        Ground truth for tracking L^1 error.
    weights : np.ndarray | None, optional
        Quadrature weights.

    Returns
    -------
    IterationResult
        Execution history and final iterate.
    """
    if weights is None and hasattr(op, "weights"):
        weights = op.weights
    elif weights is None and hasattr(op, "w_y"):
        weights = op.w_y

    u = np.asarray(u0, dtype=float).copy()
    energies = []
    l1_errs = [] if z_true is not None else None

    for k in range(n_iterations):
        if z_true is not None:
            l1_errs.append(l1_error(u, z_true, weights=weights))

        res = y - op.apply(u)
        grad = op.apply_adjoint(res)
        u = np.maximum(0.0, u + tau * grad)

    if z_true is not None:
        l1_errs.append(l1_error(u, z_true, weights=weights))

    return IterationResult(
        u_final=u,
        k_star=n_iterations,
        energies=np.array(energies, dtype=float),
        l1_errors=np.array(l1_errs, dtype=float) if l1_errs is not None else None,
    )


def expectation_maximization(
    op: Any,
    y: np.ndarray,
    u0: np.ndarray,
    n_iterations: int = 1000,
    z_true: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> IterationResult:
    r"""Run the Expectation-Maximization (Richardson-Lucy) algorithm:

    .. math::
        u_{k+1} = \frac{u_k}{A^* 1} A^*\left( \frac{y}{A u_k} \right)

    Parameters
    ----------
    op : Operator
        Linear operator with positive kernel/forward operator.
    y : np.ndarray
        Measurement vector.
    u0 : np.ndarray
        Initial strictly positive starting function.
    n_iterations : int, optional
        Maximum number of iterations, by default 1000.
    z_true : np.ndarray | None, optional
        Ground truth function for tracking L^1 errors.
    weights : np.ndarray | None, optional
        Quadrature weights.

    Returns
    -------
    IterationResult
        Execution history and final iterate.
    """
    if weights is None and hasattr(op, "weights"):
        weights = op.weights
    elif weights is None and hasattr(op, "w_y"):
        weights = op.w_y

    u = np.asarray(u0, dtype=float).copy()
    ones_measurement = np.ones(len(y), dtype=float)
    Astar_1 = op.apply_adjoint(ones_measurement)
    # Avoid zero division
    Astar_1 = np.where(Astar_1 > 0, Astar_1, 1.0)

    energies = []
    l1_errs = [] if z_true is not None else None

    for k in range(n_iterations):
        if z_true is not None:
            l1_errs.append(l1_error(u, z_true, weights=weights))

        Au = op.apply(u)
        # Avoid division by zero in ratio y / Au
        safe_Au = np.where(Au > 1e-15, Au, 1e-15)
        ratio = y / safe_Au
        adjoint_ratio = op.apply_adjoint(ratio)
        u = (u / Astar_1) * adjoint_ratio

    if z_true is not None:
        l1_errs.append(l1_error(u, z_true, weights=weights))

    return IterationResult(
        u_final=u,
        k_star=n_iterations,
        energies=np.array(energies, dtype=float),
        l1_errors=np.array(l1_errs, dtype=float) if l1_errs is not None else None,
    )
