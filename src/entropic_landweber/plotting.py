"""Plotting utilities to replicate Figures 1-5 from the paper.

Matches the layout, style, axis limits, line styles, and colours
from Burger, Resmerita, Benning (Inverse Problems, 2020).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import numpy as np
import matplotlib.pyplot as plt


def set_paper_style() -> None:
    """Set clean matplotlib publication style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "lines.linewidth": 1.5,
            "axes.grid": False,
        }
    )


def plot_figure1(
    case1_results: Mapping[str, np.ndarray],
    case2_results: Mapping[str, np.ndarray],
    case3_results: Mapping[str, np.ndarray],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Replicate Figure 1: L^1 error vs iteration number for the three test cases.

    Parameters
    ----------
    case1_results : dict
        Dict containing error arrays for 'entropy', 'randomized', 'em', 'projection'.
    case2_results : dict
        Dict containing error arrays for Case 2.
    case3_results : dict
        Dict containing error arrays for Case 3.
    save_path : str | Path | None, optional
        File path to save the generated figure.

    Returns
    -------
    plt.Figure
    """
    set_paper_style()

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.28, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # Case 3 centered in bottom row
    ax3 = fig.add_subplot(gs[1, :])
    # Place ax3 with smaller width matching top plots
    pos3 = ax3.get_position()
    center_x = (pos3.x0 + pos3.x1) / 2.0
    half_width = (ax1.get_position().x1 - ax1.get_position().x0) / 2.0
    ax3.set_position([center_x - half_width, pos3.y0, 2 * half_width, pos3.height])

    # Case 1
    ax1.plot(case1_results["entropy"], color="red", label="Entropy", lw=2.0)
    ax1.plot(case1_results["randomized"], color="black", label="Randomized Entropy", lw=1.5)
    ax1.plot(case1_results["em"], color="blue", label="EM", lw=1.8)
    ax1.plot(case1_results["projection"], color="lime", label="Projection", lw=1.8)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 1.5)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("L1 Error")
    ax1.legend(loc="upper right", frameon=True)

    # Case 2
    ax2.plot(case2_results["entropy"], color="red", label="Entropy", lw=2.0)
    ax2.plot(case2_results["randomized"], color="black", label="Randomized Entropy", lw=1.5)
    ax2.plot(case2_results["em"], color="blue", label="EM", lw=1.8)
    ax2.plot(case2_results["projection"], color="lime", label="Projection", lw=1.8)
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 0.3)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("L1 Error")
    ax2.legend(loc="upper right", frameon=True)

    # Case 3
    ax3.plot(case3_results["entropy"], color="red", label="Entropy", lw=2.0)
    ax3.plot(case3_results["randomized"], color="black", label="Randomized Entropy", lw=1.5)
    ax3.plot(case3_results["em"], color="blue", label="EM", lw=1.8)
    ax3.plot(case3_results["projection"], color="lime", label="Projection", lw=1.8)
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(0, 1.6)
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("L1 Error")
    ax3.legend(loc="upper right", frameon=True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_figure2(
    x_grid: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Replicate Figure 2: The two ground truth functions z1 and z2 on [-2, 2].

    Parameters
    ----------
    x_grid : np.ndarray
        Spatial grid coordinates.
    z1 : np.ndarray
        Ground truth function z1 satisfying source condition (30).
    z2 : np.ndarray
        Ground truth function z2 violating source condition (30).
    save_path : str | Path | None, optional
        Save path.

    Returns
    -------
    plt.Figure
    """
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    mask = (x_grid >= -2.0) & (x_grid <= 2.0)
    x_sub = x_grid[mask]

    # Subplot (a): z1
    ax1.plot(x_sub, z1[mask], color="#1f77b4", lw=1.8)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(0, 0.2)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$z_1(x)$")
    ax1.text(0.5, -0.22, "(a)", transform=ax1.transAxes, ha="center", fontsize=11)

    # Subplot (b): z2
    ax2.plot(x_sub, z2[mask], color="#1f77b4", lw=1.8)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(0, 2.0)
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$z_2(x)$")
    ax2.text(0.5, -0.22, "(b)", transform=ax2.transAxes, ha="center", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_figure3(
    x_grid: np.ndarray,
    z1: np.ndarray,
    u_exact: np.ndarray,
    energies_exact: np.ndarray,
    u_noisy: np.ndarray,
    energies_noisy: np.ndarray,
    delta_noisy: float,
    k_noisy: int,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Replicate Figure 3: Results of algorithm (4) for measurements based on z1."""
    set_paper_style()
    fig, axs = plt.subplots(2, 2, figsize=(8.5, 7.0))

    mask = (x_grid >= -2.0) & (x_grid <= 2.0)
    x_sub = x_grid[mask]

    # (a) Reconstruction for sigma = 0
    final_energy_exact = energies_exact[-1]
    axs[0, 0].plot(x_sub, z1[mask], "--", color="#1f77b4", lw=1.5, label="z(x)")
    axs[0, 0].plot(x_sub, u_exact[mask], "-", color="#d95f02", lw=1.5, label="u(x)")
    axs[0, 0].set_xlim(-2, 2)
    axs[0, 0].set_ylim(0, 0.2)
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_title(
        f"Iteration no. {len(energies_exact)}, energy/$\\delta$ = {final_energy_exact:.4e}/0",
        fontsize=9,
    )
    axs[0, 0].legend(loc="upper right", frameon=True)
    axs[0, 0].text(0.5, -0.25, "(a)", transform=axs[0, 0].transAxes, ha="center", fontsize=11)

    # (b) Energy decrease for sigma = 0
    k_axis_exact = np.arange(len(energies_exact))
    axs[0, 1].plot(k_axis_exact, energies_exact * 1e3, "--", color="#1f77b4", lw=1.5, label="Energy")
    axs[0, 1].set_xlim(0, len(energies_exact) - 1)
    axs[0, 1].set_ylim(0, 6.0)
    axs[0, 1].set_xlabel("No. of iterations")
    axs[0, 1].set_ylabel(r"Energy [$\times 10^{-3}$]")
    axs[0, 1].set_title("Energy", fontsize=10)
    axs[0, 1].text(0.5, -0.25, "(b)", transform=axs[0, 1].transAxes, ha="center", fontsize=11)

    # (c) Reconstruction for sigma = 1/500
    final_energy_noisy = energies_noisy[-1]
    axs[1, 0].plot(x_sub, z1[mask], "--", color="#1f77b4", lw=1.5, label="z(x)")
    axs[1, 0].plot(x_sub, u_noisy[mask], "-", color="#d95f02", lw=1.5, label="u(x)")
    axs[1, 0].set_xlim(-2, 2)
    axs[1, 0].set_ylim(0, 0.2)
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_title(
        f"Iteration no. {k_noisy}, energy/$\\delta$ = {final_energy_noisy:.4e}/{delta_noisy:.4e}",
        fontsize=9,
    )
    axs[1, 0].legend(loc="upper right", frameon=True)
    axs[1, 0].text(0.5, -0.25, "(c)", transform=axs[1, 0].transAxes, ha="center", fontsize=11)

    # (d) Energy decrease for sigma = 1/500
    k_axis_noisy = np.arange(len(energies_noisy))
    axs[1, 1].plot(k_axis_noisy, energies_noisy * 1e3, "--", color="#1f77b4", lw=1.5, label="Energy")
    axs[1, 1].axhline(delta_noisy * 1e3, color="red", lw=1.2, label=r"$\delta$")
    axs[1, 1].set_xlim(0, max(k_noisy, len(energies_noisy) - 1))
    axs[1, 1].set_ylim(0, 5.5)
    axs[1, 1].set_xlabel("No. of iterations")
    axs[1, 1].set_ylabel(r"Energy [$\times 10^{-3}$]")
    axs[1, 1].set_title("Energy", fontsize=10)
    axs[1, 1].legend(loc="upper right", frameon=True)
    axs[1, 1].text(0.5, -0.25, "(d)", transform=axs[1, 1].transAxes, ha="center", fontsize=11)

    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_figure4(
    x_grid: np.ndarray,
    z2: np.ndarray,
    u_exact: np.ndarray,
    energies_exact: np.ndarray,
    u_noisy: np.ndarray,
    energies_noisy: np.ndarray,
    delta_noisy: float,
    k_noisy: int,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Replicate Figure 4: Results of algorithm (4) for measurements based on z2."""
    set_paper_style()
    fig, axs = plt.subplots(2, 2, figsize=(8.5, 7.0))

    mask = (x_grid >= -2.0) & (x_grid <= 2.0)
    x_sub = x_grid[mask]

    # (a) Reconstruction for sigma = 0
    final_energy_exact = energies_exact[-1]
    axs[0, 0].plot(x_sub, z2[mask], "--", color="#1f77b4", lw=1.5, label="z(x)")
    axs[0, 0].plot(x_sub, u_exact[mask], "-", color="#d95f02", lw=1.5, label="u(x)")
    axs[0, 0].set_xlim(-2, 2)
    axs[0, 0].set_ylim(0, 2.0)
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_title(
        f"Iteration no. {len(energies_exact)}, energy/$\\delta$ = {final_energy_exact:.3e}/0",
        fontsize=9,
    )
    axs[0, 0].legend(loc="upper right", frameon=True)
    axs[0, 0].text(0.5, -0.25, "(a)", transform=axs[0, 0].transAxes, ha="center", fontsize=11)

    # (b) Energy decrease for sigma = 0
    k_axis_exact = np.arange(len(energies_exact))
    axs[0, 1].plot(k_axis_exact, energies_exact, "--", color="#1f77b4", lw=1.5, label="Energy")
    axs[0, 1].set_xlim(0, len(energies_exact) - 1)
    axs[0, 1].set_ylim(0, 0.38)
    axs[0, 1].set_xlabel("No. of iterations")
    axs[0, 1].set_ylabel("Energy")
    axs[0, 1].set_title("Energy", fontsize=10)
    axs[0, 1].text(0.5, -0.25, "(b)", transform=axs[0, 1].transAxes, ha="center", fontsize=11)

    # (c) Reconstruction for sigma = 1/50
    final_energy_noisy = energies_noisy[-1]
    axs[1, 0].plot(x_sub, z2[mask], "--", color="#1f77b4", lw=1.5, label="z(x)")
    axs[1, 0].plot(x_sub, u_noisy[mask], "-", color="#d95f02", lw=1.5, label="u(x)")
    axs[1, 0].set_xlim(-2, 2)
    axs[1, 0].set_ylim(0, 2.0)
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_title(
        f"Iteration no. {k_noisy}, energy/$\\delta$ = {final_energy_noisy:.7f}/{delta_noisy:.7f}",
        fontsize=9,
    )
    axs[1, 0].legend(loc="upper right", frameon=True)
    axs[1, 0].text(0.5, -0.25, "(c)", transform=axs[1, 0].transAxes, ha="center", fontsize=11)

    # (d) Energy decrease for sigma = 1/50
    k_axis_noisy = np.arange(len(energies_noisy))
    axs[1, 1].plot(k_axis_noisy, energies_noisy, "--", color="#1f77b4", lw=1.5, label="Energy")
    axs[1, 1].axhline(delta_noisy, color="red", lw=1.2, label=r"$\delta$")
    axs[1, 1].set_xlim(0, max(k_noisy, len(energies_noisy) - 1))
    axs[1, 1].set_ylim(0, 0.35)
    axs[1, 1].set_xlabel("No. of iterations")
    axs[1, 1].set_ylabel("Energy")
    axs[1, 1].set_title("Energy", fontsize=10)
    axs[1, 1].legend(loc="upper right", frameon=True)
    axs[1, 1].text(0.5, -0.25, "(d)", transform=axs[1, 1].transAxes, ha="center", fontsize=11)

    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_figure5(
    l1_errors_z1: np.ndarray,
    l1_errors_z2: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Replicate Figure 5: L^1 error vs iteration number for exact data."""
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    k1 = np.arange(len(l1_errors_z1))
    ax1.plot(k1, l1_errors_z1, color="#1f77b4", lw=1.8)
    ax1.set_xlim(0, len(l1_errors_z1) - 1)
    ax1.set_ylim(0, 0.16)
    ax1.set_xlabel("No. of iterations k")
    ax1.set_ylabel(r"$\|u^k - z_1\|_{L^1}$")
    ax1.text(0.5, -0.22, "(a)", transform=ax1.transAxes, ha="center", fontsize=11)

    k2 = np.arange(len(l1_errors_z2))
    ax2.plot(k2, l1_errors_z2, color="#1f77b4", lw=1.8)
    ax2.set_xlim(0, len(l1_errors_z2) - 1)
    ax2.set_ylim(0, 2.0)
    ax2.set_xlabel("No. of iterations k")
    ax2.set_ylabel(r"$\|u^k - z_2\|_{L^1}$")
    ax2.text(0.5, -0.22, "(b)", transform=ax2.transAxes, ha="center", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
