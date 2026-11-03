"""Replicate Figure 5 from Burger, Resmerita, Benning (2020).

L^1-norm of the difference of iterates u^k and ground truth:
(a) ||u^k - z1||_{L^1} over 201 iterations (convergence confirmed)
(b) ||u^k - z2||_{L^1} over 501 iterations (non-convergence due to violation of source condition)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from entropic_landweber.algorithms import entropic_landweber
from entropic_landweber.ground_truth import (
    fourier_ground_truth_z1,
    fourier_ground_truth_z2,
)
from entropic_landweber.operators import FourierSamplingOperator
from entropic_landweber.plotting import plot_figure5


def main() -> None:
    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up Fourier sampling operator on [-10, 10]...")
    op = FourierSamplingOperator(n_samples=16, a=10.0, n_grid=4001)

    u0 = np.full_like(op.x_grid, 1.0 / 20.0)
    lam = 9.0 / (10.0 * np.sqrt(2.0 * np.pi))

    # Case z1: satisfies source condition (30)
    print("Running Entropic Landweber for z1 (201 iterations)...")
    z1 = fourier_ground_truth_z1(op)
    y1 = op.apply(z1)
    res_z1 = entropic_landweber(
        op,
        y=y1,
        u0=u0,
        lam=lam,
        n_iterations=200,
        m=1,
        z_true=z1,
    )

    # Case z2: violates source condition (30)
    print("Running Entropic Landweber for z2 (501 iterations)...")
    z2 = fourier_ground_truth_z2(op)
    y2 = op.apply(z2)
    res_z2 = entropic_landweber(
        op,
        y=y2,
        u0=u0,
        lam=lam,
        n_iterations=500,
        m=1,
        z_true=z2,
    )

    png_path = figures_dir / "figure5.png"
    pdf_path = figures_dir / "figure5.pdf"

    print(f"Generating Figure 5 and saving to {png_path}...")
    fig = plot_figure5(
        l1_errors_z1=res_z1.l1_errors,
        l1_errors_z2=res_z2.l1_errors,
        save_path=png_path,
    )
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print("Figure 5 generated successfully.")


if __name__ == "__main__":
    main()
