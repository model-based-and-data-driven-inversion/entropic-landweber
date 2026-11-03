"""Replicate Figure 4 from Burger, Resmerita, Benning (2020).

Reconstruction of probability distribution z2 (violating the source condition):
(a) Ground truth z2 and u^{k*} for perfect data (sigma = 0, k* = 501)
(b) Monotonic energy decrease of objective over 501 iterations
(c) Ground truth z2 and u^{k*} for noisy data (sigma = 1/50, k* = 46 via discrepancy principle)
(d) Energy decrease of objective over 46 iterations and discrepancy threshold delta
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from entropic_landweber.algorithms import entropic_landweber
from entropic_landweber.ground_truth import fourier_ground_truth_z2
from entropic_landweber.operators import FourierSamplingOperator
from entropic_landweber.plotting import plot_figure4


def main() -> None:
    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up Fourier sampling operator on [-10, 10]...")
    op = FourierSamplingOperator(n_samples=16, a=10.0, n_grid=4001)

    print("Constructing ground truth z2 (violates source condition)...")
    z2 = fourier_ground_truth_z2(op)
    y_exact = op.apply(z2)

    # Initial function: u0(x) = 1/20 for x in [-10, 10]
    u0 = np.full_like(op.x_grid, 1.0 / 20.0)
    lam = 9.0 / (10.0 * np.sqrt(2.0 * np.pi))

    # 1. Exact data (sigma = 0): 501 iterations
    print("Running Entropic Landweber for exact data (501 iterations)...")
    res_exact = entropic_landweber(
        op,
        y=y_exact,
        u0=u0,
        lam=lam,
        n_iterations=500,
        m=1,
        z_true=z2,
    )

    # 2. Noisy data (sigma = 1/50): stops by discrepancy principle at k=46
    sigma_noise = 1.0 / 50.0
    rng = np.random.default_rng(3666)
    noise = rng.normal(0.0, sigma_noise, op.n_samples)
    delta_noisy = 0.5 * float(np.sum(noise ** 2))
    y_noisy = y_exact + noise

    print(f"Running Entropic Landweber for noisy data (sigma = 1/50, delta = {delta_noisy:.4e})...")
    res_noisy = entropic_landweber(
        op,
        y=y_noisy,
        u0=u0,
        lam=lam,
        n_iterations=500,
        m=1,
        delta=delta_noisy,
        tau_discrepancy=1.0,
        z_true=z2,
    )

    print(f"Noisy run stopped at iteration {res_noisy.k_star} with energy {res_noisy.energies[-1]:.7f}")

    png_path = figures_dir / "figure4.png"
    pdf_path = figures_dir / "figure4.pdf"

    print(f"Generating Figure 4 and saving to {png_path}...")
    fig = plot_figure4(
        x_grid=op.x_grid,
        z2=z2,
        u_exact=res_exact.u_final,
        energies_exact=res_exact.energies,
        u_noisy=res_noisy.u_final,
        energies_noisy=res_noisy.energies,
        delta_noisy=delta_noisy,
        k_noisy=res_noisy.k_star,
        save_path=png_path,
    )
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print("Figure 4 generated successfully.")


if __name__ == "__main__":
    main()
