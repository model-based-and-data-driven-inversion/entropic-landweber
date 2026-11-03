"""Replicate Figure 2 from Burger, Resmerita, Benning (2020).

Plots the two ground truth functions z1 and z2 on [-2, 2]:
(a) z1 satisfying the source condition 1 + ln z1 in R(A*)
(b) z2 violating the source condition
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from entropic_landweber.ground_truth import (
    fourier_ground_truth_z1,
    fourier_ground_truth_z2,
)
from entropic_landweber.operators import FourierSamplingOperator
from entropic_landweber.plotting import plot_figure2


def main() -> None:
    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Initialising Fourier sampling operator on [-10, 10]...")
    op = FourierSamplingOperator(n_samples=16, a=10.0, n_grid=4001)

    print("Constructing ground truth z1 (satisfies source condition)...")
    z1 = fourier_ground_truth_z1(op)

    print("Constructing ground truth z2 (violates source condition)...")
    z2 = fourier_ground_truth_z2(op)

    png_path = figures_dir / "figure2.png"
    pdf_path = figures_dir / "figure2.pdf"

    print(f"Generating Figure 2 and saving to {png_path}...")
    fig = plot_figure2(op.x_grid, z1, z2, save_path=png_path)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print("Figure 2 generated successfully.")


if __name__ == "__main__":
    main()
