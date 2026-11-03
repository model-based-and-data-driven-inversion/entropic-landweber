"""Master script to replicate all figures and numerical results from Burger et al. (2020).

Executes:
1. Figure 1: Comparison on Integral Equations (Kernels 1, 2, 3)
2. Figure 2: Ground truth distributions z1 and z2 on [-2, 2]
3. Figure 3: Reconstruction and energy decay for z1
4. Figure 4: Reconstruction and energy decay for z2
5. Figure 5: L^1 convergence analysis confirming Proposition 4.1
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_figure1
import run_figure2
import run_figure3
import run_figure4
import run_figure5


def main() -> None:
    print("=" * 70)
    print("Replicating All Results from Burger, Resmerita, Benning (2020)")
    print("=" * 70)

    print("\n[1/5] Replicating Figure 1 (Integral Equations)...")
    run_figure1.main()

    print("\n[2/5] Replicating Figure 2 (Ground Truth Distributions)...")
    run_figure2.main()

    print("\n[3/5] Replicating Figure 3 (Reconstruction of z1)...")
    run_figure3.main()

    print("\n[4/5] Replicating Figure 4 (Reconstruction of z2)...")
    run_figure4.main()

    print("\n[5/5] Replicating Figure 5 (L1 Error Analysis)...")
    run_figure5.main()

    print("\n" + "=" * 70)
    print("All figures successfully replicated and saved in 'figures/'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
