# Entropic Landweber Method for Linear Ill-Posed Problems

A Python package and replication repository for the algorithms, numerical experiments, and visualisations presented in:

> **Martin Burger, Elena Resmerita, and Martin Benning (2020)**  
> *An entropic Landweber method for linear ill-posed problems*  
> *Inverse Problems*, 36(1): 015009.  
> DOI: [10.1088/1361-6420/ab5c49](https://doi.org/10.1088/1361-6420/ab5c49)

---

## Overview

The repository provides modular, verified Python implementations of the entropic Landweber regularisation scheme for solving linear ill-posed operator equations:

$$
A u = y
$$

where $A: L^1(\Omega) \to Y$ is a bounded linear operator into a Hilbert space $Y$, and the unknown $u$ is constrained to be non-negative ($u \ge 0$) or a continuous probability density ($\int_\Omega u(t) \, \mathrm{d}t = 1$).

### Mathematical Formulation

The iterative scheme arises from linearising the $L^2$ data-fidelity term at the current iterate $u_k$ and regularising with the Kullback–Leibler divergence (Bregman distance induced by the Boltzmann–Shannon entropy):

$$
u_{k+1} = \arg\min_{u \ge 0} \left\lbrace \langle Au - y, Au_k - y \rangle + c \, d(u, u_k) + \chi_m(u) \right\rbrace, \quad m \in \lbrace 0, 1 \rbrace
$$

where $c > 0$ is a penalty parameter ($\lambda = 1/c$ is the effective step size), $d(u, u_k)$ is the Kullback–Leibler divergence:

$$
d(u, u_k) = \int_\Omega \left( u(t) \ln\frac{u(t)}{u_k(t)} - u(t) + u_k(t) \right) \mathrm{d}t
$$

and $\chi_m$ enforces mass constraints:
- **$m = 0$ (Unconstrained mass)**: $\chi_0 \equiv 0$, preserving non-negativity $u \ge 0$.
- **$m = 1$ (Probability density constraint)**: $\chi_1(u) = 0$ if $\int_\Omega u(t) \, \mathrm{d}t = 1$, and $+\infty$ otherwise.

### Explicit Update Rule

Minimising the functional yields the closed-form multiplicative update (Proposition 3.1):

$$
u_{k+1} = u_k c_k^m \exp\left( \lambda A^{\ast}(y - A u_k) \right)
$$

with normalisation scalar:

$$
c_k^m = \begin{cases} 1 & \text{if } m = 0, \\\\ \displaystyle \frac{1}{\int_\Omega u_k(t) \exp\left( \lambda (A^{\ast}(y - A u_k))(t) \right) \mathrm{d}t} & \text{if } m = 1. \end{cases}
$$

#### Constant-Cost Cumulative Formulation (Section 6.2)

As detailed on page 17 of the paper, the update can also be written directly in terms of the initialisation $u_0$ and the cumulative historical evaluations:

$$
u_k(x) = c_{k-1}^m \cdots c_0^m u_0(x) \exp\left( \lambda A^{\ast}(k y - \tilde{y}_k)(x) \right)
$$

with the recursive accumulator:

$$
\tilde{y}_0 = 0, \quad \tilde{y}_{k+1} = \tilde{y}_k + A u_k
$$

Because only a single forward operation $A u_k$ is computed per iteration, this formulation has the advantage that the marginal numerical costs for evaluating the integrals remain constant over time while eliminating accumulated step-wise multiplicative drift. Both formulations are supported in `entropic_landweber` (`use_cumulative_form=True` by default), with trapezoidal quadrature employed throughout for all integral evaluations.

### Implemented Methods

1. **Entropic Landweber (Full Operator)**: Multiplicative gradient descent with Shannon entropy Bregman distance, with or without unit integral normalisation, implementing both the recursive accumulator $\tilde{y}_k$ and step-wise representations.

2. **Randomised / Stochastic Entropic Landweber**: Row-action stochastic mirror descent where operator $A$ is partitioned into $M$ rows, updating via single sampled equations:

$$
u_{k+1} = u_k c_k^m \exp\left( \lambda M A_{J(k)}^{\ast} (y_{J(k)} - A_{J(k)} u_k) \right)
$$

3. **Projected Landweber Iteration**:

$$
u_{k+1} = \left( u_k - \tau A^{\ast}(A u_k - y) \right)_+ = \max\left(0, u_k + \tau A^{\ast}(y - A u_k)\right)
$$

4. **Expectation–Maximisation (EM / Richardson–Lucy)**:

$$
u_{k+1} = \frac{u_k}{A^{\ast} 1} A^{\ast}\left( \frac{y}{A u_k} \right)
$$

---

## Repository Structure

```
entropic-landweber/
├── pyproject.toml              # Project configuration and dependency specifications
├── uv.lock                     # Locked dependency graph generated via Astral uv
├── README.md                   # Technical documentation
├── LICENSE                     # Project licence (GPL-3.0)
├── .gitignore                  # Git ignore rules
├── src/
│   └── entropic_landweber/
│       ├── __init__.py         # Public library interface
│       ├── algorithms.py       # Entropic Landweber, Stochastic, Projected Landweber, EM
│       ├── functionals.py      # Entropy f(u), KL divergence d(u, v), D(u, v), L1 norm
│       ├── operators.py        # Fredholm integral operators and Fourier sampling operator
│       ├── ground_truth.py     # Analytical kernels and test solutions
│       └── plotting.py         # Publication-grade plotting routines
├── scripts/
│   ├── replicate_all.py        # Master script reproducing all five figures
│   ├── run_figure1.py          # Figure 1: Integral equations benchmark
│   ├── run_figure2.py          # Figure 2: Ground-truth probability densities z1 and z2
│   ├── run_figure3.py          # Figure 3: Reconstruction of z1 (satisfying source condition)
│   ├── run_figure4.py          # Figure 4: Reconstruction of z2 (violating source condition)
│   └── run_figure5.py          # Figure 5: L1 convergence rates for exact data
├── tests/
│   ├── test_algorithms.py      # Tests for monotonicity, positivity, and mass preservation
│   ├── test_functionals.py     # Tests for entropy and Bregman distance properties
│   └── test_operators.py       # Adjointness tests <Au, v> = <u, A*v>
└── figures/                    # Replicated publication figures (PNG and vector PDF)
    ├── figure1.png / figure1.pdf
    ├── figure2.png / figure2.pdf
    ├── figure3.png / figure3.pdf
    ├── figure4.png / figure4.pdf
    └── figure5.png / figure5.pdf
```

---

## Installation and Environment Setup

The repository uses modern Python packaging via `pyproject.toml` and [`uv`](https://github.com/astral-sh/uv).

### 1. Using Astral `uv` (Recommended)

To install dependencies using the bundled `uv.lock` file:

```bash
# Create and sync virtual environment
uv sync

# Run all test cases
uv run python -m unittest discover -s tests -v

# Run the master replication script
uv run python scripts/replicate_all.py
```

### 2. Using Standard `pip` and Virtual Environments

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run test suite
python -m unittest discover -s tests -v

# Replicate all figures
python scripts/replicate_all.py
```

Dependencies:
- `numpy >= 1.24`
- `matplotlib >= 3.7`

---

## Replicating Results

The numerical benchmarks described in Section 6 of the paper can be executed individually or all at once.

### Replicate All Figures

```bash
python scripts/replicate_all.py
```

This runs all simulations, saves high-resolution PNG (300 DPI) and vector PDF outputs into the `figures/` directory, and prints iteration metrics to standard output.

### Individual Experiments

#### Figure 1: Benchmark on Integral Equations (Section 6.1)
Compares the convergence of Entropic Landweber, Stochastic Entropic Landweber, Expectation–Maximisation, and Projected Landweber over 1,000 iterations for three kernels on $\Omega = (0, 1)$:
1. $k_1(x, y) = e^{xy}$, exact solution $z_1(x) \propto e^{-x^2 / (2\sigma^2)}$ with $\sigma^2 = 0.01$.
2. $k_2(x, y) = 3 e^{-(x-y)^2 / 0.04}$, exact solution $z_2(x) = 1 - \sum_{l=1}^5 c_l e^{-(x-\mu_l)^2 / (2\sigma^2)}$.
3. $k_3(x, y) = \mathbf{1}_{x \ge y}$ (numerical differentiation), exact solution $z_3(x) \propto e^{-x^2 / (2\sigma^2)}$.

```bash
python scripts/run_figure1.py
```

#### Figure 2: Ground Truth Distributions (Section 6.2)
Visualises the two probability distributions $z_1$ and $z_2$ on $[-2, 2]$:
- $z_1$: Constructed in the range of the adjoint operator ($`1 + \ln z_1 \in \mathcal{R}(A^{\ast})`$), satisfying the source condition (30).
- $z_2$: Three-component Gaussian mixture violating the source condition.

```bash
python scripts/run_figure2.py
```

#### Figure 3: Reconstruction of $z_1$ from Discrete Fourier Samples
Evaluates Entropic Landweber on 16 discrete Fourier measurements of $z_1$:
- Exact data ($\sigma = 0$): 201 iterations achieving residual energy $5.10 \times 10^{-9}$.
- Noisy data ($\sigma = 1/500$): Discrepancy stopping criterion terminates the algorithm when $\frac{1}{2}\Vert Au_k - y^\delta\Vert^2 \le \delta$.

```bash
python scripts/run_figure3.py
```

#### Figure 4: Reconstruction of $z_2$ Violating the Source Condition
Reconstructs $z_2$ from Fourier measurements:
- Exact data ($\sigma = 0$): 501 iterations. Despite energy reduction to $\sim 10^{-5}$, the iterate fails to converge to $z_2$ due to lack of source condition regularity.
- Noisy data ($\sigma = 1/50$): Discrepancy principle stops at iteration 46.

```bash
python scripts/run_figure4.py
```

#### Figure 5: $L^1$ Error Analysis Confirming Proposition 4.1
Plots $\Vert u^k - z\Vert_{L^1}$ versus iteration count $k$:
- $z_1$: Confirms $O(1/\sqrt{k})$ convergence in $L^1$ norm predicted by Proposition 4.1.
- $z_2$: Confirms stagnation and failure of asymptotic convergence when the source condition is violated.

```bash
python scripts/run_figure5.py
```

---

## Testing

A comprehensive unit test suite verifies mathematical properties and implementation invariants:
- **`test_functionals.py`**: Non-negativity and zero-annihilation of the Kullback–Leibler divergence, and entropy calculation.
- **`test_operators.py`**: Exact adjointness check $\langle Au, v \rangle = \langle u, A^{\ast} v \rangle$ for both Fredholm and discrete Fourier operators.
- **`test_algorithms.py`**: Non-negativity preservation ($u_k \ge 0$), total mass preservation ($\int u_k = 1$ when $m = 1$), and monotonic residual decrease $\frac{1}{2}\Vert Au_{k+1} - y\Vert^2 \le \frac{1}{2}\Vert Au_k - y\Vert^2$ (Proposition 3.3).

Run the tests using Python's standard `unittest` framework:

```bash
python -m unittest discover -s tests -v
```

---

## Citation

```bibtex
@article{Burger2020Entropic,
  author    = {Martin Burger and Elena Resmerita and Martin Benning},
  title     = {An entropic {L}andweber method for linear ill-posed problems},
  journal   = {Inverse Problems},
  volume    = {36},
  number    = {1},
  pages     = {015009},
  year      = {2020},
  publisher = {IOP Publishing},
  doi       = {10.1088/1361-6420/ab5c49}
}
```
