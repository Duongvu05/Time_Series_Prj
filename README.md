# RL Class Project – Silver Lectures 2 & 3

> **Course Assignment** | Reinforcement Learning  
> Based on David Silver's Lecture 2 (MDPs) and Lecture 3 (Dynamic Programming)

## Dependencies / Packages

### Project Management
[![Managed by uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v1.json)](https://github.com/astral-sh/uv)

### Core Stack
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-≥2.0-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-≥1.14-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-≥3.9-11557C?style=for-the-badge&logo=python&logoColor=white)
![Loguru](https://img.shields.io/badge/Loguru-≥0.7-FF6B35?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-≥7.3-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### Development Tools
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Pyright](https://img.shields.io/badge/Pyright-≥1.1-1E90FF?style=for-the-badge&logo=python&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-≥8.3-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

---

## Project Structure

```
Time_Series_Prj/
├── pyproject.toml          ← uv project config + ruff/pyright/pytest settings
├── README.md
├── slides/
│   └── slides.tex          ← LaTeX Beamer presentation (compile → slides.pdf)
├── src/
│   ├── q1_student_mdp/
│   │   ├── mdp.py           ← Student MDP: $V^{\pi}$, $Q^{\pi}$, $V^{\ast}$, $Q^{\ast}$, $\pi^{\ast}$
│   │   └── explanation.md
│   ├── q2_grid_world/
│   │   ├── grid_world.py    ← Iterative Policy Evaluation
│   │   └── explanation.md
│   ├── q3_jacks_car/
│   │   ├── jacks_car.py     ← Policy Iteration (precomputed Poisson transitions)
│   │   └── explanation.md
│   └── q4_contraction/
│       ├── demo.py          ← Numerical contraction verification
│       └── proof.md         ← Full formal proof (Banach theorem)
├── notebooks/
│   ├── Q1_Student_MDP.ipynb
│   ├── Q2_Grid_World.ipynb
│   ├── Q3_Jacks_Car_Rental.ipynb
│   └── Q4_Contraction_Mapping.ipynb
└── tests/
    ├── test_q1.py  (14 tests)
    ├── test_q2.py  (10 tests)
    ├── test_q3.py  (11 tests)
    └── test_q4.py  (8 tests)
```

---

## Setup and Running

### 1. Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install project dependencies
```bash
cd Time_Series_Prj
uv sync
uv add --dev pytest ruff pyright
```

### 3. Quality checks
```bash
uv run ruff check .          # linting (0 errors expected)
uv run ruff format .         # auto-format
uv run pyright src/ tests/   # type checking (0 errors expected)
uv run pytest -v             # run 43 tests
```

### 4. Run individual question demos
```bash
# Q1 – Student MDP
uv run python -m src.q1_student_mdp.mdp

# Q2 – Grid World Iterative Policy Evaluation
uv run python -m src.q2_grid_world.grid_world

# Q3 – Jack's Car Rental (~1 min, saves plot to notebooks/)
uv run python -m src.q3_jacks_car.jacks_car

# Q4 – Contraction Mapping numerical verification
uv run python -m src.q4_contraction.demo
```

### 5. Jupyter Notebooks
```bash
uv run jupyter notebook notebooks/
# then open Q1_Student_MDP.ipynb, Q2_Grid_World.ipynb, etc.
```

### 6. Compile LaTeX Slides
```bash
cd slides
pdflatex slides.tex  # run twice for table of contents
pdflatex slides.tex
# Output: slides/slides.pdf
```

> **Note:** Requires a LaTeX distribution (TeX Live / MiKTeX).  
> On Ubuntu/Debian: `sudo apt install texlive-full`

---

## Answers to Questions

### Q1 – Value Functions with Policy $\pi$ (Câu 1 - ML2.pdf)

**MDP:** 5 states: C1, C2, C3, FB, Sleep (terminal).

**Bellman Expectation Equation:**
$$V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^{\pi}(s')]$$
$$Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^{\pi}(s')]$$
$$V^{\pi}(s) = \sum_a \pi(a|s) Q^{\pi}(s,a) \quad \text{(consistency)}$$

**Computed results** ($\gamma=1$, uniform random policy):

| State | $V^{\pi}$ | $V^{\ast}$ | $\pi^{\ast}$ |
|-------|---------|-------|---------|
| Facebook | $-2.3$ | $6.0$ | $\pi^{\ast}(\text{FB}) = \text{Quit}$ |
| C1 | $-1.3$ | $6.0$ | $\pi^{\ast}(\text{C1}) = \text{Study}$ |
| C2 | $+2.7$ | $8.0$ | $\pi^{\ast}(\text{C2}) = \text{Study}$ |
| C3 | $+7.4$ | $10.0$ | $\pi^{\ast}(\text{C3}) = \text{Study}$ |
| Sleep | $0.00$ | $0.00$ | $\pi^{\ast}(\text{Sleep}) = \text{Stay}$ |

**Optimal equations:**  
$$V^{\ast}(s) = \max_{a} Q^{\ast}(s,a)$$
found by Value Iteration (apply $T^{\ast}$ until $|V_{k+1} - V_{k}|_{\infty} < \theta$).

---

### Q2 – Iterative Policy Evaluation, k=2 and k=3 (Lecture 3, pp. 10–11)

**Setup:** 4×4 grid, random policy ($1/4$ each direction), $\gamma=1$, $R=-1$, terminals: $\{0, 15\}$.

**Algorithm Options:**
1. **Synchronous:** Uses two arrays, updates after a full sweep.
2. **In-place:** Uses one array, updates values immediately (converges faster).

**k=1:** Every non-terminal cell gets $V_{1}(s) = -1.0$ (since $V_{0}=0$ everywhere).

**k=2 (hand computation):**
- State 1 (top row, col 1): $V_{2}(1) = \frac{1}{4}(-2 - 2 - 1 - 2) = -1.75$
- State 5 (interior): $V_{2}(5) = \frac{1}{4}(-8) = -2.00$

**Code Examples:**
```python
from src.q2_grid_world.grid_world import (
    iterative_policy_eval, 
    run_to_convergence, 
    run_to_convergence_inplace
)

# Standard convergence
V_inf, steps = run_to_convergence()
# Faster convergence via in-place updates
V_inf_inp, steps_inp = run_to_convergence_inplace()
```

**k → ∞:** Values decrease monotonically. Grid world converges in ~250 sweeps. Greedy policy from $V^\pi$ always points toward nearest terminal.

---

### Q3 – Jack's Car Rental (Lecture 3, pp. 14–15)

**MDP:** State $s=(n_{1},n_{2})$, action $a \in [-5,5]$ (cars moved overnight).

**Key optimisation:** Precompute Poisson sums into matrices $\text{ExpRew}[n]$ and $\text{Trans}[n,n']$, so the Bellman update reduces to:

$$\mathbb{E}[V(n_{1}', n_{2}')] = \text{Trans}_{1}[n_{1}, :] \cdot V \cdot \text{Trans}_{2}[n_{2}, :]^{\top}$$
*(Implemented as efficient matrix-vector products)*

**Policy Iteration:** Converges in $\approx 4$ steps. Final policy moves cars from loc2→loc1 when loc1 is depleted ($\lambda_{\text{rent}_{2}}=4 > \lambda_{\text{rent}_{1}}=3$, so loc2 fills up faster).

Run `uv run python -m src.q3_jacks_car.jacks_car` to reproduce the Silver slide policy heatmap.

---

### Q4 – Contraction Mapping Theorem

See full proof: [`src/q4_contraction/proof.md`](src/q4_contraction/proof.md)

**Theorem (Banach Fixed-Point):** If $T$ is a $\gamma$-contraction on a complete metric space, it has a unique fixed point, and iterates converge at rate $O(\gamma^k)$.

**$T^{\pi}$ is a $\gamma$-contraction:**
$$|T^{\pi} V_{1} - T^{\pi} V_{2}|_{\infty} \leq \gamma |V_{1} - V_{2}|_{\infty}$$
*Proof:* Pull out $\gamma$, use triangle inequality, stochastic matrix rows sum to 1.

**$T^{\ast}$ is a $\gamma$-contraction:** Use $|\max_{a} f(a) - \max_{a} g(a)| \leq \max_{a} |f(a) - g(a)|$.

**Policy iteration converges:** monotone improvement + finitely many policies $\Longrightarrow$ termination at $V^{\ast}$.

**Numerical verification:**
```python
uv run python -m src.q4_contraction.demo   # checks $\gamma \in \{0.5, 0.9, 0.99\}$
```
