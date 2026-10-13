# qiskit-hackathon-phoenix

Qiskit Fall Fest 2026, problem statement **PS-O1: Weighted Max-Cut — QAOA circuit construction and SWAP bottleneck analysis**.

## Member 3 — Weighted Max-Cut and logical QAOA

Code: `src/problem.py` (graph, objective, exact solver) and `src/qaoa_solver.py` (logical QAOA circuit, optimization, results).
Full conventions and result schema: [docs/member3_problem_qaoa.md](docs/member3_problem_qaoa.md).
These modules contain no processor topology, routing or noise. Routing onto Processor A/B is Member 4's responsibility.

### Objective

    C(x) = Σ_{(u,v) ∈ E} w_uv · [x_u ≠ x_v],   x_u ∈ {0, 1}

An edge contributes its weight when its endpoints are on opposite sides. Each undirected edge is counted once. A cut and its complement have the same value.

### Candidate graph G6 (`data/weighted_maxcut_g6.json`)

**Team-defined, not an organizer-provided benchmark.** It has 6 vertices (0–5) and 9 edges with integer weights 1–5, for a total weight of 29:

| Edge | 0–1 | 0–2 | 0–5 | 1–2 | 1–3 | 2–3 | 2–4 | 3–4 | 4–5 |
|---|---|---|---|---|---|---|---|---|---|
| Weight | 3 | 2 | 4 | 5 | 2 | 3 | 1 | 4 | 5 |

**Exact optimum: 24**, at `010101` and `101010` (vertex order), i.e. the partition {0,2,4} | {1,3,5}. `solve_exact()` verifies this by enumerating all 2^6 = 64 assignments, and the test suite checks it on every run.

G6 needs **6 logical qubits**. It does not fit the current 5-qubit candidate Processor A; this is a limitation to report, and the graph should not be altered to fit.

`load_graph()` validates the file strictly:
- integer fields reject `bool` and floats;
- weights must lie within `weight_range`;
- endpoints must be in range;
- self-loops and duplicate or reversed-duplicate edges are rejected;
- the edge-record keys must be exactly `u`, `v`, `weight`.

The default path is resolved relative to the source file, not the current directory.

### Conventions

- **QAOA:** Ĉ = Σ w_uv (I − Z_u Z_v)/2, U_C(γ) = e^{−iγĈ} (implemented as `rzz(−w_uv·γ)` per edge, up to a global phase), and U_B(β) = e^{−iβΣX} (`rx(2β)` per qubit). The initial state is H on every qubit, and each layer applies cost then mixer, p times.
- **Qubits:** vertex i → logical qubit i → classical bit i.
- **Bit order:** *vertex order* puts vertex 0 leftmost; *Qiskit counts* put vertex 0 **rightmost**. Conversion functions require `order="vertex"` or `order="qiskit"` explicitly.
  - Example: vertices {0,1} on side 1 is `110000` in vertex order but `000011` in Qiskit order. It cuts 13; misreading the key gives 9.
  - G6's optimal strings are reversal-symmetric, so tests use non-symmetric keys like this one.
- **Assignments:** a list, tuple or 1-D integer array of `0`/`1`. `True`/`False`, floats, strings, sets and dicts are rejected (`TypeError`); wrong length or other integers raise `ValueError`.
- **Initialization ranges are not bounds:**
  - γ starts in [0, π) and β in [0, π/2);
  - COBYLA runs **without bounds**, so the optimized angles may lie outside these ranges.

### Result kinds (`run_qaoa`) — report them separately

| Kind | Fields |
|---|---|
| Exact classical reference | `optimal_value`, `exact` |
| Statevector (exact for the given angles) | `expected_cut`, `expected_approximation_ratio`, `optimal_solution_probability`, `best_cut_with_statevector_probability` (max cut over states with probability **strictly greater than** `statevector_probability_threshold`, default 1e-12) |
| Finite-shot samples (noiseless, seeded; 4096 shots by default, the Challenge Kit setting) | `counts`, `sampled_mean_cut`, `sampled_approximation_ratio`, `best_observed_cut`, `best_observed_is_exact_optimum`, `sampled_optimal_frequency` (fraction of shots that hit the optimum), `most_frequent_bitstring` |
| Optimizer status | `optimizer_success`: termination only, **not** proof that the optimum was found |

`best_cut_with_statevector_probability` almost always equals the optimum, even for unoptimized angles; for example, the uniform state already supports 24. Use `optimal_solution_probability` (exact, from the statevector) and `sampled_optimal_frequency` (finite-shot success frequency, which estimates it) to judge solution quality.

### Reproducibility

- **Exact search:** deterministic.
- **Statevector values:** deterministic for fixed angles.
- **Optimizer starting points:** drawn from `numpy.random.default_rng(seed)`.
- **Shots:** `StatevectorSampler(seed=seed)`.

The same seed and config give identical results in the tested environment (Python 3.14.7, Qiskit 2.5.2, SciPy 1.18.1, NumPy 2.5.4, Windows). Identical results across other versions or platforms are not guaranteed.

### Setup and commands (Windows PowerShell, from the project root)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m pytest -q                                   # full test suite
python -m scripts.run_qaoa_g6 --p 1 --seed 7          # sample logical QAOA run on G6 (prints JSON)
python -m scripts.run_qaoa_g6 --p 2 --seed 7 --output results\tables\g6_p2.json
```

To get the logical circuit for routing (Member 4):

```python
from src.problem import load_graph
from src.qaoa_solver import logical_circuit_for_routing, decode_counts
g = load_graph()
circ = logical_circuit_for_routing(g, p, gammas, betas)  # 6 qubits; ops: h, rzz, rx, measure
# after transpiling + running: decode_counts(g, counts) maps keys back to logical vertices
```

### Limitations

- G6 is team-defined, and no organizer benchmark instance has been confirmed.
- Results are noiseless logical simulations only, with no hardware or noise model.
- COBYLA is a local optimizer that may stop at a local optimum or at `maxiter`.
- Exact search and statevector evaluation are exponential; guarded to n ≤ 20.
- No G6 benchmark results are committed. Sample runs print to the console unless `--output` is given.
