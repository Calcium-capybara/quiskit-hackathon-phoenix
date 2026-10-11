# Member 3 — Weighted Max-Cut and QAOA (`src/problem.py`, `src/qaoa_solver.py`)

> **Data provenance.** No official PS-O1 benchmark graph or weights have been supplied yet.
> The only organizer document available, *Qiskit Fall Fest 2026 — All 20 Problems, Beginner Explanation*,
> describes O1 conceptually and says the Challenge Kit determines the actual instances.
> Every graph used in this module's tests and experiments is **SYNTHETIC** (`source="synthetic"`).
> To use the official instance, load it with `WeightedGraph.from_json(...)` / `from_edges(..., source="official")`. No solver code changes are needed.

## 1. Problem and objective

Undirected graph G = (V, E), V = {0, …, n−1}, non-negative finite weights w_uv.
An assignment x ∈ {0,1}^n puts vertex i on side x_i.

    C(x) = Σ_{(u,v) ∈ E} w_uv · [x_u ≠ x_v]

`WeightedGraph` validation enforces the following:
- Integer vertices in range.
- No self-loops.
- No duplicate edges, including reversed duplicates `(0,1)` / `(1,0)`, so no edge is double-counted.
- Finite, non-negative weights.

Edges are stored canonically as `(u, v, w)` with u < v, sorted.

**Evaluating a cut.** The objective can be computed from three input forms:
- an assignment tuple, with `cut_value`;
- a bitstring with an explicit order, with `cut_value_from_bitstring`;
- a partition `(side0, side1)`, with `cut_value_from_partition`. The two sides must be disjoint and must cover every vertex.

**Exact reference.** `solve_exact(graph)` enumerates all 2^n assignments (vectorised NumPy). It returns the optimum and *all* optimal assignments, including complementary pairs. It refuses n > 20 unless `max_nodes` is raised explicitly. The optimum is never hardcoded.

## 2. QAOA formulation

Cost Hamiltonian: **Ĉ = Σ_{(u,v)} w_uv (I − Z_u Z_v)/2**.
Z_u Z_v|x⟩ = ±|x⟩ (+ if x_u = x_v), so ⟨x|Ĉ|x⟩ = C(x), and ⟨ψ|Ĉ|ψ⟩ is the expected cut. Its diagonal equals `all_cut_values(graph)`, and this is tested.

State: |γ,β⟩ = Π_{l=1..p} U_B(β_l) U_C(γ_l) |+⟩^n

| Operator | Definition | Qiskit gates |
|---|---|---|
| U_C(γ) | exp(−iγĈ) | `rzz(−w_uv·γ)` on every edge (global phase e^{−iγw/2} dropped) |
| U_B(β) | exp(−iβ Σ_j X_j) | `rx(2β)` on every qubit |

The derivation uses Qiskit's definition RZZ(θ) = exp(−iθ/2 ZZ). Then exp(−iγw(I−ZZ)/2) = e^{−iγw/2}·exp(+iγw/2 ZZ) = phase · RZZ(−wγ).

Zero-weight edges emit no gate.

The tests check this against an independent `scipy.linalg.expm` construction and the closed-form single-edge result ⟨C⟩ = (w/2)(1 + sin 4β · sin wγ). Flipping the sign makes 4 tests fail (mutation-checked).

*Convention warning:* tutorials that use `rzz(2wγ')` are using a different parameterisation (γ = −2γ'). Convert any angles taken from them.

- **p**: the number of alternating cost/mixer layers.
- **γ_l**: the cost-layer angles.
- **β_l**: the mixer angles.

The circuit has 2p `Parameter`s, named `gamma[l]` and `beta[l]`.

## 3. Bit-order convention (single mapping used everywhere)

    vertex i → logical qubit q[i] → classical bit meas[i] → counts-key character key[n−1−i]

The **rightmost** character of a Qiskit counts key is vertex 0. Decode with
`bitstring_to_assignment(key, n, order="qiskit")`. The `order` argument is mandatory so it can never be omitted by accident.
Example (n=4): key `"1000"` → x = (0,0,0,1) → cut 7 on the synthetic weighted graph. Reading the key the wrong way round gives 10, and the tests catch that (mutation-checked: 5 failures).

## 4. Optimisation and evaluation (experiment settings, not official requirements)

`QAOAConfig` defaults:

| Setting | Default |
|---|---|
| `p` | 1 |
| `optimizer` | `"COBYLA"` (any gradient-free `scipy.optimize.minimize` method) |
| `maxiter` | 200 |
| `n_restarts` | 3 |
| `objective` | `"statevector"` (exact expectation) or `"shots"` |
| `shots` | 2048 |
| `seed` | 1234 |

Random initial angles are drawn with γ ∈ [0, π) and β ∈ [0, π/2).

`run_qaoa(graph, config)` returns a `QAOAResult` (`.to_dict()` is JSON-safe) with these fields:
- **Angles:** `gammas`, `betas`.
- **Expected cut:** `expected_cut` (exact ⟨Ĉ⟩) and `expected_approximation_ratio`.
- **Optimal-state probability:** `optimal_solution_probability`.
- **Sample statistics:** `sampled_mean_cut`, `counts`, `cut_distribution`.
- **Best sample:** `best_observed_cut`, `best_observed_bitstring`, `best_observed_assignment`, `best_observed_partition`, `best_observed_ratio`.
- **Optimizer record:** `optimizer` (settings plus a per-restart trace: start/end point, nfev, success, message, history).
- **Circuit and timing:** `circuit` (logical metadata), `runtime_seconds`.

The best observed sample and the expected cut are reported separately. Ratios are `None` when the optimum is 0.

Sampling uses Qiskit's noiseless `StatevectorSampler`. **These are logical results and say nothing about either processor.**

## 5. Interface for teammates

```python
from src.problem import WeightedGraph, solve_exact, bitstring_to_assignment, cut_value
from src.qaoa_solver import build_qaoa_circuit, logical_circuit_for_routing, decode_counts, run_qaoa, QAOAConfig

graph = WeightedGraph.from_json("data/raw/<official_graph>.json")   # once available
qaoa  = build_qaoa_circuit(graph, p=1)          # QAOACircuit: .circuit, .gammas, .betas, .metadata()
bound = qaoa.bind(gammas, betas)                 # or logical_circuit_for_routing(graph, p, gammas, betas)
```

**Member 4 (routing):**
- Build the logical circuit **once** and transpile that same object for Processor A and for Processor B, with identical transpiler settings and seeds.
- The circuit contains only `h`, `rzz`, `rx` and `measure`. It has no topology assumptions and is never transpiled inside `qaoa_solver.py`.
- Keep the measurements in the circuit when transpiling. The explicit `measure(q[i], meas[i])` wiring survives layout and routing, so counts from the compiled circuit still decode in logical vertex order with `decode_counts(graph, counts)`.
- This is tested with a generic *test-only* line coupling map that forces SWAPs. That map is **not** Processor A or B.
- Logical baselines for overhead ratios are in `qaoa.metadata()["logical_depth"]` and `["logical_two_qubit_gates"]`.

**Member 5 (metrics):** use `QAOAResult.to_dict()` for the logical-quality columns, and Member 4's output for the compiled metrics (depth, 2-qubit count, SWAPs, physical qubits).
- Lower SWAP count does not imply better Max-Cut quality. They are separate metrics.
- Noiseless logical quality is the same on A and B by construction. A quality difference between them can only come from a noisy simulation, if the team implements one.

**Member 2 (processors):** this module needs nothing from you. The processor JSONs are consumed only by Member 4.
- Linn's *candidate* definitions are now in the repository: A has 5 qubits and 4 edges, B has 9 qubits and 12 edges. They are unmodified and structurally tested.
- Open issues are listed in [processor_handoff_notes.md](processor_handoff_notes.md), including the 7-vs-9-qubit discrepancy for B and the lack of a check against the official diagram.
- Width constraint only: the logical circuit uses one qubit per vertex, so a graph must have ≤ 5 vertices for A and ≤ 9 for B. No A/B comparison has been run.

## 6. Reproduction

```
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m scripts.run_qaoa_synthetic
```

The experiment script writes `results/tables/synthetic_logical_qaoa.json` and `..._summary.csv`.

## 7. Limitations

- **Synthetic data only.** Graphs are synthetic until the official benchmark is provided.
- **Exact methods are small-n.** Exact search and statevector evaluation are exponential. They are practical up to roughly 20 vertices (the guard), and in practice the target is ≤ 9 qubits, matching Processor B.
- **Local optimizer.** COBYLA can stop in a local optimum or at `maxiter`. In the synthetic run at p ≥ 2, nfev reached 200, so those runs did not converge. Raise `maxiter` or `n_restarts` if needed.
- **Heuristic γ range.** The default initial-γ range is a heuristic; the γ period depends on the weights.
- **Noiseless only.** No noise model is used. These results make no claim of quantum advantage.
