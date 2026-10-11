"""Structural tests for Member 2's (Linn's) CANDIDATE processor definitions.

These check internal consistency of processors/processor_{A,B}.json as loaded by
src/processors.py (kept unchanged). They do NOT establish that the topologies
match the official challenge diagrams; see docs/processor_handoff_notes.md.
"""

import ast
import json
from pathlib import Path

import networkx as nx
import pytest

import src.processors as processors
from src.processors import get_undirected_edges, load_processor

ROOT = Path(__file__).resolve().parents[1]

# Unique undirected edges as stated in Linn's handoff (candidate, unverified).
CANDIDATE = {
    "processor_A": (5, [(0, 1), (1, 2), (1, 3), (3, 4)]),
    "processor_B": (9, [(0, 1), (0, 8), (1, 2), (1, 6), (2, 3), (2, 5), (2, 6),
                        (3, 4), (4, 5), (5, 6), (6, 7), (6, 8)]),
}
NAMES = list(CANDIDATE)


@pytest.mark.parametrize("name", NAMES)
def test_loads_and_matches_handoff(name):
    cfg = load_processor(name)
    n, edges = CANDIDATE[name]
    assert cfg["name"] == name
    assert cfg["num_qubits"] == n
    assert get_undirected_edges(cfg) == edges


@pytest.mark.parametrize("name", NAMES)
def test_directed_entries_are_unique_symmetric_and_in_range(name):
    cfg = load_processor(name)
    n = cfg["num_qubits"]
    directed = [tuple(e) for e in cfg["coupling_map"]]
    assert len(directed) == len(set(directed))                 # no duplicate directed edges
    assert all((b, a) in set(directed) for a, b in directed)   # reverse-edge consistency
    assert all(0 <= q < n for e in directed for q in e)        # valid indices
    assert all(a != b for a, b in directed)                    # no self-loops
    assert len(directed) == 2 * len(get_undirected_edges(cfg))


@pytest.mark.parametrize("name", NAMES)
def test_every_qubit_used_and_connected(name):
    cfg = load_processor(name)
    g = nx.Graph(get_undirected_edges(cfg))
    g.add_nodes_from(range(cfg["num_qubits"]))
    assert min(dict(g.degree()).values()) >= 1   # e.g. Processor B must use all nine qubits
    assert nx.is_connected(g)


def test_qiskit_coupling_map_accepts_edges():
    """Structural only: Qiskit can represent the maps. Not a full integration test."""
    from qiskit.transpiler import CouplingMap

    for name in NAMES:
        cfg = load_processor(name)
        cm = CouplingMap(cfg["coupling_map"])
        assert cm.size() == cfg["num_qubits"]
        assert cm.is_symmetric and cm.is_connected()


# -- loader rejects malformed configs (temporary files; real JSON untouched) ----


def _write(tmp_path, monkeypatch, config, name="processor_A"):
    monkeypatch.setattr(processors, "PROCESSOR_DIR", tmp_path)
    (tmp_path / f"{name}.json").write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize(
    "config,match",
    [
        ({"name": "processor_A", "num_qubits": 3, "coupling_map": [[0, 1], [1, 0], [0, 1]]}, "Duplicate"),
        ({"name": "processor_A", "num_qubits": 3, "coupling_map": [[0, 1]]}, "reverse"),
        ({"name": "processor_A", "num_qubits": 3, "coupling_map": [[0, 3], [3, 0]]}, "Out-of-range"),
        ({"name": "processor_A", "num_qubits": 3, "coupling_map": [[1, 1]]}, "Self-loop"),
        ({"name": "processor_A", "num_qubits": 3, "coupling_map": [[0, 1.0], [1.0, 0]]}, "integers"),
        ({"name": "processor_A", "num_qubits": 3, "coupling_map": [[0, 1, 2]]}, "Invalid edge"),
        ({"name": "processor_A", "num_qubits": 0, "coupling_map": []}, "positive"),
        ({"name": "processor_A", "num_qubits": True, "coupling_map": []}, "positive"),
        ({"name": "processor_A", "num_qubits": 3, "coupling_map": {}}, "list"),
        ({"name": "processor_B", "num_qubits": 3, "coupling_map": []}, "Unexpected processor name"),
    ],
)
def test_loader_rejects_invalid(tmp_path, monkeypatch, config, match):
    _write(tmp_path, monkeypatch, config)
    with pytest.raises(ValueError, match=match):
        load_processor("processor_A")


def test_loader_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown processor"):
        load_processor("processor_C")


def test_undirected_edges_deduplicate_reverse_pairs():
    assert get_undirected_edges({"coupling_map": [[2, 0], [0, 2], [1, 2], [2, 1]]}) == [(0, 2), (1, 2)]


# -- module boundary: logical modules must not depend on processor topology -----


@pytest.mark.parametrize("module", ["src/problem.py", "src/qaoa_solver.py"])
def test_logical_modules_do_not_import_processors(module):
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("processor" in m or "routing" in m for m in imported), imported
