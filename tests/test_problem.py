"""Correctness tests for src/problem.py (Member 3).

All graphs here are SYNTHETIC test instances; expected values are hand-computed.
"""

import math

import networkx as nx
import numpy as np
import pytest

from src.problem import (
    WeightedGraph,
    all_cut_values,
    approximation_ratio,
    assignment_to_bitstring,
    assignment_to_index,
    bitstring_to_assignment,
    cut_edges,
    cut_value,
    cut_value_from_bitstring,
    index_to_assignment,
    partition,
    solve_exact,
    synthetic_unweighted_graph,
    synthetic_weighted_graph,
)

# Hand-computed cut table for synthetic_weighted_graph():
# edges (0,1,1) (1,2,2) (2,3,3) (0,3,4) (0,2,5); assignment x = (x0, x1, x2, x3)
WEIGHTED_TABLE = {
    (0, 0, 0, 0): 0,
    (0, 0, 0, 1): 7,   # (2,3)=3 + (0,3)=4
    (0, 0, 1, 0): 10,  # (1,2)=2 + (2,3)=3 + (0,2)=5
    (0, 0, 1, 1): 11,  # (1,2)=2 + (0,3)=4 + (0,2)=5
    (0, 1, 0, 0): 3,   # (0,1)=1 + (1,2)=2
    (0, 1, 0, 1): 10,  # 1 + 2 + 3 + 4
    (0, 1, 1, 0): 9,   # 1 + 3 + 5
    (0, 1, 1, 1): 10,  # 1 + 4 + 5
}


# -- construction & validation ----------------------------------------------


def test_edges_are_canonicalised_and_sorted():
    g = WeightedGraph.from_edges(3, [(2, 1, 0.5), (1, 0, 2)])
    assert g.edges == ((0, 1, 2.0), (1, 2, 0.5))
    assert g.num_edges == 2 and g.total_weight == 2.5


def test_missing_weight_defaults_to_one():
    assert WeightedGraph.from_edges(2, [(0, 1)]).edges == ((0, 1, 1.0),)


@pytest.mark.parametrize("edges", [[(0, 1), (1, 0)], [(0, 1, 1.0), (0, 1, 2.0)]])
def test_duplicate_and_reversed_edges_rejected(edges):
    with pytest.raises(ValueError, match="duplicate"):
        WeightedGraph.from_edges(2, edges)


def test_self_loop_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        WeightedGraph.from_edges(2, [(1, 1, 1.0)])


@pytest.mark.parametrize("bad", [(0, 3), (-1, 0)])
def test_out_of_range_vertex_rejected(bad):
    with pytest.raises(ValueError, match="out of range"):
        WeightedGraph.from_edges(3, [bad])


@pytest.mark.parametrize("bad", [(0.0, 1), ("0", 1), (True, 1)])
def test_non_int_vertex_rejected(bad):
    with pytest.raises(TypeError):
        WeightedGraph.from_edges(2, [bad])


@pytest.mark.parametrize("w", [-1.0, math.inf, math.nan])
def test_invalid_weights_rejected(w):
    with pytest.raises(ValueError):
        WeightedGraph.from_edges(2, [(0, 1, w)])


@pytest.mark.parametrize("w", ["heavy", None, True])
def test_non_numeric_weights_rejected(w):
    with pytest.raises(TypeError):
        WeightedGraph.from_edges(2, [(0, 1, w)])


@pytest.mark.parametrize("n", [0, -2])
def test_invalid_num_nodes(n):
    with pytest.raises(ValueError):
        WeightedGraph.from_edges(n, [])


def test_malformed_edge_rejected():
    with pytest.raises(ValueError, match="edge must be"):
        WeightedGraph.from_edges(3, [(0, 1, 2.0, 9)])


def test_dict_json_roundtrip(tmp_path):
    g = synthetic_weighted_graph()
    path = tmp_path / "g.json"
    g.to_json(path)
    g2 = WeightedGraph.from_json(path)
    assert g2 == g and g2.source == "synthetic"


def test_networkx_roundtrip_with_relabelling():
    nxg = nx.Graph()
    nxg.add_edge("b", "a", weight=2.0)
    nxg.add_edge("c", "b")  # no weight -> 1.0
    g = WeightedGraph.from_networkx(nxg)
    assert g.labels == ("a", "b", "c")
    assert g.edges == ((0, 1, 2.0), (1, 2, 1.0))
    assert WeightedGraph.from_networkx(g.to_networkx()).edges == g.edges


def test_networkx_directed_rejected():
    with pytest.raises(ValueError):
        WeightedGraph.from_networkx(nx.DiGraph([(0, 1)]))


def test_isolated_vertices_allowed():
    g = WeightedGraph.from_edges(3, [(0, 1, 1.0)])
    assert cut_value(g, (0, 1, 1)) == 1.0


# -- cut value ----------------------------------------------------------------


@pytest.mark.parametrize("x,expected", [((0, 0), 0.0), ((1, 1), 0.0), ((0, 1), 2.5), ((1, 0), 2.5)])
def test_single_edge_cut(x, expected):
    g = WeightedGraph.from_edges(2, [(0, 1, 2.5)])
    assert cut_value(g, x) == expected


@pytest.mark.parametrize("x,expected", WEIGHTED_TABLE.items())
def test_weighted_cut_table(x, expected):
    g = synthetic_weighted_graph()
    assert cut_value(g, x) == expected
    # complement symmetry
    assert cut_value(g, tuple(1 - b for b in x)) == expected


def test_weights_are_actually_used():
    """Same topology, different weights -> different cut value."""
    g1 = WeightedGraph.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
    g2 = WeightedGraph.from_edges(3, [(0, 1, 1.0), (1, 2, 7.0)])
    x = (0, 0, 1)  # cuts only edge (1,2)
    assert cut_value(g1, x) == 1.0
    assert cut_value(g2, x) == 7.0


def test_cut_edges_lists_crossing_edges():
    g = synthetic_weighted_graph()
    assert cut_edges(g, (0, 0, 1, 1)) == ((0, 2, 5.0), (0, 3, 4.0), (1, 2, 2.0))


@pytest.mark.parametrize(
    "bad,exc",
    [((0, 1, 0), ValueError), ((0, 1, 2, 0), ValueError), ((0, 1, 0.5, 0), ValueError),
     ((True, 0, 0, 0), ValueError), ("0101", TypeError)],
)
def test_invalid_assignments_rejected(bad, exc):
    with pytest.raises(exc):
        cut_value(synthetic_weighted_graph(), bad)


def test_numpy_assignment_accepted():
    assert cut_value(synthetic_weighted_graph(), np.array([0, 0, 1, 1])) == 11.0


# -- bitstrings -----------------------------------------------------------------


def test_bitstring_orders_are_mirror_images():
    assert bitstring_to_assignment("0001", 4, order="qiskit") == (1, 0, 0, 0)
    assert bitstring_to_assignment("0001", 4, order="vertex") == (0, 0, 0, 1)


def test_bitstring_roundtrip():
    x = (1, 0, 1, 1, 0)
    for order in ("qiskit", "vertex"):
        assert bitstring_to_assignment(assignment_to_bitstring(x, order=order), 5, order=order) == x


def test_reversed_bitstring_changes_weighted_cut():
    """Guard against bit-order bugs: the reversal is NOT the complement here."""
    g = synthetic_weighted_graph()
    # x = (0,0,0,1): cut 7. Qiskit string puts vertex 0 rightmost -> "1000".
    assert cut_value_from_bitstring(g, "1000", order="qiskit") == 7.0
    # Reading the same string in vertex order gives x=(1,0,0,0): cut 1+4+5=10.
    assert cut_value_from_bitstring(g, "1000", order="vertex") == 10.0


@pytest.mark.parametrize("bad", ["01", "01201", "0 101", "abcd"])
def test_invalid_bitstrings_rejected(bad):
    with pytest.raises(ValueError):
        bitstring_to_assignment(bad, 4, order="qiskit")


def test_bitstring_order_required_and_validated():
    with pytest.raises(TypeError):
        bitstring_to_assignment("0101", 4)  # order is keyword-only and required
    with pytest.raises(ValueError):
        bitstring_to_assignment("0101", 4, order="msb")


def test_index_assignment_roundtrip():
    assert assignment_to_index((1, 0, 1)) == 5
    assert index_to_assignment(5, 3) == (1, 0, 1)
    assert index_to_assignment(1, 4) == (1, 0, 0, 0)  # vertex 0 is the least-significant bit
    with pytest.raises(ValueError):
        index_to_assignment(8, 3)


def test_partition():
    assert partition((0, 1, 1, 0)) == ((0, 3), (1, 2))


# -- exact reference ---------------------------------------------------------------


def test_all_cut_values_matches_cut_value():
    g = synthetic_weighted_graph()
    values = all_cut_values(g)
    for k in range(16):
        assert values[k] == cut_value(g, index_to_assignment(k, 4))


def test_exact_weighted_optimum():
    sol = solve_exact(synthetic_weighted_graph())
    assert sol.optimal_value == 11.0
    assert set(sol.optimal_assignments) == {(0, 0, 1, 1), (1, 1, 0, 0)}
    assert sol.num_evaluated == 16
    assert set(sol.to_dict()["optimal_bitstrings_qiskit"]) == {"1100", "0011"}


def test_exact_unweighted_optimum():
    sol = solve_exact(synthetic_unweighted_graph())
    # Square 0-1-2-3 plus diagonal 0-2: best is {0,2}|{1,3}, cutting the 4 square edges.
    assert sol.optimal_value == 4.0
    assert set(sol.optimal_assignments) == {(0, 1, 0, 1), (1, 0, 1, 0)}


def test_exact_triangle_has_six_optima():
    g = WeightedGraph.from_edges(3, [(0, 1), (1, 2), (0, 2)])
    sol = solve_exact(g)
    assert sol.optimal_value == 2.0 and len(sol.optimal_assignments) == 6


def test_exact_matches_networkx_bruteforce_on_random_graphs():
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = int(rng.integers(3, 8))
        edges = [(u, v, float(rng.integers(1, 10))) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.6]
        g = WeightedGraph.from_edges(n, edges)
        nxg = g.to_networkx()
        best = 0.0
        for k in range(2**n):
            side = {i for i in range(n) if (k >> i) & 1}
            best = max(best, nx.cut_size(nxg, side, weight="weight"))
        assert solve_exact(g).optimal_value == best


def test_exact_guard_against_large_graphs():
    g = WeightedGraph.from_edges(25, [(0, 1)])
    with pytest.raises(ValueError, match="max_nodes"):
        solve_exact(g)


def test_edgeless_graph_has_zero_optimum_and_undefined_ratio():
    sol = solve_exact(WeightedGraph.from_edges(3, []))
    assert sol.optimal_value == 0.0
    assert approximation_ratio(0.0, sol.optimal_value) is None


def test_approximation_ratio():
    assert approximation_ratio(8.25, 11.0) == pytest.approx(0.75)
