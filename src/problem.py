"""Weighted Max-Cut problem definition (Member 3).

This module is pure classical code: it has no Qiskit dependency and knows nothing
about processor topology. It provides

* :class:`WeightedGraph` -- a validated, immutable, undirected weighted graph,
* the weighted cut objective ``C(x) = sum_{(u,v) in E} w_uv * [x_u != x_v]``,
* bitstring <-> assignment conversion with an explicit, documented bit order,
* an exact brute-force reference for small graphs.

Conventions
-----------
* Vertices are the integers ``0 .. n-1``. This index is the canonical vertex order
  used everywhere (objective, QAOA qubit index, classical bit index).
* An *assignment* is a tuple ``x`` of length ``n`` with ``x[i] in {0, 1}`` giving
  the side of the partition that vertex ``i`` belongs to.
* Edges are stored canonically as ``(u, v, w)`` with ``u < v``, sorted by
  ``(u, v)``. Self-loops and duplicate edges (including reversed duplicates such
  as ``(0, 1)`` and ``(1, 0)``) are rejected so no edge is ever double-counted.
* Weights must be finite and non-negative (the standard weighted Max-Cut setting).

Bitstring order
---------------
Two string orders exist and conversion functions always require the caller to
name one explicitly:

* ``"qiskit"`` (little-endian): the *rightmost* character is vertex 0. This is
  how Qiskit displays measurement counts when qubit ``i`` is measured into
  classical bit ``i``. Example, n=4: ``"0011"`` means x0=1, x1=1, x2=0, x3=0.
* ``"vertex"`` (big-endian, human reading order): the *leftmost* character is
  vertex 0. Example, n=4: ``"0011"`` means x0=0, x1=0, x2=1, x3=1.

PROVENANCE: no official PS-O1 benchmark graph has been supplied to the team yet.
The graphs returned by the ``synthetic_*`` functions below are SYNTHETIC
development/test instances only and must not be reported as the official benchmark.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

BitOrder = Literal["qiskit", "vertex"]

#: Default safety cap for exhaustive search (2**20 ~ 1e6 assignments).
DEFAULT_MAX_EXACT_NODES = 20

#: Absolute tolerance used when comparing floating-point cut values.
CUT_ATOL = 1e-9


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightedGraph:
    """Validated undirected weighted graph on vertices ``0 .. num_nodes-1``.

    Build instances with :meth:`from_edges`, :meth:`from_networkx`,
    :meth:`from_dict` or :meth:`from_json` rather than the raw constructor, so
    edges are canonicalised and validated.

    Attributes:
        num_nodes: Number of vertices ``n``.
        edges: Canonical edges ``((u, v, w), ...)`` with ``u < v``, sorted.
        name: Free-text identifier, recorded in results.
        source: Provenance label, e.g. ``"synthetic"`` or ``"official"``.
        labels: Original vertex labels (``labels[i]`` is the label of vertex ``i``).
    """

    num_nodes: int
    edges: tuple[tuple[int, int, float], ...]
    name: str = "unnamed"
    source: str = "unspecified"
    labels: tuple[Any, ...] = field(default=())

    def __post_init__(self) -> None:
        if isinstance(self.num_nodes, bool) or not isinstance(self.num_nodes, (int, np.integer)):
            raise TypeError(f"num_nodes must be an int, got {type(self.num_nodes).__name__}")
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")
        object.__setattr__(self, "num_nodes", int(self.num_nodes))
        object.__setattr__(self, "edges", _canonical_edges(self.edges, self.num_nodes))
        if not self.labels:
            object.__setattr__(self, "labels", tuple(range(self.num_nodes)))
        elif len(self.labels) != self.num_nodes:
            raise ValueError(
                f"labels has length {len(self.labels)}, expected num_nodes={self.num_nodes}"
            )
        else:
            object.__setattr__(self, "labels", tuple(self.labels))

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_edges(
        cls,
        num_nodes: int,
        edges: Iterable[Sequence[Any]],
        *,
        name: str = "unnamed",
        source: str = "unspecified",
    ) -> "WeightedGraph":
        """Create a graph from ``(u, v)`` or ``(u, v, w)`` tuples (missing weight = 1.0)."""
        return cls(num_nodes=num_nodes, edges=tuple(edges), name=name, source=source)

    @classmethod
    def from_networkx(
        cls, graph: Any, *, weight: str = "weight", name: str | None = None, source: str = "unspecified"
    ) -> "WeightedGraph":
        """Create a graph from an undirected ``networkx.Graph``.

        Node labels are sorted (falling back to ``str`` ordering for mixed types)
        and mapped to ``0 .. n-1``; the original labels are kept in ``labels``.
        Edges without a ``weight`` attribute get weight 1.0.
        """
        if graph.is_directed():
            raise ValueError("Max-Cut requires an undirected graph")
        if graph.is_multigraph():
            raise ValueError("multigraphs are not supported; merge parallel edges first")
        nodes = list(graph.nodes)
        try:
            nodes = sorted(nodes)
        except TypeError:
            nodes = sorted(nodes, key=str)
        index = {node: i for i, node in enumerate(nodes)}
        edges = [(index[u], index[v], data.get(weight, 1.0)) for u, v, data in graph.edges(data=True)]
        return cls(
            num_nodes=len(nodes),
            edges=tuple(edges),
            name=name if name is not None else str(graph.graph.get("name", "networkx")),
            source=source,
            labels=tuple(nodes),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeightedGraph":
        """Inverse of :meth:`to_dict`.

        Expected keys: ``num_nodes`` and ``edges`` (list of ``[u, v]`` or
        ``[u, v, w]``); optional ``name``, ``source``, ``labels``.
        """
        if "num_nodes" not in data or "edges" not in data:
            raise ValueError("graph dict must contain 'num_nodes' and 'edges'")
        return cls(
            num_nodes=data["num_nodes"],
            edges=tuple(tuple(e) for e in data["edges"]),
            name=data.get("name", "unnamed"),
            source=data.get("source", "unspecified"),
            labels=tuple(data.get("labels", ())),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "WeightedGraph":
        """Load a graph saved in the :meth:`to_dict` format."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "name": self.name,
            "source": self.source,
            "num_nodes": self.num_nodes,
            "edges": [[u, v, w] for u, v, w in self.edges],
            "labels": [_jsonable(label) for label in self.labels],
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def to_networkx(self) -> Any:
        import networkx as nx

        g = nx.Graph(name=self.name)
        g.add_nodes_from(range(self.num_nodes))
        g.add_weighted_edges_from(self.edges)
        return g

    # -- convenience --------------------------------------------------------

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def total_weight(self) -> float:
        """Sum of all edge weights (an upper bound on any cut)."""
        return float(sum(w for _, _, w in self.edges))

    def weight(self, u: int, v: int) -> float:
        """Weight of edge ``{u, v}``; raises ``KeyError`` if absent."""
        a, b = (u, v) if u < v else (v, u)
        for x, y, w in self.edges:
            if (x, y) == (a, b):
                return w
        raise KeyError(f"no edge between {u} and {v}")


def _jsonable(value: Any) -> Any:
    return value if isinstance(value, (int, float, str, bool)) or value is None else str(value)


def _canonical_edges(edges: Iterable[Sequence[Any]], num_nodes: int) -> tuple[tuple[int, int, float], ...]:
    seen: dict[tuple[int, int], Sequence[Any]] = {}
    out: list[tuple[int, int, float]] = []
    for raw in edges:
        if isinstance(raw, (str, bytes)) or not hasattr(raw, "__len__") or len(raw) not in (2, 3):
            raise ValueError(f"edge must be (u, v) or (u, v, w), got {raw!r}")
        u, v = _vertex(raw[0], num_nodes), _vertex(raw[1], num_nodes)
        w = _weight(raw[2] if len(raw) == 3 else 1.0, (u, v))
        if u == v:
            raise ValueError(f"self-loop on vertex {u} is not allowed in Max-Cut")
        key = (min(u, v), max(u, v))
        if key in seen:
            raise ValueError(
                f"duplicate undirected edge {key}: {seen[key]!r} and {raw!r} "
                "(merge parallel edges before constructing the graph)"
            )
        seen[key] = raw
        out.append((key[0], key[1], w))
    out.sort(key=lambda e: (e[0], e[1]))
    return tuple(out)


def _vertex(value: Any, num_nodes: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"vertex must be an int, got {value!r}")
    if not 0 <= value < num_nodes:
        raise ValueError(f"vertex {value} out of range for num_nodes={num_nodes}")
    return int(value)


def _weight(value: Any, edge: tuple[int, int]) -> float:
    if isinstance(value, bool):
        raise TypeError(f"weight of edge {edge} must be a number, got bool")
    try:
        w = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"weight of edge {edge} must be a number, got {value!r}") from None
    if not math.isfinite(w):
        raise ValueError(f"weight of edge {edge} must be finite, got {w}")
    if w < 0:
        raise ValueError(f"weight of edge {edge} must be non-negative, got {w}")
    return w


# ---------------------------------------------------------------------------
# Assignments and bitstrings
# ---------------------------------------------------------------------------


def validate_assignment(graph: WeightedGraph, assignment: Sequence[Any]) -> tuple[int, ...]:
    """Return ``assignment`` as a tuple of 0/1 ints, or raise a descriptive error."""
    if isinstance(assignment, str):
        raise TypeError(
            "assignment must be a sequence of 0/1 ints; convert strings with "
            "bitstring_to_assignment(..., order=...) so the bit order is explicit"
        )
    values = tuple(assignment)
    if len(values) != graph.num_nodes:
        raise ValueError(f"assignment has length {len(values)}, expected {graph.num_nodes}")
    out = []
    for i, b in enumerate(values):
        if isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)) or b not in (0, 1):
            raise ValueError(f"assignment[{i}] must be 0 or 1, got {b!r}")
        out.append(int(b))
    return tuple(out)


def bitstring_to_assignment(bitstring: str, num_nodes: int, *, order: BitOrder) -> tuple[int, ...]:
    """Convert a bitstring into an assignment indexed by vertex.

    Args:
        bitstring: String of ``'0'``/``'1'`` of length ``num_nodes``.
        num_nodes: Expected number of vertices.
        order: ``"qiskit"`` (rightmost char = vertex 0) or ``"vertex"``
            (leftmost char = vertex 0). Required, to prevent silent reversals.
    """
    if not isinstance(bitstring, str):
        raise TypeError(f"bitstring must be a str, got {type(bitstring).__name__}")
    if " " in bitstring:
        raise ValueError(
            f"bitstring {bitstring!r} contains spaces (multiple classical registers?); "
            "measure all vertices into a single register"
        )
    if len(bitstring) != num_nodes:
        raise ValueError(f"bitstring {bitstring!r} has length {len(bitstring)}, expected {num_nodes}")
    if set(bitstring) - {"0", "1"}:
        raise ValueError(f"bitstring {bitstring!r} may only contain '0' and '1'")
    chars = bitstring[::-1] if order == "qiskit" else bitstring if order == "vertex" else None
    if chars is None:
        raise ValueError(f"order must be 'qiskit' or 'vertex', got {order!r}")
    return tuple(int(c) for c in chars)


def assignment_to_bitstring(assignment: Sequence[int], *, order: BitOrder) -> str:
    """Inverse of :func:`bitstring_to_assignment`."""
    s = "".join(str(int(b)) for b in assignment)
    if order == "qiskit":
        return s[::-1]
    if order == "vertex":
        return s
    raise ValueError(f"order must be 'qiskit' or 'vertex', got {order!r}")


def assignment_to_index(assignment: Sequence[int]) -> int:
    """Computational-basis index ``k = sum_i x_i 2**i`` (vertex i = bit i, Qiskit convention)."""
    return sum(int(b) << i for i, b in enumerate(assignment))


def index_to_assignment(index: int, num_nodes: int) -> tuple[int, ...]:
    """Inverse of :func:`assignment_to_index`."""
    if not 0 <= index < 2**num_nodes:
        raise ValueError(f"index {index} out of range for {num_nodes} vertices")
    return tuple((index >> i) & 1 for i in range(num_nodes))


def partition(assignment: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split vertices into ``(side_0, side_1)`` according to the assignment."""
    side0 = tuple(i for i, b in enumerate(assignment) if b == 0)
    side1 = tuple(i for i, b in enumerate(assignment) if b == 1)
    return side0, side1


def partition_to_assignment(
    num_nodes: int, side0: Iterable[int], side1: Iterable[int]
) -> tuple[int, ...]:
    """Inverse of :func:`partition`: vertices in ``side0`` get 0, in ``side1`` get 1.

    The two sides must be disjoint and together contain every vertex exactly once.
    """
    s0, s1 = list(side0), list(side1)
    for v in s0 + s1:
        _vertex(v, num_nodes)
    if len(set(s0)) != len(s0) or len(set(s1)) != len(s1):
        raise ValueError("a vertex is listed twice within one side of the partition")
    overlap = set(s0) & set(s1)
    if overlap:
        raise ValueError(f"vertices {sorted(overlap)} appear on both sides of the partition")
    missing = set(range(num_nodes)) - set(s0) - set(s1)
    if missing:
        raise ValueError(f"vertices {sorted(missing)} are not assigned to either side")
    x = [0] * num_nodes
    for v in s1:
        x[v] = 1
    return tuple(x)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def cut_value(graph: WeightedGraph, assignment: Sequence[int]) -> float:
    """Weighted cut ``C(x) = sum_{(u,v)} w_uv [x_u != x_v]`` (each undirected edge once)."""
    x = validate_assignment(graph, assignment)
    return float(sum(w for u, v, w in graph.edges if x[u] != x[v]))


def cut_edges(graph: WeightedGraph, assignment: Sequence[int]) -> tuple[tuple[int, int, float], ...]:
    """Edges crossing the cut."""
    x = validate_assignment(graph, assignment)
    return tuple(e for e in graph.edges if x[e[0]] != x[e[1]])


def cut_value_from_bitstring(graph: WeightedGraph, bitstring: str, *, order: BitOrder) -> float:
    return cut_value(graph, bitstring_to_assignment(bitstring, graph.num_nodes, order=order))


def cut_value_from_partition(graph: WeightedGraph, side0: Iterable[int], side1: Iterable[int]) -> float:
    """Weighted cut of the partition ``side0 | side1`` (must cover all vertices)."""
    return cut_value(graph, partition_to_assignment(graph.num_nodes, side0, side1))


def all_cut_values(graph: WeightedGraph, *, max_nodes: int = DEFAULT_MAX_EXACT_NODES) -> np.ndarray:
    """Cut value of every basis state, as an array indexed by ``k = sum_i x_i 2**i``.

    This is exactly the diagonal of the QAOA cost Hamiltonian in Qiskit's
    little-endian computational basis.
    """
    _guard_size(graph.num_nodes, max_nodes)
    n = graph.num_nodes
    k = np.arange(2**n, dtype=np.int64)
    values = np.zeros(2**n, dtype=float)
    for u, v, w in graph.edges:
        values += w * (((k >> u) ^ (k >> v)) & 1)
    return values


def _guard_size(num_nodes: int, max_nodes: int) -> None:
    if num_nodes > max_nodes:
        raise ValueError(
            f"exhaustive search over 2**{num_nodes} assignments refused "
            f"(max_nodes={max_nodes}); raise max_nodes explicitly if intended"
        )


# ---------------------------------------------------------------------------
# Exact classical reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactSolution:
    """Brute-force Max-Cut optimum.

    ``optimal_assignments`` lists *every* optimal assignment, including both
    members of each complementary pair (x and 1-x give the same cut).
    """

    optimal_value: float
    optimal_assignments: tuple[tuple[int, ...], ...]
    num_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimal_value": self.optimal_value,
            "optimal_assignments": [list(a) for a in self.optimal_assignments],
            "optimal_bitstrings_qiskit": [assignment_to_bitstring(a, order="qiskit") for a in self.optimal_assignments],
            "optimal_partitions": [[list(s) for s in partition(a)] for a in self.optimal_assignments],
            "num_evaluated": self.num_evaluated,
        }


def solve_exact(
    graph: WeightedGraph, *, max_nodes: int = DEFAULT_MAX_EXACT_NODES, atol: float = CUT_ATOL
) -> ExactSolution:
    """Exhaustively evaluate all ``2**n`` assignments and return all optima.

    Refuses graphs with more than ``max_nodes`` vertices to avoid accidental
    exponential blow-up.
    """
    values = all_cut_values(graph, max_nodes=max_nodes)
    best = float(values.max())
    idx = np.flatnonzero(np.isclose(values, best, rtol=0.0, atol=atol))
    return ExactSolution(
        optimal_value=best,
        optimal_assignments=tuple(index_to_assignment(int(k), graph.num_nodes) for k in idx),
        num_evaluated=int(values.size),
    )


def approximation_ratio(value: float, optimal_value: float, *, atol: float = CUT_ATOL) -> float | None:
    """``value / optimal_value``; ``None`` when the optimum is 0 (ratio undefined)."""
    if abs(optimal_value) <= atol:
        return None
    return float(value) / float(optimal_value)


# ---------------------------------------------------------------------------
# SYNTHETIC development graphs (NOT the official benchmark)
# ---------------------------------------------------------------------------


def synthetic_weighted_graph() -> WeightedGraph:
    """SYNTHETIC 4-vertex weighted development graph (not an official benchmark).

    Edges: (0,1,1), (1,2,2), (2,3,3), (0,3,4), (0,2,5). Hand-verified optimum:
    cut 11 for x = (0,0,1,1) and its complement (1,1,0,0), i.e. {0,1} | {2,3}.
    """
    return WeightedGraph.from_edges(
        4,
        [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0), (3, 0, 4.0), (0, 2, 5.0)],
        name="synthetic_weighted_4",
        source="synthetic",
    )


def synthetic_unweighted_graph() -> WeightedGraph:
    """SYNTHETIC 4-vertex unweighted graph (square plus diagonal 0-2), optimum 4.

    This is the team's earlier development example; it is NOT an official benchmark.
    """
    return WeightedGraph.from_edges(
        4,
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        name="synthetic_square_diagonal_4",
        source="synthetic",
    )
