import json
from pathlib import Path

import pytest

PROCESSOR_DIR = Path(__file__).resolve().parents[1] / "processors"

EXPECTED = {
    "processor_A": {
        "num_qubits": 5,
        "edges": {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        },
    },
    "processor_B": {
        "num_qubits": 9,
        "edges": {
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5),
            (5, 6),
            (4, 7),
            (7, 8),
        },
    },
}


@pytest.mark.parametrize("name", ["processor_A", "processor_B"])
def test_processor_topology(name):
    path = PROCESSOR_DIR / f"{name}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = EXPECTED[name]
    n = config["num_qubits"]
    directed = [tuple(edge) for edge in config["coupling_map"]]

    assert config["name"] == name
    assert n == expected["num_qubits"]
    assert len(directed) == len(set(directed))
    assert all(len(edge) == 2 for edge in directed)
    assert all(0 <= a < n and 0 <= b < n for a, b in directed)
    assert all(a != b for a, b in directed)

    directed_set = set(directed)
    assert all((b, a) in directed_set for a, b in directed_set)

    physical_edges = {
        (min(a, b), max(a, b)) for a, b in directed_set
    }
    assert physical_edges == expected["edges"]

    adjacency = {q: set() for q in range(n)}
    for a, b in physical_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    visited = set()
    stack = [0]
    while stack:
        q = stack.pop()
        if q not in visited:
            visited.add(q)
            stack.extend(adjacency[q] - visited)

    assert len(visited) == n
