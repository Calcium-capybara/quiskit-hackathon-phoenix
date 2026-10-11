import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSOR_DIR = PROJECT_ROOT / "processors"


def load_processor(name):
    """Load and validate a processor JSON configuration."""
    if name not in {"processor_A", "processor_B"}:
        raise ValueError(f"Unknown processor: {name}")

    path = PROCESSOR_DIR / f"{name}.json"

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("name") != name:
        raise ValueError(f"Unexpected processor name in {path}")

    n = config.get("num_qubits")
    edges = config.get("coupling_map")

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("num_qubits must be a positive integer")

    if not isinstance(edges, list):
        raise ValueError("coupling_map must be a list")

    seen = set()

    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"Invalid edge: {edge!r}")

        a, b = edge

        if any(not isinstance(q, int) or isinstance(q, bool)
               for q in edge):
            raise ValueError(f"Qubit indices must be integers: {edge!r}")

        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Out-of-range edge: {edge!r}")

        if a == b:
            raise ValueError(f"Self-loop: {edge!r}")

        if (a, b) in seen:
            raise ValueError(f"Duplicate directed edge: {edge!r}")

        seen.add((a, b))

    for a, b in seen:
        if (b, a) not in seen:
            raise ValueError(f"Missing reverse edge for {(a, b)}")

    return config


def get_undirected_edges(config):
    """Return each physical connection once."""
    return sorted({
        (min(a, b), max(a, b))
        for a, b in config["coupling_map"]
    })
