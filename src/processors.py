
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSOR_DIR = PROJECT_ROOT / "processors"


def load_processor(name):
    """Load and validate a processor's JSON configuration."""
    if name not in {"processor_A", "processor_B"}:
        raise ValueError(f"Unknown processor: {name}")

    path = PROCESSOR_DIR / f"{name}.json"

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {"name", "num_qubits", "coupling_map"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")

    if config["name"] != name:
        raise ValueError(
            f"Expected name {name!r}, got {config['name']!r}"
        )

    num_qubits = config["num_qubits"]
    if not isinstance(num_qubits, int) or isinstance(num_qubits, bool):
        raise ValueError("num_qubits must be an integer")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    edges = config["coupling_map"]
    if not isinstance(edges, list):
        raise ValueError("coupling_map must be a list")

    seen = set()

    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(
                not isinstance(q, int) or isinstance(q, bool)
                for q in edge
            )
        ):
            raise ValueError(f"Invalid edge format: {edge!r}")

        source, target = edge

        if not (0 <= source < num_qubits):
            raise ValueError(f"Invalid qubit index: {source}")
        if not (0 <= target < num_qubits):
            raise ValueError(f"Invalid qubit index: {target}")
        if source == target:
            raise ValueError(f"Self-loop detected: {edge}")

        directed_edge = (source, target)
        if directed_edge in seen:
            raise ValueError(f"Duplicate directed edge: {edge}")

        seen.add(directed_edge)

    # Every physical connection must be represented in both directions.
    for source, target in seen:
        if (target, source) not in seen:
            raise ValueError(
                f"Missing reverse edge for ({source}, {target})"
            )

    return config


def get_undirected_edges(config):
    """Return each physical connection once."""
    return sorted({
        (min(source, target), max(source, target))
        for source, target in config["coupling_map"]
    })


if __name__ == "__main__":
    for processor_name in ("processor_A", "processor_B"):
        processor = load_processor(processor_name)
        unique_edges = get_undirected_edges(processor)

        print(f"{processor_name}:")
        print(f"  Qubits: {processor['num_qubits']}")
        print(f"  Directed entries: {len(processor['coupling_map'])}")
        print(f"  Unique physical edges: {len(unique_edges)}")
        print(f"  Edges: {unique_edges}")
