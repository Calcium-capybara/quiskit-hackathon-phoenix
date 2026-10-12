"""Generate the Processor A JSON configuration."""

from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "processor_A.json"

UNDIRECTED_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
]

COUPLING_MAP = [
    list(edge)
    for a, b in UNDIRECTED_EDGES
    for edge in ((a, b), (b, a))
]

CONFIG = {
    "name": "processor_A",
    "num_qubits": 5,
    "coupling_map": COUPLING_MAP,
}

OUTPUT.write_text(
    json.dumps(CONFIG, indent=2) + "\n",
    encoding="utf-8",
)

print(f"Generated {OUTPUT}")
