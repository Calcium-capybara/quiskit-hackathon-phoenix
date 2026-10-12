"""Generate the Processor B JSON configuration."""

from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "processor_B.json"

UNDIRECTED_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (2, 4),
    (4, 5),
    (5, 6),
    (4, 7),
    (7, 8),
]

COUPLING_MAP = [
    list(edge)
    for a, b in UNDIRECTED_EDGES
    for edge in ((a, b), (b, a))
]

CONFIG = {
    "name": "processor_B",
    "num_qubits": 9,
    "coupling_map": COUPLING_MAP,
}

OUTPUT.write_text(
    json.dumps(CONFIG, indent=2) + "\n",
    encoding="utf-8",
)

print(f"Generated {OUTPUT}")

