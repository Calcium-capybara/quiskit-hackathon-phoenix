from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "processor_B.json"

CONFIG = {
    "name": "processor_B",
    "num_qubits": 9,
    "coupling_map": [
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [1, 6], [6, 1],
        [2, 3], [3, 2],
        [2, 6], [6, 2],
        [2, 5], [5, 2],
        [3, 4], [4, 3],
        [4, 5], [5, 4],
        [5, 6], [6, 5],
        [6, 7], [7, 6],
        [6, 8], [8, 6],
        [8, 0], [0, 8],
    ],
}

OUTPUT.write_text(
    json.dumps(CONFIG, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Generated {OUTPUT}")
