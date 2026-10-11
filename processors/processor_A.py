from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "processor_A.json"

CONFIG = {
    "name": "processor_A",
    "num_qubits": 5,
    "coupling_map": [
        [0, 1], [1, 0],
        [1, 3], [3, 1],
        [1, 2], [2, 1],
        [4, 3], [3, 4],
    ],
}

OUTPUT.write_text(
    json.dumps(CONFIG, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Generated {OUTPUT}")
