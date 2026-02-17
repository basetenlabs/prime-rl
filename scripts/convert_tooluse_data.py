"""Convert the paper's tool-use data to the format expected by self_distill_context.py.

Usage:
    uv run python scripts/convert_tooluse_data.py
"""

import json
from pathlib import Path

INPUT_DIR = Path("replication_data")
OUTPUT_DIR = Path("replication_data/converted")


def convert_train(input_path: Path, output_path: Path) -> None:
    with open(input_path) as f:
        data = json.load(f)

    converted = []
    for i, entry in enumerate(data):
        converted.append(
            {
                "example_id": i,
                "messages": [
                    {"role": "user", "content": entry["prompt"]},
                    {"role": "assistant", "content": "\n".join(entry["golden_response"])},
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {len(converted)} training examples -> {output_path}")


def convert_eval(input_path: Path, output_path: Path) -> None:
    with open(input_path) as f:
        data = json.load(f)

    converted = []
    for i, entry in enumerate(data):
        converted.append(
            {
                "example_id": i,
                "prompt": entry["prompt"],
                "instruction": entry["instruction"],
                "golden_answer": entry["golden_answer"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {len(converted)} eval examples -> {output_path}")


if __name__ == "__main__":
    convert_train(INPUT_DIR / "train_data.json", OUTPUT_DIR / "train_data.json")
    convert_eval(INPUT_DIR / "eval_data.json", OUTPUT_DIR / "eval_data.json")
