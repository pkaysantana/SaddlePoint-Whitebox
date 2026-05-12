"""Generate a synthetic benzene-electrophile topology label dataset."""

from __future__ import annotations

from pathlib import Path

from saddlepoint_whitebox.dataset_generation import generate_topology_dataset


def main() -> None:
    data_dir = Path("data")
    jsonl_path = data_dir / "topology_labels.jsonl"
    csv_path = data_dir / "topology_labels_summary.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = generate_topology_dataset(
        jsonl_path,
        csv_path,
    )
    print(f"wrote {len(labels)} labels")
    print(f"jsonl: {jsonl_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
