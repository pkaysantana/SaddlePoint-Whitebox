"""Generate a synthetic benzene-electrophile topology label dataset."""

from __future__ import annotations

from pathlib import Path

from saddlepoint_whitebox.dataset_generation import generate_topology_dataset


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = generate_topology_dataset(
        data_dir / "topology_labels.jsonl",
        data_dir / "topology_labels_summary.csv",
    )
    print(f"wrote {len(labels)} labels to {data_dir}")


if __name__ == "__main__":
    main()
