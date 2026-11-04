"""Split supplied train by image, preserving multilabel proportions.

The supplied val is reserved as test. No external dependencies required.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def create_splits(source, output, validation_fraction=0.2, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    labels = {"train": defaultdict(set), "val": defaultdict(set)}
    for subset in labels:
        for path in sorted(source.glob(f"*_{subset}.txt")):
            category = path.stem.removesuffix(f"_{subset}")
            for image in path.read_text().splitlines():
                if image.strip():
                    labels[subset][image.strip()].add(category)
    if not labels["train"] or not labels["val"]:
        raise ValueError("Expected nonempty supplied train and val partitions")
    if set(labels["train"]) & set(labels["val"]):
        raise ValueError("Supplied train and val contain overlapping image IDs")

    # Iterative multilabel stratification: allocate the rarest remaining label
    # first, preferring the fold with the largest unmet quota for that label.
    rng = random.Random(seed)
    ids = sorted(labels["train"])
    categories = sorted(set().union(*[v for group in labels.values() for v in group.values()]))
    n_val = round(len(ids) * validation_fraction)
    if not 0 < n_val < len(ids):
        raise ValueError("Not enough images for two nonempty partitions")
    remaining_capacity = {"train": len(ids) - n_val, "val": n_val}
    counts = Counter(c for cats in labels["train"].values() for c in cats)
    fractions = {"train": 1 - validation_fraction, "val": validation_fraction}
    desired = {fold: {c: counts[c] * fractions[fold] for c in categories}
               for fold in fractions}
    remaining = set(ids)
    assigned = {}
    while remaining:
        pending = Counter(c for image in sorted(remaining) for c in labels["train"][image])
        rare_count = min(pending.values())
        rare = rng.choice(sorted(c for c, n in pending.items() if n == rare_count))
        candidates = sorted(image for image in remaining if rare in labels["train"][image])
        rng.shuffle(candidates)
        for image in candidates:
            available = [f for f in ("train", "val") if remaining_capacity[f] > 0]
            best = max(desired[f][rare] for f in available)
            available = [f for f in available if desired[f][rare] == best]
            capacity = max(remaining_capacity[f] for f in available)
            fold = rng.choice([f for f in available if remaining_capacity[f] == capacity])
            assigned[image] = fold
            remaining.remove(image)
            remaining_capacity[fold] -= 1
            for category in labels["train"][image]:
                desired[fold][category] -= 1

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for supplied_subset, mapping in labels.items():
        for image, cats in sorted(mapping.items()):
            split = assigned[image] if supplied_subset == "train" else "test"
            rows.append({"image_id": image, "split": split,
                         "source_split": supplied_subset,
                         **{c: int(c in cats) for c in categories}})
    rows.sort(key=lambda r: r["image_id"])
    assert len({r["image_id"] for r in rows}) == len(rows)
    for fold in ("train", "val", "test"):
        for category in categories:
            image_ids = [r["image_id"] for r in rows if r["split"] == fold and r[category]]
            (output / f"{category}_{fold}.txt").write_text(
                "\n".join(image_ids) + ("\n" if image_ids else ""), encoding="utf-8")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "split", "source_split", *categories])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seed": seed, "validation_fraction_of_supplied_train": validation_fraction,
        "method": "iterative multilabel stratification with exact fold sizes",
        "source": str(source.resolve()),
        "unique_images": dict(Counter(r["split"] for r in rows)),
        "category_counts": {c: {f: sum(r[c] for r in rows if r["split"] == f)
                                for f in ("train", "val", "test")} for c in categories},
    }
    (output / "split_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "data/splits")
    parser.add_argument("--output", type=Path, default=root / "data/splits_internal")
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_splits(args.source, args.output, args.validation_fraction, args.seed)
