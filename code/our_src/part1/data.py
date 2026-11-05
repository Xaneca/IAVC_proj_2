"""Global image manifest; reject leakage across all category lists."""
from pathlib import Path

def manifest(split_dir):
    split_dir = Path(split_dir)
    # The checked-in historical lists are the protocol for every stage.
    root = Path(__file__).resolve().parents[1]
    references = [root/'stagec/split_reference', root/'stage_c_setup/split_reference']
    reference = next((p for p in references if p.is_dir()), None)
    if reference is not None:
        expected = {p.name: set(p.read_text().split()) for p in reference.glob('*_*.txt')}
        actual = {p.name: set(p.read_text().split()) for p in split_dir.glob('*_*.txt')
                  if p.stem.endswith(('_train', '_val', '_test'))}
        if actual != expected:
            raise ValueError('Splits differ from the historical A/B protocol; do not regenerate them')
    rows = {}
    for split in ("train", "val", "test"):
        for path in sorted(split_dir.glob(f"*_{split}.txt")):
            category = path.stem.removesuffix(f"_{split}")
            for image_id in path.read_text().splitlines():
                image_id = image_id.strip()
                if not image_id:
                    continue
                if Path(image_id).name != image_id or image_id in (".", ".."):
                    raise ValueError("Image IDs must be plain filenames without a path")
                row = rows.setdefault(image_id, {"image_id": image_id, "split": split, "labels": []})
                if row["split"] != split:
                    raise ValueError(f"Image in different splits/categories: {image_id}")
                row["labels"].append(category)
    for row in rows.values():
        row["labels"] = sorted(set(row["labels"]))
    if not any(r["split"] == "train" for r in rows.values()):
        raise ValueError("No training images found")
    return [rows[i] for i in sorted(rows)]
