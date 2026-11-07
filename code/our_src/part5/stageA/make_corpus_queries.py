"""
Creates the corpus and queries for Part 5 (image retrieval) from PASCAL VOC 2007.

Target class: horse
  - Hard negatives: cow, dog, sheep, cat
  - Easy negatives: aeroplane, bus, bicycle

Rules (VOC is multi-label: an image may contain multiple objects):
  - Horse (relevant): image with at least one horse not marked as "difficult".
    Images where the only horse is "difficult" are excluded (ambiguous relevance).
  - Negative for class C: contains C, contains NO horse, and contains none
    of the other selected classes (to provide a single label).
    People and other VOC classes may appear (realistic context/clutter).

Avoids duplicates / near-duplicates between queries and the corpus:
  1. During selection: a horse is chosen as a query only if it is not a
     near-duplicate of any other collected image.
  2. At the end: verification of the saved files -> near_duplicates_check.csv
  Method: difference hash (dHash, 64 bits) + Hamming distance (including flip).

Usage:
  python create_corpus.py                     # download VOC2007 from Hugging Face
  python create_corpus.py --voc_root PATH     # use an already downloaded VOC2007/VOC2012
                                              # (directory containing Annotations/ and JPEGImages/)

Output:
  voc_retrieval/corpus/       -> corpus images (original resolution)
  voc_retrieval/queries/      -> query images
  corpus.csv                  -> id, path, label, is_relevant, objects
  queries.csv                 -> id, path, label, objects
  near_duplicates_check.csv

Source: PASCAL VOC 2007 (Everingham et al., 2010), http://host.robots.ox.ac.uk/pascal/VOC/
Download mirror: huggingface.co/datasets/JimmyUnleashed/Pascal_VOC
The images come from Flickr; see the terms of use on the official VOC page.
"""

import argparse
import csv
import os
import random
import shutil
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageOps

SEED = 42
TARGET = "horse"
N_QUERIES = 10
N_CORPUS_TARGET = 60

HARD = ["cow", "dog", "sheep", "cat"]
EASY = ["aeroplane", "bus", "bicycle"]
N_PER_HARD = 25
N_PER_EASY = 20

HASH_THRESHOLD = 10        # Hamming distance (over 64 bits) <= this = near-duplicate

OUT_DIR = "voc_retrieval"
CORPUS_DIR = os.path.join(OUT_DIR, "corpus")
QUERY_DIR = os.path.join(OUT_DIR, "queries")

HF_REPO = "JimmyUnleashed/Pascal_VOC"
HF_FILE = "VOCtrainval_06-Nov-2007.tar"


# ------------------------------------------------------------- download -----

def get_voc_root(voc_root_arg):
    if voc_root_arg:
        return voc_root_arg, None

    from huggingface_hub import hf_hub_download

    data_dir = "data"
    voc_root = os.path.join(data_dir, "VOCdevkit", "VOC2007")
    tar_path = None
    if not os.path.isdir(os.path.join(voc_root, "Annotations")):
        print("Downloading PASCAL VOC 2007 (~450 MB)...")
        tar_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
        print("Extracting...")
        os.makedirs(data_dir, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(data_dir)
    return voc_root, tar_path


def cleanup_downloaded_voc(tar_path):
    """Remove the extracted VOC data and the downloaded archive after processing."""
    extracted_dir = os.path.join("data", "VOCdevkit")
    if os.path.isdir(extracted_dir):
        shutil.rmtree(extracted_dir)
        print(f"Removed temporary dataset directory: {extracted_dir}")

    if tar_path and os.path.isfile(tar_path):
        os.remove(tar_path)
        print(f"Removed downloaded archive: {tar_path}")


# ------------------------------------------------------------ annotations -----

def read_annotations(voc_root):
    """Return {image_id: {"all": set of classes, "clear": set of non-difficult classes}}."""
    ann_dir = os.path.join(voc_root, "Annotations")
    info = {}
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".xml"):
            continue
        root = ET.parse(os.path.join(ann_dir, fname)).getroot()
        all_cls, clear_cls = set(), set()
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            diff = obj.find("difficult")
            difficult = diff is not None and diff.text.strip() == "1"
            all_cls.add(name)
            if not difficult:
                clear_cls.add(name)
        info[fname[:-4]] = {"all": all_cls, "clear": clear_cls}
    return info


def candidate_pools(info):
    """Group images by role according to the rules described at the top."""
    chosen = set(HARD + EASY)
    pools = {c: [] for c in [TARGET] + HARD + EASY}
    for img_id, cls in info.items():
        if TARGET in cls["all"]:
            if TARGET in cls["clear"]:
                pools[TARGET].append(img_id)
            continue  # horse is only "difficult" -> exclude
        present = cls["clear"] & chosen
        if len(present) == 1 and not (cls["all"] - cls["clear"]) & chosen:
            pools[present.pop()].append(img_id)
    return pools


# ---------------------------------------------------------------- hashing ---

def dhash(img, hash_size=8):
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hamming(h1, h2):
    return int(np.count_nonzero(h1 != h2))


def hashes_with_flip(img):
    return [dhash(img), dhash(ImageOps.mirror(img))]


def check_near_duplicates(query_paths, corpus_paths, threshold=HASH_THRESHOLD,
                          report_path="near_duplicates_check.csv"):
    """Find the closest corpus image for each query and write a report."""
    corpus_hashes = [(p, dhash(Image.open(p))) for p in corpus_paths]
    rows, duplicates = [], []
    for qp in query_paths:
        qh = hashes_with_flip(Image.open(qp))
        best_path, best_d = None, 65
        for cp, ch in corpus_hashes:
            d = min(hamming(h, ch) for h in qh)
            if d < best_d:
                best_path, best_d = cp, d
        is_dup = best_d <= threshold
        rows.append({"query": qp, "closest_corpus_image": best_path,
                     "hamming_distance": best_d, "near_duplicate": is_dup})
        if is_dup:
            duplicates.append((qp, best_path, best_d))

    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "closest_corpus_image",
                                          "hamming_distance", "near_duplicate"])
        w.writeheader()
        w.writerows(rows)

    if duplicates:
        print(f"WARNING: {len(duplicates)} near-duplicate pair(s):")
        for q, c, d in duplicates:
            print(f"  {q}  <->  {c}  (distance {d})")
    else:
        print(f"No near-duplicates between queries and corpus "
              f"(smallest distance: {min(r['hamming_distance'] for r in rows)}, threshold: {threshold}).")
    return duplicates


# ---------------------------------------------------------------- selection ---

def select(pools, img_path):
    rng = random.Random(SEED)

    quotas = {TARGET: N_QUERIES + N_CORPUS_TARGET}
    quotas.update({c: N_PER_HARD for c in HARD})
    quotas.update({c: N_PER_EASY for c in EASY})

    for c, q in quotas.items():
        print(f"  {c:10s}: {len(pools[c]):4d} candidates, using {q}")
        if len(pools[c]) < q:
            raise RuntimeError(f"Only {len(pools[c])} valid images of '{c}' found (requested {q}).")

    # Negativos
    negatives = {c: rng.sample(pools[c], quotas[c]) for c in HARD + EASY}

    # Horses: shuffle and choose queries that are not near-duplicates of anything
    horses = pools[TARGET][:]
    rng.shuffle(horses)
    pool_ids = horses + [i for ids in negatives.values() for i in ids]
    hashes = {i: dhash(Image.open(img_path(i))) for i in pool_ids}

    queries, rejected = [], 0
    for h in horses:
        if len(queries) == N_QUERIES:
            break
        qh = hashes_with_flip(Image.open(img_path(h)))
        d = min(hamming(a, hashes[o]) for a in qh for o in pool_ids if o != h)
        if d > HASH_THRESHOLD:
            queries.append(h)
        else:
            rejected += 1
    print(f"Candidate queries rejected as near-duplicates: {rejected}")

    remaining = [h for h in horses if h not in set(queries)]
    corpus = dict(negatives)
    corpus[TARGET] = remaining[:N_CORPUS_TARGET]
    return queries, corpus


# ------------------------------------------------------------------ main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_root", default=None,
                        help="Already downloaded VOC directory (with Annotations/ and JPEGImages/)")
    args = parser.parse_args()

    voc_root, tar_path = get_voc_root(args.voc_root)
    img_path = lambda i: os.path.join(voc_root, "JPEGImages", f"{i}.jpg")

    info = read_annotations(voc_root)
    print(f"{len(info)} annotated images in VOC.")
    pools = candidate_pools(info)
    queries, corpus = select(pools, img_path)

    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(QUERY_DIR, exist_ok=True)
    objects = lambda i: ";".join(sorted(info[i]["all"]))

    # Queries
    query_rows = []
    for n, vid in enumerate(queries, start=1):
        qid = f"q_{n:02d}"
        path = os.path.join(QUERY_DIR, f"{qid}_{TARGET}_{vid}.jpg")
        shutil.copy(img_path(vid), path)
        query_rows.append({"id": qid, "path": path, "label": TARGET, "objects": objects(vid)})

    # Corpus (shuffled)
    items = [(c, vid) for c, ids in corpus.items() for vid in ids]
    random.Random(SEED).shuffle(items)
    corpus_rows = []
    for n, (c, vid) in enumerate(items, start=1):
        cid = f"c_{n:04d}"
        path = os.path.join(CORPUS_DIR, f"{cid}_{c}_{vid}.jpg")
        shutil.copy(img_path(vid), path)
        corpus_rows.append({"id": cid, "path": path, "label": c,
                            "is_relevant": c == TARGET, "objects": objects(vid)})

    with open("corpus.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "path", "label", "is_relevant", "objects"])
        w.writeheader()
        w.writerows(corpus_rows)
    with open("queries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "path", "label", "objects"])
        w.writeheader()
        w.writerows(query_rows)

    n_rel = sum(r["is_relevant"] for r in corpus_rows)
    print(f"Corpus: {len(corpus_rows)} images ({n_rel} relevant, "
          f"random baseline AP ≈ {n_rel / len(corpus_rows):.2f})")
    print(f"Queries: {len(query_rows)} images")

    check_near_duplicates([r["path"] for r in query_rows],
                          [r["path"] for r in corpus_rows])

    if args.voc_root is None:
        cleanup_downloaded_voc(tar_path)


if __name__ == "__main__":
    main()