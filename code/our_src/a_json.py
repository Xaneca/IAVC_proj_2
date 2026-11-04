"""Small JSON artifact store shared by Stage A steps."""
import hashlib
import json
import os
from pathlib import Path


SCHEMA = 1


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, path)


def file_signature(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
