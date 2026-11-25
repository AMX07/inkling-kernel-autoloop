"""Small, portable artifacts with hashes; never deserialize uploaded pickle files."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest(folder, *, kind, status, metadata=None):
    folder = Path(folder)
    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append({"path": str(path.relative_to(folder)), "bytes": path.stat().st_size,
                          "sha256": digest.hexdigest()})
    versions = {}
    for package in ("torch", "vllm", "modal"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    value = {"schema_version": 1, "kind": kind, "status": status, "generated_unix": time.time(),
             "python": platform.python_version(), "platform": platform.platform(),
             "versions": versions, "files": files, "metadata": metadata or {}}
    write_json(folder / "manifest.json", value)
    return value
