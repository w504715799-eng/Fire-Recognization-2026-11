from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


class ProjectError(RuntimeError):
    pass


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".write-", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child(root, relative):
    """Reject traversal; shared input symlinks deliberately remain permitted."""
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ProjectError(f"Expected a relative path without '..': {relative}")
    return Path(root) / p


def output_path(root, relative):
    p = child(root, relative).resolve()
    if not p.is_relative_to((Path(root) / "artifacts").resolve()):
        raise ProjectError("Generated files must stay under this project's artifacts/")
    # Do not write through an artifacts symlink to shared inputs.
    if not p.is_relative_to(Path(root).resolve()):
        raise ProjectError("Output resolves outside the project")
    return p


@contextlib.contextmanager
def file_lock(path):
    """Cooperative cross-process lock; kernel releases it on process termination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProjectError(f"Another task owns the lock: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
