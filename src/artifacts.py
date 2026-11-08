"""Atomic log replacement with bounded retries for transient Windows file errors."""

import errno
import json
import os
from pathlib import Path
import time
from uuid import uuid4


def _atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        for attempt in range(6):
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                return
            except OSError as error:
                transient = (error.errno in {errno.EINVAL, errno.EACCES, errno.EBUSY}
                             or getattr(error, "winerror", None) in {5, 32, 33})
                if not transient or attempt == 5:
                    raise
                time.sleep(min(.1 * 2 ** attempt, 1.6))
    finally:
        # A failed replacement must leave the previous complete log intact.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_json(path, payload):
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _atomic_text(path, content)


def write_csv(path, frame):
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))
