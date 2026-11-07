"""A JSON file of LLM results keyed by a hash of everything that decides them."""

import hashlib
import json
import threading
from pathlib import Path


def hash_key(parts: list[str]) -> str:
    """A stable key for a call: the hash of its model, instructions and prompt."""
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


class JsonCache:
    """Results saved in a JSON file, so rerunning never pays twice for the same call."""

    def __init__(self, path: Path) -> None:
        """Load the results saved so far.

        Args:
            path: The cache file.
        """
        self.path = path
        self.entries: dict[str, dict[str, object]] = (
            json.loads(path.read_text()) if path.exists() else {}
        )
        self.lock = threading.Lock()

    def get(self, key: str) -> dict[str, object] | None:
        """The saved result for a key, or None."""
        return self.entries.get(key)

    def put(self, key: str, value: dict[str, object]) -> None:
        """Remember a result; `save` writes it to disk."""
        with self.lock:
            self.entries[key] = value

    def save(self) -> None:
        """Write every result to the cache file."""
        with self.lock:
            self.path.write_text(json.dumps(self.entries, indent=1, ensure_ascii=False))
