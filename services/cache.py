"""
Simple JSON file cache for scan results.

Stores cached data as JSON files in a configurable cache directory.
Provides convenience methods for scan results and account lookups.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent


class CacheManager:
    """Simple JSON file cache for scan results and other data.

    Each cache entry is stored as ``cache_dir/<key>.json``.  The cache
    directory is resolved relative to PROJECT_DIR (the parent of the
    ``services/`` package).

    Usage::

        cache = CacheManager()
        cache.save("scan_results", accounts_data)
        data = cache.load("scan_results")
        if cache.is_fresh("scan_results", max_age_hours=12):
            ...
    """

    def __init__(self, cache_dir: str = "data") -> None:
        # Resolve relative to PROJECT_DIR
        self.cache_dir = PROJECT_DIR / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Cache directory: %s", self.cache_dir)

    # -- Core methods --------------------------------------------------------

    def _path(self, key: str) -> Path:
        """Return the file path for a given cache key."""
        # Sanitize key to avoid directory traversal
        safe_key = key.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.cache_dir / f"{safe_key}.json"

    def save(self, key: str, data: Any) -> None:
        """Save *data* to ``cache_dir/<key>.json``.

        Args:
            key: Cache key (becomes the filename).
            data: Any JSON-serialisable data.
        """
        path = self._path(key)
        try:
            path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Cached %s → %s", key, path)
        except (OSError, TypeError) as exc:
            logger.error("Failed to cache %s: %s", key, exc)

    def load(self, key: str) -> Optional[Any]:
        """Load data from ``cache_dir/<key>.json``.

        Returns:
            The deserialized data, or ``None`` if the file does not exist
            or cannot be parsed.
        """
        path = self._path(key)
        if not path.exists():
            logger.debug("Cache miss: %s", key)
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load cache %s: %s", key, exc)
            return None

    def is_fresh(self, key: str, max_age_hours: int = 24) -> bool:
        """Check whether a cache entry exists and is younger than *max_age_hours*.

        Args:
            key: Cache key.
            max_age_hours: Maximum age in hours.  Defaults to 24.

        Returns:
            ``True`` if the cache file exists and its modification time is
            within the allowed window.
        """
        path = self._path(key)
        if not path.exists():
            return False

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return False

        age_hours = (time.time() - mtime) / 3600.0
        return age_hours < max_age_hours

    def clear(self, key: Optional[str] = None) -> None:
        """Clear cache entries.

        Args:
            key: If given, delete only ``<key>.json``.  If ``None``,
                delete **all** ``*.json`` files in the cache directory.
        """
        if key is not None:
            path = self._path(key)
            if path.exists():
                try:
                    path.unlink()
                    logger.info("Cleared cache: %s", key)
                except OSError as exc:
                    logger.error("Failed to clear cache %s: %s", key, exc)
        else:
            count = 0
            for f in self.cache_dir.glob("*.json"):
                try:
                    f.unlink()
                    count += 1
                except OSError as exc:
                    logger.error("Failed to delete %s: %s", f, exc)
            logger.info("Cleared %d cache file(s) from %s", count, self.cache_dir)

    # -- Convenience methods -------------------------------------------------

    def get_scan_results(self) -> Optional[list[dict]]:
        """Load the ``scan_results`` cache entry.

        Returns:
            List of account dicts, or ``None`` if not cached.
        """
        return self.load("scan_results")  # type: ignore[return-value]

    def save_scan_results(self, accounts: list) -> None:
        """Save the ``scan_results`` cache entry.

        Args:
            accounts: List of account dicts or ``AccountData`` objects
                (must support ``to_dict()`` or be dict-like).
        """
        data = []
        for acct in accounts:
            if hasattr(acct, "to_dict"):
                data.append(acct.to_dict())
            else:
                data.append(acct)
        self.save("scan_results", data)

    def get_account(self, user_id: str) -> Optional[dict]:
        """Find a specific account in the cached scan results by *user_id*.

        Args:
            user_id: The Telnyx user ID to look up.

        Returns:
            The matching account dict, or ``None`` if not found.
        """
        results = self.get_scan_results()
        if not results:
            return None

        for acct in results:
            if isinstance(acct, dict) and acct.get("user_id") == user_id:
                return acct
        return None
