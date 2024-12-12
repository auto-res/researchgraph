"""Cache implementation for research graph nodes."""
import os
import json
import hashlib
import logging
from typing import Any, Optional, Dict, Union

from .types import NodeState, NodeResult, NodeInput, NodeOutput


class NodeCache:
    """Cache implementation for storing and retrieving node results."""

    def __init__(self, cache_dir: str) -> None:
        """Initialize NodeCache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(self.__class__.__name__)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            self.logger.info(f"Created cache directory: {cache_dir}")

    def _generate_key(self, key: str) -> str:
        """Generate a cache key from input string.

        Args:
            key: Input string to generate key from

        Returns:
            Cache key as string
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for a key.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        hashed_key = self._generate_key(key)
        return os.path.join(self.cache_dir, f"{hashed_key}.json")

    def get(self, key: str) -> Optional[Dict[str, NodeOutput]]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if exists, None otherwise
        """
        cache_path = self._get_cache_path(key)

        if not os.path.exists(cache_path):
            self.logger.debug(f"Cache miss for key: {key}")
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                value = json.load(f)
                self.logger.debug(f"Cache hit for key: {key}")
                return value
        except Exception as e:
            self.logger.error(f"Failed to read cache for key {key}: {str(e)}")
            return None

    def set(self, key: str, value: Dict[str, NodeOutput]) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(value, f)
            self.logger.debug(f"Cached value for key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write cache for key {key}: {str(e)}")
            return False

    def clear(self) -> bool:
        """Clear all cached values.

        Returns:
            True if successful, False otherwise
        """
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            self.logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
            return False
