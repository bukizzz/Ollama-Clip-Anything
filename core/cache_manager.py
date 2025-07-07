import os
import pickle
import time
import logging
from typing import Any, Dict, Optional

from core.config import config

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.memory_cache: Dict[str, Any] = {}
        self.disk_cache_dir = os.path.join(config.get('temp_dir'), "cache")
        # Ensure the disk cache directory exists on initialization
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        self.cache_ttl = config.get('cache.cache_ttl', 86400) # Default to 24 hours

    def _get_disk_cache_path(self, key: str) -> str:
        return os.path.join(self.disk_cache_dir, f"{key}.cache")

    def set(self, key: str, value: Any, level: str = "memory"):
        """Sets a value in the specified cache level."""
        if level == "memory":
            self.memory_cache[key] = {'value': value, 'timestamp': time.time()}
            logger.debug(f"Cached '{key}' in memory.")
        elif level == "disk":
            # Ensure the directory exists before writing
            os.makedirs(self.disk_cache_dir, exist_ok=True)
            file_path = self._get_disk_cache_path(key)
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump({'value': value, 'timestamp': time.time()}, f)
                logger.debug(f"Cached '{key}' on disk at {file_path}.")
            except Exception as e:
                logger.error(f"Failed to write to disk cache for '{key}': {e}")
        elif level == "remote":
            logger.warning("Remote caching is not yet implemented.")
        else:
            logger.warning(f"Unknown cache level: {level}")

    def get(self, key: str, level: str = "memory") -> Optional[Any]:
        """Retrieves a value from the specified cache level."""
        if level == "memory":
            entry = self.memory_cache.get(key)
            if entry and (time.time() - entry['timestamp'] < self.cache_ttl):
                logger.debug(f"Retrieved '{key}' from memory cache.")
                return entry['value']
            elif entry:
                logger.debug(f"Memory cache for '{key}' expired.")
                del self.memory_cache[key]
            return None
        elif level == "disk":
            file_path = self._get_disk_cache_path(key)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        entry = pickle.load(f)
                    if time.time() - entry['timestamp'] < self.cache_ttl:
                        logger.debug(f"Retrieved '{key}' from disk cache.")
                        return entry['value']
                    else:
                        logger.debug(f"Disk cache for '{key}' expired. Deleting.")
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Failed to read from disk cache for '{key}': {e}")
                    if os.path.exists(file_path):
                        os.remove(file_path) # Remove corrupted file
            return None
        elif level == "remote":
            logger.warning("Remote caching is not yet implemented.")
            return None
        else:
            logger.warning(f"Unknown cache level: {level}")
            return None

    def delete(self, key: str, level: str = "all"):
        """Deletes a value from cache."""
        if level == "memory" or level == "all":
            if key in self.memory_cache:
                del self.memory_cache[key]
                logger.debug(f"Deleted '{key}' from memory cache.")
        if level == "disk" or level == "all":
            file_path = self._get_disk_cache_path(key)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted '{key}' from disk cache.")
        if level == "remote" or level == "all":
            logger.warning("Remote cache deletion is not yet implemented.")

    def clear_all(self):
        """Clears all cache levels."""
        self.memory_cache = {}
        if os.path.exists(self.disk_cache_dir):
            for f in os.listdir(self.disk_cache_dir):
                os.remove(os.path.join(self.disk_cache_dir, f))
        logger.info("All cache levels cleared.")

# Instantiate a global cache manager
cache_manager = CacheManager()
