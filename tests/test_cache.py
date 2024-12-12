"""Tests for the NodeCache implementation."""
import os
import pytest
from typing import Dict, Any

from researchgraph.core.cache import NodeCache
from researchgraph.core.types import NodeOutput


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    return str(cache_path)


@pytest.fixture
def node_cache(cache_dir):
    """Create a NodeCache instance."""
    return NodeCache(cache_dir)


def test_cache_initialization(cache_dir):
    """Test cache initialization."""
    cache = NodeCache(cache_dir)
    assert os.path.exists(cache_dir)


def test_cache_set_get(node_cache):
    """Test setting and getting cached values."""
    test_key = "test_key"
    test_value: Dict[str, NodeOutput] = {"output": "test_value"}

    # Test set
    assert node_cache.set(test_key, test_value) is True

    # Test get
    cached_value = node_cache.get(test_key)
    assert cached_value == test_value


def test_cache_missing_key(node_cache):
    """Test getting non-existent cache key."""
    assert node_cache.get("nonexistent_key") is None


def test_cache_clear(node_cache):
    """Test clearing cache."""
    test_key = "test_key"
    test_value: Dict[str, NodeOutput] = {"output": "test_value"}

    node_cache.set(test_key, test_value)
    assert node_cache.get(test_key) is not None

    assert node_cache.clear() is True
    assert node_cache.get(test_key) is None
