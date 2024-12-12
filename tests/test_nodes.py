"""Tests for the Node base class implementation."""
import pytest
from typing import Dict, Any, Optional
from unittest.mock import MagicMock

from researchgraph.core.node import Node, NodeExecutionError
from researchgraph.core.types import NodeInput, NodeOutput


class TestNode(Node):
    """Test node implementation."""
    def execute(self, state: Dict[str, NodeInput]) -> Dict[str, NodeOutput]:
        if state.get("fail"):
            raise ValueError("Test error")
        return {"output": state.get("input", "default")}


@pytest.fixture
def test_node(tmp_path):
    """Create a test node instance."""
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "logs"
    cache_dir.mkdir()
    log_dir.mkdir()

    return TestNode(
        input_key=["input"],
        output_key=["output"],
        cache_dir=str(cache_dir),
        log_dir=str(log_dir)
    )


def test_node_initialization():
    """Test node initialization."""
    node = TestNode(input_key=["input"], output_key=["output"])
    assert node.input_key == ["input"]
    assert node.output_key == ["output"]


def test_node_execution(test_node):
    """Test successful node execution."""
    result = test_node({"input": "test_value"})
    assert result["output"] == "test_value"


def test_node_execution_error(test_node):
    """Test node execution error handling."""
    with pytest.raises(NodeExecutionError) as exc_info:
        test_node({"fail": True})
    assert "Test error" in str(exc_info.value)


def test_node_caching(test_node):
    """Test node result caching."""
    state = {"input": "cache_test"}

    # First execution
    result1 = test_node(state)

    # Mock execute to verify cache hit
    test_node.execute = MagicMock()

    # Second execution should use cache
    result2 = test_node(state)

    assert result1 == result2
    test_node.execute.assert_not_called()


def test_node_hooks(test_node):
    """Test node lifecycle hooks."""
    before_called = False
    after_called = False

    def before_hook():
        nonlocal before_called
        before_called = True

    def after_hook():
        nonlocal after_called
        after_called = True

    test_node.before_execute = before_hook
    test_node.after_execute = after_hook

    test_node({"input": "test"})

    assert before_called
    assert after_called
