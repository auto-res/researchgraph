"""Tests for the NodeLogger implementation."""
import os
import json
import pytest
from typing import Dict, Any

from researchgraph.core.logging import NodeLogger
from researchgraph.core.types import NodeInput, NodeOutput, NodeState


@pytest.fixture
def log_dir(tmp_path):
    """Create a temporary log directory."""
    log_path = tmp_path / "logs"
    log_path.mkdir()
    return str(log_path)


@pytest.fixture
def node_logger(log_dir):
    """Create a NodeLogger instance."""
    return NodeLogger(log_dir)


def test_logger_initialization(log_dir):
    """Test logger initialization."""
    logger = NodeLogger(log_dir)
    assert os.path.exists(log_dir)


def test_log_start(node_logger, log_dir):
    """Test logging start of node execution."""
    node_name = "test_node"
    input_state: Dict[str, NodeInput] = {"input": "test_input"}

    node_logger.log_start(node_name, input_state)

    log_file = os.path.join(log_dir, f"{node_name}.json")
    assert os.path.exists(log_file)

    with open(log_file, 'r') as f:
        log_data = json.load(f)
        assert log_data["node"] == node_name
        assert log_data["input_state"] == input_state


def test_log_complete(node_logger):
    """Test logging completion of node execution."""
    node_name = "test_node"
    output_state: Dict[str, NodeOutput] = {"output": "test_output"}
    execution_time = 1.0

    result = node_logger.log_complete(node_name, output_state, execution_time)

    assert result["success"] is True
    assert result["execution_time"] == execution_time


def test_log_error(node_logger):
    """Test logging node execution error."""
    node_name = "test_node"
    error = ValueError("Test error")
    state: NodeState = {
        "input_data": {"input": "test_input"},
        "output_data": None,
        "status": "error",
        "error": str(error),
        "metadata": {"test": "metadata"}
    }

    result = node_logger.log_error(node_name, error, state)

    assert result["success"] is False
    assert result["error"] == str(error)
