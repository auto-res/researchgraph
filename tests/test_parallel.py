"""Tests for parallel execution in StateGraph."""
import pytest
import time
from typing import Dict, Any

from langgraph.graph import StateGraph
from researchgraph.core.node import Node
from researchgraph.core.types import NodeInput, NodeOutput


class DelayNode(Node):
    """Test node that introduces a delay."""
    def __init__(self, delay: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def execute(self, state: Dict[str, NodeInput]) -> Dict[str, NodeOutput]:
        time.sleep(self.delay)
        return {"output": f"node_{id(self)}"}


def test_parallel_execution():
    """Test parallel execution of nodes."""
    # Create nodes with delays
    node1 = DelayNode(0.5, input_key=["input"], output_key=["output"])
    node2 = DelayNode(0.5, input_key=["input"], output_key=["output"])
    node3 = DelayNode(0.5, input_key=["input"], output_key=["output"])

    # Create graph with parallel execution
    graph = StateGraph(Dict)
    graph.set_parallel_execution(True)

    # Add nodes
    graph.add_node("node1", node1)
    graph.add_node("node2", node2)
    graph.add_node("node3", node3)

    # Add edges for parallel execution
    graph.add_edge("node1", "node3")
    graph.add_edge("node2", "node3")

    # Set entry and end points
    graph.set_entry_point("node1")
    graph.set_finish_point("node3")

    # Compile graph
    workflow = graph.compile()

    # Execute and measure time
    start_time = time.time()
    workflow({"input": "test"})
    execution_time = time.time() - start_time

    # With parallel execution, node1 and node2 should run in parallel
    # Total time should be less than sum of all delays (1.5s)
    assert execution_time < 1.5, "Parallel execution not working as expected"


def test_node_dependencies():
    """Test that node dependencies are respected in parallel execution."""
    results = []

    class OrderedNode(Node):
        def __init__(self, name: str, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = name

        def execute(self, state: Dict[str, NodeInput]) -> Dict[str, NodeOutput]:
            results.append(self.name)
            return {"output": self.name}

    # Create graph with parallel execution
    graph = StateGraph(Dict)
    graph.set_parallel_execution(True)

    # Add nodes
    nodes = {
        name: OrderedNode(name, input_key=["input"], output_key=["output"])
        for name in ["A", "B", "C", "D"]
    }

    for name, node in nodes.items():
        graph.add_node(name, node)

    # Add edges with dependencies
    # A -> B -> D
    # A -> C -> D
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")

    graph.set_entry_point("A")
    graph.set_finish_point("D")

    # Execute workflow
    workflow = graph.compile()
    workflow({"input": "test"})

    # Verify execution order
    assert results[0] == "A"  # A must be first
    assert results[-1] == "D"  # D must be last
    # B and C can be in any order, but must be after A and before D
    assert set(results[1:3]) == {"B", "C"}
