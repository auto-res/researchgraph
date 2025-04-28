import pytest
from unittest.mock import patch
from airas.generator_subgraph.generator_subgraph import GeneratorSubgraph


@pytest.fixture
def dummy_input():
    return {
        "base_method_text": "Base method",
        "add_method_texts": ["Add method 1", "Add method 2"],
    }


@pytest.fixture
def expected_output():
    return {"new_method": "Generated new method"}


@patch(
    "airas.generator_subgraph.generator_subgraph.generator_node",
    return_value="Generated new method",
)
def test_generator_subgraph(mock_generator_node, dummy_input, expected_output):
    subgraph = GeneratorSubgraph(llm_name="dummy-llm")
    graph = subgraph.build_graph()
    result = graph.invoke(dummy_input)
    assert result["new_method"] == expected_output["new_method"]
