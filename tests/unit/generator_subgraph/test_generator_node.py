import pytest
from unittest.mock import patch
from airas.generator_subgraph.nodes.generator_node import generator_node


# Normal case test: generator_node returns the expected string.
@patch(
    "airas.generator_subgraph.nodes.generator_node.openai_client",
    return_value="generated_method",
)
def test_generator_node_success(mock_openai):
    result = generator_node(
        llm_name="dummy-llm",
        base_method_text="base",
        add_method_text_list=["add1", "add2"],
    )
    assert result == "generated_method"


# Abnormal case test: generator_node raises ValueError when openai_client returns None.
@patch("airas.generator_subgraph.nodes.generator_node.openai_client", return_value=None)
def test_generator_node_no_response(mock_openai):
    with pytest.raises(ValueError):
        generator_node(
            llm_name="dummy-llm",
            base_method_text="base",
            add_method_text_list=["add1", "add2"],
        )
