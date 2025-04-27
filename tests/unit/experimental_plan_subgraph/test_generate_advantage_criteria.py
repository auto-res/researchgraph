import pytest
from unittest.mock import patch
from airas.experimental_plan_subgraph.nodes.generate_advantage_criteria import (
    generate_advantage_criteria,
)


# Normal case test: generate_advantage_criteria returns expected string.
@patch(
    "airas.experimental_plan_subgraph.nodes.generate_advantage_criteria.openai_client",
    return_value="criteria result",
)
def test_generate_advantage_criteria_success(mock_openai):
    result = generate_advantage_criteria(llm_name="dummy", new_method="method")
    assert result == "criteria result"


# Abnormal case test: generate_advantage_criteria raises ValueError when openai_client returns None.
@patch(
    "airas.experimental_plan_subgraph.nodes.generate_advantage_criteria.openai_client",
    return_value=None,
)
def test_generate_advantage_criteria_no_response(mock_openai):
    with pytest.raises(ValueError):
        generate_advantage_criteria(llm_name="dummy", new_method="method")
