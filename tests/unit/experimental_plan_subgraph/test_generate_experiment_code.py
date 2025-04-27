import pytest
from unittest.mock import patch
from airas.experimental_plan_subgraph.nodes.generate_experiment_code import (
    generate_experiment_code,
)


# Normal case test: generate_experiment_code returns expected string.
@patch(
    "airas.experimental_plan_subgraph.nodes.generate_experiment_code.openai_client",
    return_value="code result",
)
def test_generate_experiment_code_success(mock_openai):
    result = generate_experiment_code(
        llm_name="dummy",
        experiment_details="details",
        base_experimental_code="code",
        base_experimental_info="info",
    )
    assert result == "code result"


# Abnormal case test: generate_experiment_code raises ValueError when openai_client returns None.
@patch(
    "airas.experimental_plan_subgraph.nodes.generate_experiment_code.openai_client",
    return_value=None,
)
def test_generate_experiment_code_no_response(mock_openai):
    with pytest.raises(ValueError):
        generate_experiment_code(
            llm_name="dummy",
            experiment_details="details",
            base_experimental_code="code",
            base_experimental_info="info",
        )
