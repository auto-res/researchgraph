import pytest
from unittest.mock import patch
from airas.experimental_plan_subgraph.nodes.generate_experiment_details import (
    generate_experiment_details,
)


# Normal case test: generate_experiment_details returns expected string.
@patch(
    "airas.experimental_plan_subgraph.nodes.generate_experiment_details.openai_client",
    return_value="details result",
)
def test_generate_experiment_details_success(mock_openai):
    result = generate_experiment_details(
        llm_name="dummy",
        verification_policy="policy",
        base_experimental_code="code",
        base_experimental_info="info",
    )
    assert result == "details result"


# Abnormal case test: generate_experiment_details raises ValueError when openai_client returns None.
@patch(
    "airas.experimental_plan_subgraph.nodes.generate_experiment_details.openai_client",
    return_value=None,
)
def test_generate_experiment_details_no_response(mock_openai):
    with pytest.raises(ValueError):
        generate_experiment_details(
            llm_name="dummy",
            verification_policy="policy",
            base_experimental_code="code",
            base_experimental_info="info",
        )
