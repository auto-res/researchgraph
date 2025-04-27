import pytest
from unittest.mock import patch
from airas.analytic_subgraph.nodes.analytic_node import analytic_node


@pytest.fixture
def sample_inputs():
    return {
        "llm_name": "dummy-llm",
        "new_method": "new method",
        "verification_policy": "policy",
        "experiment_code": "code",
        "output_text_data": "output",
    }


# Normal case test: analytic_node returns the expected analysis_report string.
@patch(
    "airas.analytic_subgraph.nodes.analytic_node.openai_client",
    return_value='{"analysis_report": "This is a report."}',
)
def test_analytic_node_success(mock_openai, sample_inputs):
    result = analytic_node(**sample_inputs)
    assert result == "This is a report."


# Abnormal case test: analytic_node returns None when openai_client returns None.
@patch("airas.analytic_subgraph.nodes.analytic_node.openai_client", return_value=None)
def test_analytic_node_no_response(mock_openai, sample_inputs):
    result = analytic_node(**sample_inputs)
    assert result is None


# Abnormal case test: analytic_node returns None when 'analysis_report' key is missing in response.
@patch("airas.analytic_subgraph.nodes.analytic_node.openai_client", return_value="{}")
def test_analytic_node_no_analysis_report(mock_openai, sample_inputs):
    result = analytic_node(**sample_inputs)
    assert result is None
