import pytest
from unittest.mock import patch, MagicMock
from airas.analysis.analytic_subgraph.nodes.analytic_node import analytic_node


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
@patch("airas.analysis.analytic_subgraph.nodes.analytic_node.LLMFacadeClient")
def test_analytic_node_success(mock_llm_client, sample_inputs):
    mock_instance = MagicMock()
    mock_instance.structured_outputs.return_value = (
        {"analysis_report": "This is a report."},
        0.0,
    )
    mock_llm_client.return_value = mock_instance

    result = analytic_node(**sample_inputs)
    assert result == "This is a report."


# Abnormal case test: analytic_node returns None when LLMFacadeClient returns None.
@patch("airas.analysis.analytic_subgraph.nodes.analytic_node.LLMFacadeClient")
def test_analytic_node_no_response(mock_llm_client, sample_inputs):
    mock_instance = MagicMock()
    mock_instance.structured_outputs.return_value = (None, 0.0)
    mock_llm_client.return_value = mock_instance

    result = analytic_node(**sample_inputs)
    assert result is None


# Abnormal case test: analytic_node returns None when 'analysis_report' key is missing in response.
@patch("airas.analysis.analytic_subgraph.nodes.analytic_node.LLMFacadeClient")
def test_analytic_node_no_analysis_report(mock_llm_client, sample_inputs):
    mock_instance = MagicMock()
    mock_instance.structured_outputs.return_value = ({}, 0.0)
    mock_llm_client.return_value = mock_instance

    result = analytic_node(**sample_inputs)
    assert result is None
