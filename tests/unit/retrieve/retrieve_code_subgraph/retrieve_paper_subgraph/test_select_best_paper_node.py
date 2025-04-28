from unittest.mock import patch
from airas.retrieve_paper_subgraph.nodes.select_best_paper_node import (
    select_best_paper_node,
)


# Normal case: vertexai_client returns a valid arxiv_id string
@patch(
    "airas.retrieve_paper_subgraph.nodes.select_best_paper_node.vertexai_client",
    return_value={"selected_arxiv_id": "1234.5678v1\n2345.6789v2"},
)
def test_select_best_paper_node_success(mock_vertexai):
    llm_name = "dummy-llm"
    prompt_template = "template"
    candidate_papers = ["paper1", "paper2"]
    result = select_best_paper_node(llm_name, prompt_template, candidate_papers)
    assert result == ["1234.5678v1", "2345.6789v2"]


# Abnormal case: vertexai_client returns dict without selected_arxiv_id
@patch(
    "airas.retrieve_paper_subgraph.nodes.select_best_paper_node.vertexai_client",
    return_value={},
)
def test_select_best_paper_node_no_selected_id(mock_vertexai):
    llm_name = "dummy-llm"
    prompt_template = "template"
    candidate_papers = ["paper1", "paper2"]
    result = select_best_paper_node(llm_name, prompt_template, candidate_papers)
    assert result == []
