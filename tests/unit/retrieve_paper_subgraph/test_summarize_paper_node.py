from unittest.mock import patch
from airas.retrieve_paper_subgraph.nodes.summarize_paper_node import (
    summarize_paper_node,
)


# Normal case: vertexai_client returns all required fields
@patch(
    "airas.retrieve_paper_subgraph.nodes.summarize_paper_node.vertexai_client",
    return_value={
        "main_contributions": "contrib",
        "methodology": "method",
        "experimental_setup": "setup",
        "limitations": "limits",
        "future_research_directions": "future",
    },
)
def test_summarize_paper_node_success(mock_vertexai):
    llm_name = "dummy-llm"
    prompt_template = "template"
    paper_text = "text"
    result = summarize_paper_node(llm_name, prompt_template, paper_text)
    assert result == ("contrib", "method", "setup", "limits", "future")
