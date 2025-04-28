import pytest
from unittest.mock import patch
import json
from airas.html_subgraph.html_subgraph import HtmlSubgraph


@pytest.fixture
def dummy_input():
    return {
        "paper_content": {
            "Title": "Test Paper",
            "Abstract": "This is a test abstract.",
            "Introduction": "Test introduction.",
        }
    }


@pytest.fixture
def expected_output():
    return {
        "paper_html_content": "<div>HTML Content</div>",
        "full_html": "<html><body>HTML Content</body></html>",
    }


@patch(
    "airas.html_subgraph.nodes.render_html.openai_client",
    return_value=json.dumps({"full_html": "<html><body>HTML Content</body></html>"}),
)
@patch(
    "airas.html_subgraph.nodes.convert_to_html.openai_client",
    return_value=json.dumps({"paper_html_content": "<div>HTML Content</div>"}),
)
def test_html_subgraph(
    mock_convert_openai_client, mock_render_openai_client, dummy_input, expected_output
):
    subgraph = HtmlSubgraph(llm_name="dummy-llm", save_dir="/tmp")
    graph = subgraph.build_graph()
    result = graph.invoke(dummy_input)
    assert result["full_html"] == expected_output["full_html"]
