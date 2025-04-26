import requests
import unittest.mock
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from airas.nodes.retrievenode.github.extract_github_urls import ExtractGithubUrlsNode


class State(BaseModel):
    paper_text: str = Field(default="")
    github_url: list[str] = Field(default_factory=list)


# NOTE：It is executed by Github actions.
def test_extract_github_url_node():
    input_key = ["paper_text"]
    output_key = ["github_url"]

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "ExtractGithubUrlsNode",
        ExtractGithubUrlsNode(
            input_key=input_key,
            output_key=output_key,
        ),
    )
    graph_builder.set_entry_point("ExtractGithubUrlsNode")
    graph_builder.set_finish_point("ExtractGithubUrlsNode")

    graph = graph_builder.compile()
    state = {
        "paper_text": "This is a sample text with a GitHub URL: http://github.com/user/repo and another one: https://github.com/another/repo",
    }
    with unittest.mock.patch("requests.get") as mock_get:
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        result = graph.invoke(state, debug=True)
        mock_get.assert_called()

        assert "github_url" in result
        assert len(result["github_url"]) == 2
        assert "https://github.com/user/repo" in result["github_url"]
        assert "https://github.com/another/repo" in result["github_url"]

        assert mock_get.call_count == 2

    with unittest.mock.patch("requests.get") as mock_get:
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error: Not Found"
        )
        mock_get.return_value = mock_response
        result = graph.invoke(state, debug=True)
        assert "github_url" in result
        assert len(result["github_url"]) == 0
