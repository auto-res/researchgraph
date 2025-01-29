import json
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from researchgraph.nodes.retrievenode.github.retrieve_github_url import RetrieveGithubUrlNode


class State(BaseModel):
    paper_text: str = Field(default="")
    github_url: list[str] = Field(default_factory=list)

# NOTE：It is executed by Github actions.
def test_retrieve_github_url_node():
    input_key = ["paper_text"]
    output_key = ["github_url"]

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "RetrieveGithubUrlNode",
        RetrieveGithubUrlNode(
            input_key=input_key,
            output_key=output_key,
        ),
    )
    graph_builder.set_entry_point("RetrieveGithubUrlNode")
    graph_builder.set_finish_point("RetrieveGithubUrlNode")

    graph = graph_builder.compile()
    state = {
        "paper_text": "This is a sample text with a GitHub URL: http://github.com/user/repo and another one: https://github.com/another/repo",
    }
    result = graph.invoke(state, debug=True)
    assert "github_url" in result
    assert len(result["github_url"]) == 2
    assert "https://github.com/user/repo" in result["github_url"]
    assert "https://github.com/another/repo" in result["github_url"]
