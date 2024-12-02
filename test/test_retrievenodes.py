from typing import TypedDict
from langgraph.graph import StateGraph

from researchgraph.nodes.retrievenode import RetrievearXivTextNode
from researchgraph.nodes.retrievenode import RetrieveGithubRepositoryNode


class State(TypedDict):
    arxiv_url: str
    paper_text: str
    github_url: str
    folder_structure: str
    github_file: str


def test_retrieve_arxiv_text_node():
    input_variable = ["arxiv_url"]
    output_variable = ["paper_text"]
    save_dir = "/workspaces/researchgraph/data"
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "arxivretriever",
        RetrievearXivTextNode(
            input_variable=input_variable,
            output_variable=output_variable,
            save_dir=save_dir,
        ),
    )
    graph_builder.set_entry_point("arxivretriever")
    graph_builder.set_finish_point("arxivretriever")
    graph = graph_builder.compile()

    state = {
        "arxiv_url": "https://arxiv.org/abs/1604.03540v1",
    }

    assert graph.invoke(state, debug=True)


def test_retrieve_github_repository_node():
    input_variable = ["github_url"]
    output_variable = ["folder_structure", "github_file"]
    save_dir = "/workspaces/researchgraph/data"

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "githubretriever",
        RetrieveGithubRepositoryNode(
            save_dir=save_dir,
            input_variable=input_variable,
            output_variable=output_variable,
        ),
    )
    graph_builder.set_entry_point("githubretriever")
    graph_builder.set_finish_point("githubretriever")
    graph = graph_builder.compile()

    state = {
        "github_url": "https://github.com/adelnabli/acid?tab=readme-ov-file/info/refs"
    }
    assert graph.invoke(state)
