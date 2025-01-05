import os
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph

from researchgraph.nodes.retrievenode import RetrievearXivTextNode
from researchgraph.nodes.retrievenode import RetrieveGithubRepositoryNode


class State(BaseModel):
    arxiv_url: str = Field(default="")
    paper_text: str = Field(default="")
    github_url: str = Field(default="")
    folder_structure: str = Field(default="")
    github_file: str = Field(default="")


SAVE_DIR = os.environ.get("SAVE_DIR", "/workspaces/researchgraph/data")


def test_retrieve_arxiv_text_node():
    input_key = ["arxiv_url"]
    output_key = ["paper_text"]
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "arxivretriever",
        RetrievearXivTextNode(
            input_key=input_key,
            output_key=output_key,
            save_dir=SAVE_DIR,
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
    input_key = ["github_url"]
    output_key = ["folder_structure", "github_file"]
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "githubretriever",
        RetrieveGithubRepositoryNode(
            input_key=input_key,
            output_key=output_key,
            save_dir=SAVE_DIR,
        ),
    )
    graph_builder.set_entry_point("githubretriever")
    graph_builder.set_finish_point("githubretriever")
    graph = graph_builder.compile()

    state = {
        "github_url": "https://github.com/adelnabli/acid?tab=readme-ov-file/info/refs"
    }
    assert graph.invoke(state, debug=True)
