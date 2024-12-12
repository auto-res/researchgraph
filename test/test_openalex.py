import os
from typing import TypedDict
from langgraph.graph import StateGraph
from researchgraph.nodes.retrievenode.open_alex.openalex import OpenAlexNode
from unittest.mock import patch


class State(TypedDict):
    keywords: list[str]
    paper_results: dict


@patch("pyalex.Works.filter")
def test_openalex(mock_filter):
    mock_filter.return_value.search.return_value.get.return_value = [
        {
            "abstract": "This is a mock abstract.",
            "author": "Mock Author",
            "publication_date": "2023-01-01",
            "indexed_in": ["arxiv"],
            "locations": [{"landing_page_url": "https://arxiv.org/abs/1234.5678"}],
        },
    ]

    SAVE_DIR = os.environ.get("SAVE_DIR", "/workspaces/researchgraph/data")
    input_key = ["keywords"]
    output_key = ["paper_results"]

    memory = {"keywords": '["Grokking"]'}

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "openalexretriever",
        OpenAlexNode(
            input_key=input_key,
            output_key=output_key,
            save_dir=SAVE_DIR,
            num_keywords=1,
            num_retrieve_paper=3,
        ),
    )
    graph_builder.set_entry_point("openalexretriever")
    graph_builder.set_finish_point("openalexretriever")
    graph = graph_builder.compile()

    memory = {"keywords": '["Grokking"]'}

    assert graph.invoke(memory, debug=True)
