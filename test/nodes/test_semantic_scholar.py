from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from researchgraph.nodes.retrievenode.semantic_scholar.semantic_scholar import (
    SemanticScholarNode,
)
from unittest.mock import patch


class State(BaseModel):
     queries : list = Field(default_factory=list)
     search_results: list[dict] = Field(default_factory=list)


# NOTE：It is executed by Github actions.
@patch("semanticscholar.SemanticScholar.search_paper")
def test_semantic_scholar_node(mock_search_paper):
    mock_search_paper.return_value = [
        {
            "title": "Mock Paper 1",
            "authors": [{"name": "Author One"}],
            "publicationDate": "2023-01-01",
            "journal": "Jouranal One",
            "doi": "http", 
            "externalIds": {"ArXiv": "1234.5678"},
        },
        {
            "title": "Mock Paper 2",
            "authors": [{"name": "Author Two"}],
            "publicationDate": "2023-02-02",
            "journal": "Journal Two",
            "doi": "http",
            "externalIds": {"ArXiv": "8765.4321"},
        },
    ]
    input_key = ["queries"]
    output_key = ["search_results"]

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "semanticscholarretriever",
        SemanticScholarNode(
            input_key=input_key,
            output_key=output_key,
            num_retrieve_paper=2,
        ),
    )
    graph_builder.set_entry_point("semanticscholarretriever")
    graph_builder.set_finish_point("semanticscholarretriever")
    graph = graph_builder.compile()

    memory = {"queries": ["Grokking"]}
    result = graph.invoke(memory, debug=True)
    assert len(result["search_results"]) == 2
    assert result["search_results"][0]["arxiv_url"] == "https://arxiv.org/abs/1234.5678"
    assert result["search_results"][1]["arxiv_url"] == "https://arxiv.org/abs/8765.4321"
    assert result["search_results"][0]["paper_title"] == "Mock Paper 1"
    assert result["search_results"][1]["paper_title"] == "Mock Paper 2"