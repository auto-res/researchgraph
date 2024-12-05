from typing import TypedDict
from langgraph.graph import StateGraph
from researchgraph.nodes.retrievenode.semantic_scholar.semantic_scholar import SemanticScholarNode
from unittest.mock import patch


class State(TypedDict):
    keywords: str
    collection_of_papers: dict

@patch("semanticscholar.SemanticScholar.search_paper")
def test_semantic_scholar_node(mock_search_paper):
    mock_search_paper.return_value = [
        {
            "title": "Mock Paper 1",
            "abstract": "This is a mock abstract.",
            "authors": [{"name": "Author One"}],
            "publicationDate": "2023-01-01",
            "arxivId": "1234.5678"
        },
        {
            "title": "Mock Paper 2",
            "abstract": "This is another mock abstract.",
            "authors": [{"name": "Author Two"}],
            "publicationDate": "2023-01-02",
            "arxivId": "2345.6789"
        }
    ]

    save_dir = "/workspaces/researchgraph/data"
    input_key = ["keywords"]
    output_key = ["collection_of_papers"]

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "semanticscholarretriever",
        SemanticScholarNode(
            input_key=input_key,
            output_key=output_key,
            save_dir=save_dir,
            num_retrieve_paper=3,
        ),
    )
    graph_builder.set_entry_point("semanticscholarretriever")
    graph_builder.set_finish_point("semanticscholarretriever")
    graph = graph_builder.compile()

    memory = {"keywords": '["Grokking"]'}

    graph.invoke(memory, debug=True)