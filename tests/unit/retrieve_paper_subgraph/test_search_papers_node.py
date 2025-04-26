from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from airas.nodes.retrievenode.search_papers_node import SearchPapersNode
from unittest.mock import patch


class State(BaseModel):
    queries: list = Field(default_factory=list)
    search_results: list[dict] = Field(default_factory=list)


# NOTE：It is executed by Github actions.
def test_search_paper_node():
    input_key = ["queries"]
    output_key = ["search_results"]
    period_days = 30
    num_retrieve_paper = 2
    api_type = "arxiv"
    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "search_papers_node",
        SearchPapersNode(
            input_key=input_key,
            output_key=output_key,
            period_days=period_days,
            num_retrieve_paper=num_retrieve_paper,
            api_type=api_type,
        ),
    )
    graph_builder.set_entry_point("search_papers_node")
    graph_builder.set_finish_point("search_papers_node")
    graph = graph_builder.compile()
    state = {
        "queries": ["deep learning"],
    }
    with patch(
        "researchgraph.nodes.retrievenode.arxiv_api.arxiv_api_node.ArxivNode.search_paper"
    ) as mock_search_paper:
        mock_search_paper.return_value = [
            {
                "arxiv_id": "1234.5678",
                "arxiv_url": "https://arxiv.org/abs/1234.5678",
                "title": "Paper 1",
                "authors": [{"name": "Author 1"}, {"name": "Author 2"}],
                "publication_date": "2023-10-01",
                "journal": "Journal 1",
                "externalIds": {"ArXiv": "1234.5678"},
            },
            {
                "arxiv_id": "2345.6789",
                "arxiv_url": "https://arxiv.org/abs/2345.6789",
                "title": "Paper 2",
                "authors": [{"name": "Author 3"}, {"name": "Author 4"}],
                "publication_date": "2023-10-02",
                "journal": "Journal 2",
                "externalIds": {"ArXiv": "2345.6789"},
            },
        ]
        result = graph.invoke(state, debug=True)
        assert result == {
            "queries": ["deep learning"],
            "search_results": [
                {
                    "arxiv_id": "1234.5678",
                    "arxiv_url": "https://arxiv.org/abs/1234.5678",
                    "title": "Paper 1",
                    "authors": [{"name": "Author 1"}, {"name": "Author 2"}],
                    "publication_date": "2023-10-01",
                    "journal": "Journal 1",
                    "externalIds": {"ArXiv": "1234.5678"},
                },
                {
                    "arxiv_id": "2345.6789",
                    "arxiv_url": "https://arxiv.org/abs/2345.6789",
                    "title": "Paper 2",
                    "authors": [{"name": "Author 3"}, {"name": "Author 4"}],
                    "publication_date": "2023-10-02",
                    "journal": "Journal 2",
                    "externalIds": {"ArXiv": "2345.6789"},
                },
            ],
        }
