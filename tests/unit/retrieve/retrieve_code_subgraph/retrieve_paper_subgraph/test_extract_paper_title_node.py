from unittest.mock import patch
from airas.retrieve_paper_subgraph.nodes.extract_paper_title_node import (
    extract_paper_title_node,
)


# Normal case: openai_client returns valid paper_titles for each scraped result
@patch(
    "airas.retrieve_paper_subgraph.nodes.extract_paper_title_node.openai_client",
    side_effect=[
        '{"paper_titles": ["Title1", "Title2"]}',
        '{"paper_titles": ["Title3"]}',
    ],
)
def test_extract_paper_title_node_success(mock_openai):
    llm_name = "dummy-llm"
    queries = ["query1"]
    scraped_results = ["result1", "result2"]
    result = extract_paper_title_node(llm_name, queries, scraped_results)
    assert result == ["Title1", "Title2", "Title3"]


# Abnormal case: openai_client returns None for one result
@patch(
    "airas.retrieve_paper_subgraph.nodes.extract_paper_title_node.openai_client",
    side_effect=[None, '{"paper_titles": ["Title3"]}'],
)
def test_extract_paper_title_node_partial_none(mock_openai):
    llm_name = "dummy-llm"
    queries = ["query1"]
    scraped_results = ["result1", "result2"]
    result = extract_paper_title_node(llm_name, queries, scraped_results)
    assert result == ["Title3"]
