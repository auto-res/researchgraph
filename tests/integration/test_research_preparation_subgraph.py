import pytest
from unittest.mock import patch
from airas.research_preparation_subgraph.research_preparation_subgraph import (
    ResearchPreparationSubgraph,
)


@pytest.fixture
def dummy_input():
    return {"github_repository": "test-owner/test-repo", "branch_name": "test-branch"}


@pytest.fixture
def expected_output():
    return {
        "repository_exists": True,
        "fork_result": True,
        "target_branch_sha": "sha123",
        "main_sha": "sha_main",
        "create_result": True,
        "github_owner": "test-owner",
        "repository_name": "test-repo",
    }


@patch(
    "airas.research_preparation_subgraph.nodes.fork_repository.fork_repository",
    return_value=True,
)
@patch(
    "airas.research_preparation_subgraph.nodes.check_github_repository.check_github_repository",
    return_value=True,
)
@patch(
    "airas.research_preparation_subgraph.nodes.check_branch_existence.check_branch_existence",
    return_value="sha123",
)
@patch(
    "airas.research_preparation_subgraph.nodes.retrieve_main_branch_sha.retrieve_main_branch_sha",
    return_value="sha_main",
)
@patch(
    "airas.research_preparation_subgraph.nodes.create_branch.create_branch",
    return_value=True,
)
def test_research_preparation_subgraph(
    mock_create_branch,
    mock_retrieve_main_sha,
    mock_check_branch_existence,
    mock_check_github_repo,
    mock_fork_repo,
    dummy_input,
    expected_output,
):
    subgraph = ResearchPreparationSubgraph(device_type="cpu", organization="test-org")
    graph = subgraph.build_graph()
    result = graph.invoke(dummy_input)
    assert result["repository_exists"] == expected_output["repository_exists"]
    assert result["fork_result"] == expected_output["fork_result"]
    assert result["target_branch_sha"] == expected_output["target_branch_sha"]
    assert result["main_sha"] == expected_output["main_sha"]
    assert result["create_result"] == expected_output["create_result"]
    assert result["github_owner"] == expected_output["github_owner"]
    assert result["repository_name"] == expected_output["repository_name"]
