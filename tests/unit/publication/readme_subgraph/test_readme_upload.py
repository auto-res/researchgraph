import pytest
from unittest.mock import patch
from airas.readme_subgraph.nodes.readme_upload import readme_upload


@pytest.fixture
def sample_inputs():
    return {
        "github_owner": "test-owner",
        "repository_name": "test-repo",
        "branch_name": "main",
        "title": "Test Title",
        "abstract": "Test Abstract",
        "devin_url": "https://example.com/devin",
    }


# Normal case test: readme_upload returns True when all API calls succeed.
@patch(
    "airas.readme_subgraph.nodes.readme_upload._request_github_file_upload",
    return_value=None,
)
@patch(
    "airas.readme_subgraph.nodes.readme_upload._request_get_github_content",
    return_value={"sha": "dummy-sha"},
)
def test_readme_upload_success(mock_get_content, mock_file_upload, sample_inputs):
    result = readme_upload(**sample_inputs)
    assert result is True


# Abnormal case test: readme_upload still returns True even if README does not exist (sha is None).
@patch(
    "airas.readme_subgraph.nodes.readme_upload._request_github_file_upload",
    return_value=None,
)
@patch(
    "airas.readme_subgraph.nodes.readme_upload._request_get_github_content",
    return_value=None,
)
def test_readme_upload_no_readme(mock_get_content, mock_file_upload, sample_inputs):
    result = readme_upload(**sample_inputs)
    assert result is True
