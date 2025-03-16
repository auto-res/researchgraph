import pytest
import base64
import json
from unittest.mock import patch
from researchgraph.upload_subgraph.nodes.github_upload import GithubUploadNode


@pytest.fixture(scope="function")
def test_environment(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("github_upload_tests")
    pdf_file = temp_dir / "test_paper.pdf"
    pdf_file.write_bytes(b"PDF content")

    return {
        "temp_dir": temp_dir,
        "paper_content": {
            "Title": "Test Paper",
            "Abstract": "This is a test abstract.",
        },
        "pdf_file_path": str(pdf_file),
        "github_owner": "test-owner",
        "repository_name": "test-repo",
        "branch_name": "test-branch",
        "add_github_url": "https://github.com/test/add",
        "base_github_url": "https://github.com/test/base",
        "devin_url": "https://devin.test/logs",
        "all_logs": {
            "objective": "Testing execution function.",
            "base_method_text": "Baseline method description...",
            "add_method_text": "Added method description...",
            "new_method_text": ["New combined method description..."],
            "base_method_code": "def base_method(): pass",
            "add_method_code": "def add_method(): pass",
        },
    }


@pytest.fixture
def github_upload_node():
    return GithubUploadNode()


@patch("researchgraph.writer_subgraph.nodes.github_upload_node.fetch_api_data")
@patch("researchgraph.writer_subgraph.nodes.github_upload_node.retry_request")
def test_request_get_github_content(
    mock_retry_request, mock_fetch_api_data, github_upload_node, test_environment
):
    """正常系テスト: GitHub のファイル取得リクエストの動作を確認"""
    mock_fetch_api_data.return_value = {"sha": "mocked_sha"}
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = github_upload_node._request_get_github_content(
        test_environment["github_owner"],
        test_environment["repository_name"],
        test_environment["branch_name"],
        "README.md",
    )
    assert response["sha"] == "mocked_sha"


@patch("researchgraph.writer_subgraph.nodes.github_upload_node.fetch_api_data")
@patch("researchgraph.writer_subgraph.nodes.github_upload_node.retry_request")
def test_request_github_file_upload(
    mock_retry_request, mock_fetch_api_data, github_upload_node, test_environment
):
    """正常系テスト: GitHub へのファイルアップロードリクエストの動作を確認"""
    mock_fetch_api_data.return_value = {"commit": {"sha": "mocked_commit_sha"}}
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = github_upload_node._request_github_file_upload(
        test_environment["github_owner"],
        test_environment["repository_name"],
        test_environment["branch_name"],
        "logs/all_logs.json",
        "encoded_data_example",
    )
    assert "commit" in response
    assert response["commit"]["sha"] == "mocked_commit_sha"


def test_encoded_pdf_file(test_environment):
    """正常系テスト: PDF ファイルの Base64 エンコードが正しく行われるか"""
    encoded_data = GithubUploadNode._encoded_pdf_file(test_environment["pdf_file_path"])
    assert isinstance(encoded_data, str)
    assert base64.b64decode(encoded_data) == b"PDF content"


def test_encoded_markdown_data(test_environment):
    """正常系テスト: Markdown テキストの Base64 エンコードが正しく行われるか"""
    encoded_data = GithubUploadNode._encoded_markdown_data(
        test_environment["paper_content"]["Title"],
        test_environment["paper_content"]["Abstract"],
        test_environment["add_github_url"],
        test_environment["base_github_url"],
        test_environment["devin_url"],
    )
    assert isinstance(encoded_data, str)
    decoded_text = base64.b64decode(encoded_data).decode("utf-8")
    assert test_environment["paper_content"]["Title"] in decoded_text
    assert test_environment["paper_content"]["Abstract"] in decoded_text


def test_encoding_all_data(test_environment):
    """正常系テスト: all_logs の JSON エンコードが正しく行われるか"""
    test_environment["temp_dir"] = str(test_environment["temp_dir"])
    encoded_data = GithubUploadNode._encoding_all_data(test_environment)
    assert isinstance(encoded_data, str)
    decoded_json = json.loads(base64.b64decode(encoded_data).decode("utf-8"))
    assert decoded_json == test_environment


@patch(
    "researchgraph.writer_subgraph.nodes.github_upload_node.GithubUploadNode._request_get_github_content"
)
@patch(
    "researchgraph.writer_subgraph.nodes.github_upload_node.GithubUploadNode._request_github_file_upload"
)
def test_execute(
    mock_file_upload, mock_get_content, github_upload_node, test_environment
):
    """正常系テスト: execute メソッドが正しく動作することを確認"""
    mock_get_content.return_value = {"sha": "mocked_sha"}
    mock_file_upload.return_value = {"commit": {"sha": "mocked_commit_sha"}}

    result = github_upload_node.execute(
        pdf_file_path=test_environment["pdf_file_path"],
        github_owner=test_environment["github_owner"],
        repository_name=test_environment["repository_name"],
        branch_name=test_environment["branch_name"],
        title=test_environment["paper_content"]["Title"],
        abstract=test_environment["paper_content"]["Abstract"],
        add_github_url=test_environment["add_github_url"],
        base_github_url=test_environment["base_github_url"],
        devin_url=test_environment["devin_url"],
        all_logs=test_environment["all_logs"],
    )

    assert result is True
    assert mock_get_content.call_count == 1
    assert mock_file_upload.call_count == 3

    mock_file_upload.assert_any_call(
        github_owner=test_environment["github_owner"],
        repository_name=test_environment["repository_name"],
        branch_name=test_environment["branch_name"],
        encoded_data=GithubUploadNode._encoded_markdown_data(
            test_environment["paper_content"]["Title"],
            test_environment["paper_content"]["Abstract"],
            test_environment["add_github_url"],
            test_environment["base_github_url"],
            test_environment["devin_url"],
        ),
        repository_path="README.md",
        sha="mocked_sha",
    )
