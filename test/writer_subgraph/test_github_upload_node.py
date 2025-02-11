import os
import pytest
import base64
import json
from unittest.mock import patch, MagicMock
from typing import TypedDict
from langgraph.graph import StateGraph
from researchgraph.writer_subgraph.nodes.github_upload_node import GithubUploadNode

class State(TypedDict):
    paper_content: dict
    pdf_file_path: str
    github_owner: str
    repository_name: str
    branch_name: str
    add_github_url: str
    base_github_url: str
    devin_url: str
    completion: bool

@pytest.fixture(scope="function")
def test_environment(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("github_upload_tests")
    pdf_file = temp_dir / "test_paper.pdf"
    pdf_file.write_bytes(b"PDF content")

    state = {
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
    }
    return {"temp_dir": temp_dir, **state}

@pytest.fixture
def github_upload_node():
    return GithubUploadNode()

@patch("researchgraph.writer_subgraph.nodes.github_upload_node.fetch_api_data")
@patch("researchgraph.writer_subgraph.nodes.github_upload_node.retry_request")
def test_request_get_github_content(mock_retry_request, mock_fetch_api_data, github_upload_node, test_environment):
    """ 正常系テスト: GitHub のファイル取得リクエストの動作を確認 """
    mock_fetch_api_data.return_value = {"sha": "mocked_sha"}
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = github_upload_node._request_get_github_content(
        test_environment["github_owner"],
        test_environment["repository_name"],
        test_environment["branch_name"],
        "README.md"
    )
    assert response["sha"] == "mocked_sha"

@patch("researchgraph.writer_subgraph.nodes.github_upload_node.fetch_api_data")
@patch("researchgraph.writer_subgraph.nodes.github_upload_node.retry_request")
def test_request_github_file_upload(mock_retry_request, mock_fetch_api_data, github_upload_node, test_environment):
    """ 正常系テスト: GitHub へのファイルアップロードリクエストの動作を確認 """
    mock_fetch_api_data.return_value = {"commit": {"sha": "mocked_commit_sha"}}
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = github_upload_node._request_github_file_upload(
        test_environment["github_owner"],
        test_environment["repository_name"],
        test_environment["branch_name"],
        "logs/all_logs.json",
        "encoded_data_example"
    )
    assert "commit" in response
    assert response["commit"]["sha"] == "mocked_commit_sha"

def test_encoded_pdf_file(test_environment):
    """ 正常系テスト: PDF ファイルの Base64 エンコードが正しく行われるか """
    encoded_data = GithubUploadNode._encoded_pdf_file(test_environment["pdf_file_path"])
    assert isinstance(encoded_data, str)
    assert base64.b64decode(encoded_data) == b"PDF content"

def test_encoded_markdown_data(test_environment):
    """ 正常系テスト: Markdown テキストの Base64 エンコードが正しく行われるか """
    encoded_data = GithubUploadNode._encoded_markdown_data(
        test_environment["paper_content"]["Title"],
        test_environment["paper_content"]["Abstract"],
        test_environment["add_github_url"],
        test_environment["base_github_url"],
        test_environment["devin_url"]
    )
    assert isinstance(encoded_data, str)
    decoded_text = base64.b64decode(encoded_data).decode("utf-8")
    assert test_environment["paper_content"]["Title"] in decoded_text
    assert test_environment["paper_content"]["Abstract"] in decoded_text

def test_encoding_all_data(test_environment):
    """ 正常系テスト: all_logs の JSON エンコードが正しく行われるか """
    test_environment["temp_dir"] = str(test_environment["temp_dir"]) 
    encoded_data = GithubUploadNode._encoding_all_data(test_environment)
    assert isinstance(encoded_data, str)
    decoded_json = json.loads(base64.b64decode(encoded_data).decode("utf-8"))
    assert decoded_json == test_environment

@patch("researchgraph.writer_subgraph.nodes.github_upload_node.fetch_api_data")
@patch("researchgraph.writer_subgraph.nodes.github_upload_node.retry_request")
def test_github_upload_node_execution(mock_retry_request, mock_fetch_api_data, github_upload_node, test_environment):
    """ 正常系テスト: GithubUploadNode が正しく実行されるか """
    mock_fetch_api_data.return_value = {"sha": "mocked_commit_sha"}
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
    
    def github_upload_node_callable(state):
        return {"completion": github_upload_node.execute(
            pdf_file_path=state["pdf_file_path"],
            github_owner=state["github_owner"],
            repository_name=state["repository_name"],
            branch_name=state["branch_name"],
            title=state["paper_content"]["Title"],
            abstract=state["paper_content"]["Abstract"],
            add_github_url=state["add_github_url"],
            base_github_url=state["base_github_url"],
            devin_url=state["devin_url"],
            all_logs=state,
        )}

    graph_builder = StateGraph(State)
    graph_builder.add_node("github_upload_node", github_upload_node_callable)
    graph_builder.set_entry_point("github_upload_node")
    graph_builder.set_finish_point("github_upload_node")
    graph = graph_builder.compile()

    result_state = graph.invoke(test_environment, debug=True)

    assert result_state is not None
    assert result_state["completion"] is True