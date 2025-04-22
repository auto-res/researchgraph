
import pytest
from unittest.mock import patch, MagicMock
from researchgraph.executor_subgraph.nodes.generate_code_with_devin import GenerateCodeWithDevinNode


@pytest.fixture(scope="function")
def test_environment():
    """ テスト用の環境変数と入力データを設定 """
    return {
        "github_owner": "test-owner",
        "repository_name": "test-repo",
        "repository_url": "https://github.com/test-owner/test-repo",
        "new_detailed_description_of_methodology": "This is a test methodology.",
        "new_novelty": "This is a test novelty.",
        "new_experimental_procedure": "This is a test experimental procedure.",
        "new_method_code": "print('Hello, world!')",
        "headers": {
            "Authorization": "Bearer TEST_DEVIN_API_KEY",
            "Content-Type": "application/json",
        }
    }


@pytest.fixture
def generate_code_with_devin_node():
    return GenerateCodeWithDevinNode()


@patch("researchgraph.executor_subgraph.nodes.generate_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.generate_code_with_devin.retry_request")
def test_request_create_session(mock_retry_request, mock_fetch_api_data, generate_code_with_devin_node, test_environment):
    """ 正常系テスト: Devin のセッション作成リクエストが正常に動作するか """
    mock_fetch_api_data.return_value = {
        "session_id": "devin-test-session",
        "url": "https://devin.test/sessions/devin-test-session",
    }
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = generate_code_with_devin_node._request_create_session(
        test_environment["repository_url"],
        test_environment["new_detailed_description_of_methodology"],
        test_environment["new_novelty"],
        test_environment["new_experimental_procedure"],
        test_environment["new_method_code"],
    )
    assert response is not None
    assert response["session_id"] == "devin-test-session"
    assert response["url"] == "https://devin.test/sessions/devin-test-session"


@patch("researchgraph.executor_subgraph.nodes.generate_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.generate_code_with_devin.retry_request")
def test_request_devin_output(mock_retry_request, mock_fetch_api_data, generate_code_with_devin_node, test_environment):
    """ 正常系テスト: Devin の実行結果取得リクエストが正常に動作するか """
    mock_fetch_api_data.return_value = {
        "status_enum": "completed",
        "structured_output": {"branch_name": "devin-test-branch"},
    }
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    session_id = "devin-test-session"
    response = generate_code_with_devin_node._request_devin_output(session_id)

    assert response is not None
    assert response["status_enum"] == "completed"
    assert response["structured_output"]["branch_name"] == "devin-test-branch"


@patch("researchgraph.executor_subgraph.nodes.generate_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.generate_code_with_devin.retry_request")
@patch("time.sleep", return_value=None) 
def test_execute(mock_time_sleep, mock_retry_request, mock_fetch_api_data, generate_code_with_devin_node, test_environment):
    """ 正常系テスト: GenerateCodeWithDevinNode の execute メソッドが正しく動作するか """
    mock_fetch_api_data.return_value = {
        "session_id": "devin-test-session",
        "url": "https://devin.test/sessions/devin-test-session",
    }
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    session_id, branch_name, devin_url = generate_code_with_devin_node.execute(
        test_environment["github_owner"],
        test_environment["repository_name"],
        test_environment["new_detailed_description_of_methodology"],
        test_environment["new_novelty"],
        test_environment["new_experimental_procedure"],
        test_environment["new_method_code"],
    )
    
    assert session_id == "devin-test-session"
    assert branch_name == "devin-test-session"
    assert devin_url == "https://devin.test/sessions/devin-test-session"
    mock_time_sleep.assert_called_once_with(120)