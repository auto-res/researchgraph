import pytest
from unittest.mock import patch
from requests.exceptions import HTTPError, ConnectionError, Timeout
from airas.executor_subgraph.nodes.fix_code_with_devin import FixCodeWithDevinNode


@pytest.fixture
def test_environment():
    """テスト用の環境変数と入力データを設定"""
    return {
        "session_id": "devin-test-session",
        "output_text_data": "Test output",
        "error_text_data": "Test error",
        "fix_iteration_count": 1,
    }


@pytest.fixture
def fix_code_with_devin_node():
    """FixCodeWithDevinNode のインスタンスを返す"""
    return FixCodeWithDevinNode()


@pytest.mark.parametrize(
    "mock_response, expected_result",
    [
        ({"status_enum": "completed"}, {"status_enum": "completed"}),
        ({"status_enum": "running"}, {"status_enum": "running"}),
    ],
)
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.retry_request")
def test_get_devin_response_success(
    mock_retry_request,
    mock_fetch_api_data,
    fix_code_with_devin_node,
    test_environment,
    mock_response,
    expected_result,
):
    """_get_devin_response() が Devin API から正常にレスポンスを取得できるか"""
    mock_fetch_api_data.return_value = mock_response
    mock_retry_request.return_value = mock_response  # 追加

    result = fix_code_with_devin_node._get_devin_response(
        test_environment["session_id"]
    )
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize(
    "exception",
    [
        ConnectionError("Mocked Connection Error"),
        Timeout("Mocked Timeout Error"),
        HTTPError("Mocked Internal Server Error (500)"),
        HTTPError("Mocked Rate Limit Error (429)"),
    ],
)
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.retry_request")
def test_get_devin_response_api_errors(
    mock_retry_request,
    mock_fetch_api_data,
    fix_code_with_devin_node,
    test_environment,
    exception,
):
    """_get_devin_response() が API エラーを適切に処理するか"""
    mock_fetch_api_data.side_effect = exception
    mock_retry_request.return_value = None  # 追加

    result = fix_code_with_devin_node._get_devin_response(
        test_environment["session_id"]
    )
    assert result is None, f"Expected None when {exception} occurs"


@pytest.mark.parametrize(
    "mock_response",
    [
        None,
        # {},
        # {"unexpected_key": "unknown_value"},
    ],
)
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.retry_request")
def test_get_devin_response_invalid_response(
    mock_retry_request,
    mock_fetch_api_data,
    fix_code_with_devin_node,
    test_environment,
    mock_response,
):
    """_get_devin_response() が不正なレスポンスを適切に処理するか"""
    mock_fetch_api_data.return_value = mock_response
    mock_retry_request.return_value = mock_response

    result = fix_code_with_devin_node._get_devin_response(
        test_environment["session_id"]
    )
    assert result is None, f"Expected None for invalid response: {mock_response}"


@pytest.mark.parametrize(
    "mock_response, expected_result",
    [
        ({"message": "Fix request accepted"}, {"message": "Fix request accepted"}),
    ],
)
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.retry_request")
def test_request_revision_to_devin_success(
    mock_retry_request,
    mock_fetch_api_data,
    fix_code_with_devin_node,
    test_environment,
    mock_response,
    expected_result,
):
    """_request_revision_to_devin() が API に正常にリクエストを送信できるか"""
    mock_fetch_api_data.return_value = mock_response
    mock_retry_request.return_value = mock_response  # 追加

    result = fix_code_with_devin_node._request_revision_to_devin(
        test_environment["session_id"],
        test_environment["output_text_data"],
        test_environment["error_text_data"],
    )
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize(
    "exception",
    [
        ConnectionError("Mocked Connection Error"),
        Timeout("Mocked Timeout Error"),
        HTTPError("Mocked Internal Server Error (500)"),
        HTTPError("Mocked Rate Limit Error (429)"),
    ],
)
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.retry_request")
def test_request_revision_to_devin_api_errors(
    mock_retry_request,
    mock_fetch_api_data,
    fix_code_with_devin_node,
    test_environment,
    exception,
):
    """_request_revision_to_devin() が API エラーを適切に処理するか"""
    mock_fetch_api_data.side_effect = exception
    mock_retry_request.return_value = None  # 追加

    result = fix_code_with_devin_node._request_revision_to_devin(
        test_environment["session_id"],
        test_environment["output_text_data"],
        test_environment["error_text_data"],
    )
    assert result is None, f"Expected None when {exception} occurs"


@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.fetch_api_data")
@patch("researchgraph.executor_subgraph.nodes.fix_code_with_devin.retry_request")
@patch("time.sleep", return_value=None)
def test_execute(
    mock_sleep,
    mock_retry_request,
    mock_fetch_api_data,
    fix_code_with_devin_node,
    test_environment,
):
    """execute() が Devin へのリクエストを適切に行い、 fix_iteration_count を増やすかをテスト"""
    mock_fetch_api_data.side_effect = [
        {"message": "Fix request accepted"},
        {"status_enum": "completed"},
    ]
    mock_retry_request.side_effect = [
        {"message": "Fix request accepted"},
        {"status_enum": "completed"},
    ]

    result = fix_code_with_devin_node.execute(
        test_environment["session_id"],
        test_environment["output_text_data"],
        test_environment["error_text_data"],
        test_environment["fix_iteration_count"],
    )

    assert (
        result == test_environment["fix_iteration_count"] + 1
    ), "fix_iteration_count が正しくインクリメントされていない"
