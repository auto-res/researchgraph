import pytest
import json
from unittest.mock import patch
from requests.exceptions import HTTPError
from airas.executor_subgraph.nodes.llm_decide import llm_decide


@pytest.fixture
def test_environment():
    """テスト用の環境変数と入力データを設定"""
    return {
        "llm_name": "gpt-4o-mini-2024-07-18",
        "output_text_data": "No error",
        "error_text_data": "",
    }


@pytest.mark.parametrize(
    "mock_response_content, expected_result",
    [
        ({"judgment_result": True}, True),
        ({"judgment_result": False}, False),
    ],
)
@patch("airas.executor_subgraph.nodes.llm_decide.openai_client")
def test_llm_decide_success(
    mock_completion, test_environment, mock_response_content, expected_result
):
    """llm_decide() が LLM からのレスポンスを正しく処理するかをテスト"""
    mock_completion.return_value = json.dumps(mock_response_content)

    result = llm_decide(
        test_environment["llm_name"],
        test_environment["output_text_data"],
        test_environment["error_text_data"],
    )
    assert result == expected_result


@pytest.mark.parametrize(
    "exception",
    [
        ConnectionError("Mocked Connection Error"),
        TimeoutError("Mocked Timeout Error"),
        HTTPError("Mocked Internal Server Error (500)"),
        HTTPError("Mocked Rate Limit Error (429)"),
    ],
)
@patch("airas.executor_subgraph.nodes.llm_decide.openai_client")
def test_llm_decide_api_errors(mock_completion, test_environment, exception):
    """llm_decide() が API 呼び出し時の異常を適切に処理するかをテスト"""
    mock_completion.side_effect = exception

    result = llm_decide(
        test_environment["llm_name"],
        test_environment["output_text_data"],
        test_environment["error_text_data"],
    )
    assert result is None, f"Expected None when {exception} occurs"


@pytest.mark.parametrize(
    "mock_response_content",
    [
        "Invalid response format",
        json.dumps({"wrong_key": True}),
    ],
)
@patch("airas.executor_subgraph.nodes.llm_decide.openai_client")
def test_llm_decide_invalid_response(
    mock_completion, test_environment, mock_response_content
):
    """llm_decide() が不正なレスポンスを適切に処理するかをテスト"""
    mock_completion.return_value = json.dumps(mock_response_content)

    result = llm_decide(
        test_environment["llm_name"],
        test_environment["output_text_data"],
        test_environment["error_text_data"],
    )
    assert (
        result is None
    ), f"Expected None for invalid response: {mock_response_content}"
