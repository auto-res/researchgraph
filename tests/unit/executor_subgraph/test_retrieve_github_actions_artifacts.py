import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
from airas.executor_subgraph.nodes.retrieve_github_actions_artifacts import (
    RetrieveGithubActionsArtifactsNode,
)


@pytest.fixture(scope="function")
def test_environment():
    """テスト用の環境変数と入力データを設定"""
    return {
        "github_owner": "test-owner",
        "repository_name": "test-repo",
        "workflow_run_id": 123456789,
        "save_dir": "/tmp/test_artifacts",
        "fix_iteration_count": 1,
        "headers": {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer TEST_GITHUB_ACCESS_TOKEN",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    }


@pytest.fixture
def retrieve_github_actions_artifacts_node():
    return RetrieveGithubActionsArtifactsNode()


@patch(
    "researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts.fetch_api_data"
)
@patch(
    "researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts.retry_request"
)
def test_request_github_actions_artifacts(
    mock_retry_request,
    mock_fetch_api_data,
    retrieve_github_actions_artifacts_node,
    test_environment,
):
    """正常系テスト: GitHub Actions アーティファクト情報取得リクエストが正常に動作するか"""
    mock_fetch_api_data.return_value = {"artifacts": [{"id": 1, "name": "artifact1"}]}
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = retrieve_github_actions_artifacts_node._request_github_actions_artifacts(
        test_environment["github_owner"], test_environment["repository_name"]
    )

    assert response is not None
    assert len(response["artifacts"]) == 1
    assert response["artifacts"][0]["id"] == 1


def test_parse_artifacts_info(retrieve_github_actions_artifacts_node, test_environment):
    """正常系テスト: GitHub Actions アーティファクト情報のパースが正常に動作するか"""
    artifacts_info = {
        "artifacts": [
            {
                "name": "artifact1",
                "archive_download_url": "https://github.com/test/artifact1.zip",
                "workflow_run": {"id": 123456789},
            },
            {
                "name": "artifact2",
                "archive_download_url": "https://github.com/test/artifact2.zip",
                "workflow_run": {"id": 987654321},
            },
        ]
    }

    parsed_info = retrieve_github_actions_artifacts_node._parse_artifacts_info(
        artifacts_info, test_environment["workflow_run_id"]
    )

    assert len(parsed_info) == 1
    assert parsed_info["artifact1"] == "https://github.com/test/artifact1.zip"


@patch(
    "researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts.fetch_api_data"
)
@patch(
    "researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts.retry_request"
)
def test_request_download_artifacts(
    mock_retry_request,
    mock_fetch_api_data,
    retrieve_github_actions_artifacts_node,
    test_environment,
):
    """正常系テスト: GitHub Actions アーティファクトのダウンロードリクエストが正常に動作するか"""
    mock_fetch_api_data.return_value = b"Fake Zip File Content"
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    artifacts_redirect_url_dict = {"artifact1": "https://github.com/test/artifact1.zip"}

    with patch.object(
        retrieve_github_actions_artifacts_node, "_zip_to_txt"
    ) as mock_zip_to_txt:
        retrieve_github_actions_artifacts_node._request_download_artifacts(
            artifacts_redirect_url_dict, test_environment["save_dir"]
        )

        mock_zip_to_txt.assert_called_once()


@patch("os.remove")
@patch("os.path.exists", return_value=True)
@patch("zipfile.ZipFile")
@patch("builtins.open", new_callable=mock_open)
def test_zip_to_txt(
    mock_open_file,
    mock_zipfile,
    mock_exists,
    mock_remove,
    retrieve_github_actions_artifacts_node,
    test_environment,
):
    """正常系テスト: ZIP ファイルの展開と削除が正常に動作するか"""
    response_mock = b"Fake Zip File Content"
    iteration_save_dir = test_environment["save_dir"]
    key = "artifact1"
    zip_file_path = os.path.join(iteration_save_dir, f"{key}.zip")
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

    retrieve_github_actions_artifacts_node._zip_to_txt(
        response_mock, iteration_save_dir, key
    )

    mock_open_file.assert_called_once_with(zip_file_path, "wb")
    mock_open_file().write.assert_called_once_with(response_mock)
    mock_zip_instance.extractall.assert_called_once_with(iteration_save_dir)
    mock_remove.assert_called_once_with(zip_file_path)


@patch(
    "researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts.fetch_api_data"
)
@patch(
    "researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts.retry_request"
)
@patch("builtins.open", new_callable=mock_open, read_data="Fake output data")
@patch("os.makedirs")
def test_execute(
    mock_makedirs,
    mock_open_file,
    mock_retry_request,
    mock_fetch_api_data,
    retrieve_github_actions_artifacts_node,
    test_environment,
):
    """正常系テスト: RetrieveGithubActionsArtifactsNode の execute メソッドが正しく動作するか"""
    mock_fetch_api_data.return_value = {
        "artifacts": [
            {
                "name": "artifact1",
                "archive_download_url": "https://github.com/test/artifact1.zip",
                "workflow_run": {"id": 123456789},
            }
        ]
    }
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    with patch.object(
        retrieve_github_actions_artifacts_node, "_request_download_artifacts"
    ) as mock_download:
        mock_download.return_value = None

        output_text_data, error_text_data = (
            retrieve_github_actions_artifacts_node.execute(
                test_environment["github_owner"],
                test_environment["repository_name"],
                test_environment["workflow_run_id"],
                test_environment["save_dir"],
                test_environment["fix_iteration_count"],
            )
        )
        assert output_text_data == "Fake output data"
        assert error_text_data == "Fake output data"

        mock_makedirs.assert_called_once()
        mock_open_file.assert_any_call(
            os.path.join(test_environment["save_dir"] + "/iteration_1", "output.txt"),
            "r",
        )
        mock_open_file.assert_any_call(
            os.path.join(test_environment["save_dir"] + "/iteration_1", "error.txt"),
            "r",
        )
        mock_download.assert_called_once()
