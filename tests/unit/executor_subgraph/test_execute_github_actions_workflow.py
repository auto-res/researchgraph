import pytest
from unittest.mock import patch
from airas.executor_subgraph.nodes.execute_github_actions_workflow import (
    ExecuteGithubActionsWorkflowNode,
)


@pytest.fixture(scope="function")
def test_environment():
    """テスト用の環境変数と入力データを設定"""
    return {
        "github_owner": "test-owner",
        "repository_name": "test-repo",
        "branch_name": "test-branch",
        "headers": {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer TEST_GITHUB_ACCESS_TOKEN",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    }


@pytest.fixture
def execute_github_actions_workflow_node():
    return ExecuteGithubActionsWorkflowNode()


@patch(
    "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.fetch_api_data"
)
@patch(
    "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.retry_request"
)
def test_request_github_actions_workflow_execution(
    mock_retry_request,
    mock_fetch_api_data,
    execute_github_actions_workflow_node,
    test_environment,
):
    """正常系テスト: GitHub Actions ワークフロー実行リクエストが正常に動作するか"""
    mock_fetch_api_data.return_value = {"status": "success"}
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = (
        execute_github_actions_workflow_node._request_github_actions_workflow_execution(
            test_environment["github_owner"],
            test_environment["repository_name"],
            test_environment["branch_name"],
        )
    )

    assert response is not None
    assert response["status"] == "success"


@patch(
    "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.fetch_api_data"
)
@patch(
    "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.retry_request"
)
def test_request_github_actions_workflow_info_before_execution(
    mock_retry_request,
    mock_fetch_api_data,
    execute_github_actions_workflow_node,
    test_environment,
):
    """正常系テスト: 実行前の GitHub Actions ワークフロー情報取得が正常に動作するか"""
    mock_fetch_api_data.return_value = {
        "workflow_runs": [{"id": 1, "status": "completed"}]
    }
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = execute_github_actions_workflow_node._request_github_actions_workflow_info_before_execution(
        test_environment["github_owner"],
        test_environment["repository_name"],
        test_environment["branch_name"],
    )

    assert response is not None
    assert len(response["workflow_runs"]) == 1
    assert response["workflow_runs"][0]["id"] == 1


@patch(
    "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.fetch_api_data"
)
@patch(
    "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.retry_request"
)
def test_request_github_actions_workflow_info_after_execution(
    mock_retry_request,
    mock_fetch_api_data,
    execute_github_actions_workflow_node,
    test_environment,
):
    """正常系テスト: 実行後の GitHub Actions ワークフロー情報取得が正常に動作するか"""
    mock_fetch_api_data.return_value = {
        "workflow_runs": [
            {"id": 1, "status": "completed"},
            {"id": 2, "status": "in_progress"},
        ]
    }
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    response = execute_github_actions_workflow_node._request_github_actions_workflow_info_after_execution(
        test_environment["github_owner"],
        test_environment["repository_name"],
        test_environment["branch_name"],
        num_workflow_runs_before_execution=1,
    )

    assert response is not None
    assert len(response["workflow_runs"]) == 2
    assert response["workflow_runs"][1]["id"] == 2


@patch(
    "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.fetch_api_data"
)
@patch(
    "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.retry_request"
)
def test_execute(
    mock_retry_request,
    mock_fetch_api_data,
    execute_github_actions_workflow_node,
    test_environment,
):
    """正常系テスト: ExecuteGithubActionsWorkflowNode の execute メソッドが正しく動作するか"""
    mock_fetch_api_data.return_value = {
        "workflow_runs": [
            {"id": 1, "status": "completed", "created_at": "2024-02-12T12:34:56Z"}
        ]
    }
    mock_retry_request.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    with patch.object(
        execute_github_actions_workflow_node,
        "_request_github_actions_workflow_execution",
    ) as mock_execute_workflow:
        mock_execute_workflow.return_value = {"status": "success"}

        with patch.object(
            execute_github_actions_workflow_node,
            "_request_github_actions_workflow_info_after_execution",
        ) as mock_info_after_execution:
            mock_info_after_execution.return_value = {
                "workflow_runs": [
                    {
                        "id": 1,
                        "status": "completed",
                        "created_at": "2024-02-12T12:34:56Z",
                    },
                    {
                        "id": 2,
                        "status": "completed",
                        "created_at": "2024-02-12T13:00:00Z",
                    },
                ]
            }
            workflow_run_id = execute_github_actions_workflow_node.execute(
                test_environment["github_owner"],
                test_environment["repository_name"],
                test_environment["branch_name"],
            )
            assert workflow_run_id == 2

            mock_execute_workflow.assert_called_once_with(
                test_environment["github_owner"],
                test_environment["repository_name"],
                test_environment["branch_name"],
            )
            mock_info_after_execution.assert_called_once_with(
                test_environment["github_owner"],
                test_environment["repository_name"],
                test_environment["branch_name"],
                1,
            )
