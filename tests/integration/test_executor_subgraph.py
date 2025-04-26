import pytest
from unittest.mock import patch
from airas.executor_subgraph.executor_subgraph import ExecutorSubgraph, ExecutorState


@pytest.fixture(scope="function")
def test_state():
    """テスト用の ExecutorState を作成"""
    return ExecutorState(
        new_detailed_description_of_methodology="Mock methodology",
        new_novelty="Mock novelty",
        new_experimental_procedure="Mock experiment",
        new_method_code="Mock method code",
        branch_name="mock_branch",
        github_owner="mock_owner",
        repository_name="mock_repo",
        workflow_run_id=123456,
        save_dir="/mock/path/to/save",
        fix_iteration_count=0,
        session_id="mock_session",
        output_text_data="Mock output",
        error_text_data="Mock error",
        devin_url="https://devin.ai/mock",
        judgment_result=False,
    )


@pytest.fixture
def mock_nodes():
    """各ノードのモックを作成"""
    mocks = {}
    fix_iteration_counter = [0]

    def fix_code_side_effect(
        session_id, output_text_data, error_text_data, fix_iteration_count
    ):
        fix_iteration_counter[0] += 1
        print(
            f"Mocked fix_code_with_devin returning fix_iteration_count={fix_iteration_counter[0]}"
        )
        return fix_iteration_counter[0]

    with (
        patch(
            "researchgraph.executor_subgraph.nodes.generate_code_with_devin.GenerateCodeWithDevinNode.execute",
            return_value=("mock_session", "mock_branch", "https://devin.ai/mock"),
        ) as mock_generate,
        patch(
            "researchgraph.executor_subgraph.nodes.execute_github_actions_workflow.ExecuteGithubActionsWorkflowNode.execute",
            return_value=123456,
        ) as mock_execute_workflow,
        patch(
            "researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts.RetrieveGithubActionsArtifactsNode.execute",
            return_value=("Mock output", "Mock error"),
        ) as mock_retrieve_artifacts,
        patch(
            "researchgraph.executor_subgraph.executor_subgraph.llm_decide",
            return_value=False,
        ) as mock_llm_decide,
        patch(
            "researchgraph.executor_subgraph.nodes.fix_code_with_devin.FixCodeWithDevinNode.execute",
            side_effect=fix_code_side_effect,
        ) as mock_fix_code,
    ):
        mocks["generate_code_with_devin"] = mock_generate
        mocks["execute_github_actions_workflow"] = mock_execute_workflow
        mocks["retrieve_github_actions_artifacts"] = mock_retrieve_artifacts
        mocks["llm_decide"] = mock_llm_decide
        mocks["fix_code_with_devin"] = mock_fix_code

        yield mocks


@pytest.fixture
def executor_subgraph():
    return ExecutorSubgraph(
        max_fix_iteration=3,
    ).build_graph()


def test_executor_subgraph(mock_nodes, test_state, executor_subgraph):
    """ExecutorSubgraph の統合テスト"""
    result = executor_subgraph.invoke(test_state)

    mock_nodes["generate_code_with_devin"].assert_called_once()
    mock_nodes["execute_github_actions_workflow"].assert_called()
    mock_nodes["retrieve_github_actions_artifacts"].assert_called()
    mock_nodes["llm_decide"].assert_called()

    assert result["fix_iteration_count"] <= 3

    expected_calls = 1 + min(3, result["fix_iteration_count"])
    assert mock_nodes["execute_github_actions_workflow"].call_count == expected_calls


def test_generate_code_with_devin_node(mock_nodes, test_state, executor_subgraph):
    """LangGraphを通じた GenerateCodeWithDevinNode の統合テスト"""
    result = executor_subgraph.invoke(test_state)
    mock_nodes["generate_code_with_devin"].assert_called_once()

    assert result["session_id"] == "mock_session"
    assert result["branch_name"] == "mock_branch"
    assert result["devin_url"] == "https://devin.ai/mock"


def test_execute_github_actions_workflow_node(
    mock_nodes, test_state, executor_subgraph
):
    """LangGraphを通じた ExecuteGithubActionsWorkflowNode の統合テスト"""
    result = executor_subgraph.invoke(test_state)
    mock_nodes["execute_github_actions_workflow"].assert_called()

    assert result["workflow_run_id"] == 123456


def test_retrieve_github_actions_artifacts_node(
    mock_nodes, test_state, executor_subgraph
):
    """LangGraphを通じた RetrieveGithubActionsArtifactsNode の統合テスト"""
    result = executor_subgraph.invoke(test_state)
    mock_nodes["retrieve_github_actions_artifacts"].assert_called()

    assert result["output_text_data"] == "Mock output"
    assert result["error_text_data"] == "Mock error"


def test_llm_decide_node(mock_nodes, test_state, executor_subgraph):
    """LangGraphを通じた llm_decide_node の統合テスト"""
    result = executor_subgraph.invoke(test_state)
    mock_nodes["llm_decide"].assert_called()

    assert result["judgment_result"] is False


def test_fix_code_with_devin_node(mock_nodes, test_state, executor_subgraph):
    """LangGraphを通じた FixCodeWithDevinNode の統合テスト"""
    test_state["judgment_result"] = False
    result = executor_subgraph.invoke(test_state)

    mock_nodes["fix_code_with_devin"].assert_called()
    assert result["fix_iteration_count"] == 3


def test_iteration_function():
    """iteration_function() のテスト"""
    subgraph = ExecutorSubgraph(max_fix_iteration=3)

    state_pass = {"judgment_result": True, "fix_iteration_count": 0}
    assert subgraph.iteration_function(state_pass) == "finish"

    state_retry = {"judgment_result": False, "fix_iteration_count": 1}
    assert subgraph.iteration_function(state_retry) == "correction"

    state_max_retry = {"judgment_result": False, "fix_iteration_count": 3}
    assert subgraph.iteration_function(state_max_retry) == "finish"
