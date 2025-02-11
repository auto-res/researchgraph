from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.executor_subgraph.nodes.generate_code_with_devin import (
    GenerateCodeWithDevinNode,
)
from researchgraph.executor_subgraph.nodes.execute_github_actions_workflow import (
    ExecuteGithubActionsWorkflowNode,
)
from researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts import (
    RetrieveGithubActionsArtifactsNode,
)
from researchgraph.executor_subgraph.nodes.fix_code_with_devin import (
    FixCodeWithDevinNode,
)
from researchgraph.executor_subgraph.nodes.llm_decide import llm_decide

from researchgraph.executor_subgraph.input_data import (
    executor_subgraph_input_data,
)


class ExecutorState(TypedDict):
    new_detailed_description_of_methodology: str
    new_novelty: str
    new_experimental_procedure: str
    new_method_code: str

    branch_name: str
    github_owner: str
    repository_name: str
    workflow_run_id: int
    save_dir: str
    fix_iteration_count: int
    session_id: str
    output_text_data: str
    error_text_data: str
    devin_url: str
    judgment_result: bool


class ExecutorSubgraph:
    def __init__(
        self,
        max_fix_iteration: int = 3,
    ):
        self.max_fix_iteration = max_fix_iteration

    def _generate_code_with_devin_node(self, state: ExecutorState) -> dict:
        print("---ExecutorSubgraph---")

        print("generate_code_with_devin_node")
        session_id, branch_name, devin_url = GenerateCodeWithDevinNode().execute(
            github_owner=state["github_owner"],
            repository_name=state["repository_name"],
            new_detailed_description_of_methodology=state[
                "new_detailed_description_of_methodology"
            ],
            new_novelty=state["new_novelty"],
            new_experimental_procedure=state["new_experimental_procedure"],
            new_method_code=state["new_method_code"],
        )

        return {
            "session_id": session_id,
            "branch_name": branch_name,
            "devin_url": devin_url,
        }

    def _execute_github_actions_workflow_node(self, state: ExecutorState) -> dict:
        print("execute_github_actions_workflow_node")
        github_owner = state["github_owner"]
        repository_name = state["repository_name"]
        branch_name = state["branch_name"]
        workflow_run_id = ExecuteGithubActionsWorkflowNode().execute(
            github_owner=github_owner,
            repository_name=repository_name,
            branch_name=branch_name,
        )
        return {
            "workflow_run_id": workflow_run_id,
        }

    def _retrieve_github_actions_artifacts_node(self, state: ExecutorState) -> dict:
        print("retrieve_github_actions_artifacts_node")
        github_owner = state["github_owner"]
        repository_name = state["repository_name"]
        workflow_run_id = state["workflow_run_id"]
        save_dir = state["save_dir"]
        fix_iteration_count = state["fix_iteration_count"]
        output_text_data, error_text_data = (
            RetrieveGithubActionsArtifactsNode().execute(github_owner=github_owner, repository_name=repository_name, workflow_run_id=workflow_run_id, save_dir=save_dir, fix_iteration_count=fix_iteration_count)
        )
        return {
            "output_text_data": output_text_data,
            "error_text_data": error_text_data,
        }

    def _llm_decide_node(self, state: ExecutorState) -> dict:
        output_text_data = state["output_text_data"]
        error_text_data = state["error_text_data"]
        judgment_result = llm_decide(
            llm_name="gpt-4o-mini-2024-07-18",
            output_text_data=output_text_data,
            error_text_data=error_text_data,
        )
        return {
            "judgment_result": judgment_result,
        }

    def _fix_code_with_devin_node(self, state: ExecutorState) -> dict:
        print("fix_code_with_devin_node")
        session_id = state["session_id"]
        output_text_data = state["output_text_data"]
        error_text_data = state["error_text_data"]
        fix_iteration_count = state["fix_iteration_count"]
        fix_iteration_count = FixCodeWithDevinNode().execute(
            session_id=session_id,
            output_text_data=output_text_data,
            error_text_data=error_text_data,
            fix_iteration_count=fix_iteration_count,
        )
        return {
            "fix_iteration_count": fix_iteration_count,
        }

    def iteration_function(self, state: ExecutorState):
        if state["judgment_result"] is True:
            return "finish"
        else:
            if state["fix_iteration_count"] <= self.max_fix_iteration:
                return "correction"
            else:
                return "finish"

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(ExecutorState)
        # make nodes
        graph_builder.add_node(
            "generate_code_with_devin_node", self._generate_code_with_devin_node
        )
        graph_builder.add_node(
            "execute_github_actions_workflow_node",
            self._execute_github_actions_workflow_node,
        )
        graph_builder.add_node(
            "retrieve_github_actions_artifacts_node",
            self._retrieve_github_actions_artifacts_node,
        )
        graph_builder.add_node("llm_decide_node", self._llm_decide_node)
        graph_builder.add_node(
            "fix_code_with_devin_node", self._fix_code_with_devin_node
        )

        # make edges
        graph_builder.add_edge(START, "generate_code_with_devin_node")
        graph_builder.add_edge(
            "generate_code_with_devin_node", "execute_github_actions_workflow_node"
        )
        graph_builder.add_edge(
            "execute_github_actions_workflow_node",
            "retrieve_github_actions_artifacts_node",
        )
        graph_builder.add_edge(
            "retrieve_github_actions_artifacts_node", "llm_decide_node"
        )
        graph_builder.add_conditional_edges(
            "llm_decide_node",
            self.iteration_function,
            {
                "correction": "fix_code_with_devin_node",
                "finish": END,
            },
        )
        graph_builder.add_edge(
            "fix_code_with_devin_node", "execute_github_actions_workflow_node"
        )
        return graph_builder.compile()


if __name__ == "__main__":
    graph = ExecutorSubgraph(
        max_fix_iteration=3,
    ).build_graph()

    # executor_subgraph.output_mermaid
    result = graph.invoke(executor_subgraph_input_data)
