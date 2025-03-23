import os
import time
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.executor_subgraph.nodes.generate_code_with_devin import (
    generate_code_with_devin,
)
from researchgraph.executor_subgraph.nodes.execute_github_actions_workflow import (
    execute_github_actions_workflow,
)
from researchgraph.executor_subgraph.nodes.retrieve_github_actions_artifacts import (
    retrieve_github_actions_artifacts,
)
from researchgraph.executor_subgraph.nodes.fix_code_with_devin import (
    fix_code_with_devin,
)
from researchgraph.executor_subgraph.nodes.check_devin_completion import (
    check_devin_completion,
)
from researchgraph.executor_subgraph.nodes.llm_decide import llm_decide

from researchgraph.executor_subgraph.input_data import (
    executor_subgraph_input_data,
)

DEVIN_API_KEY = os.getenv("DEVIN_API_KEY")


class ExecutorSubgraphInputState(TypedDict):
    new_method: str
    experiment_code: str


class ExecutorSubgraphHiddenState(TypedDict):
    experiment_session_id: str
    devin_completion: bool
    fix_iteration_count: int
    error_text_data: str
    judgment_result: bool
    workflow_run_id: int


class ExecutorSubgraphOutputState(TypedDict):
    experiment_devin_url: str
    branch_name: str
    output_text_data: str


class ExecutorSubgraphState(
    ExecutorSubgraphInputState, ExecutorSubgraphHiddenState, ExecutorSubgraphOutputState
):
    pass


class ExecutorSubgraph:
    def __init__(
        self,
        github_owner: str,
        repository_name: str,
        save_dir: str,
        max_code_fix_iteration: int = 3,
    ):
        self.github_owner = github_owner
        self.repository_name = repository_name
        self.save_dir = save_dir
        self.max_code_fix_iteration = max_code_fix_iteration
        self.headers = {
            "Authorization": f"Bearer {DEVIN_API_KEY}",
            "Content-Type": "application/json",
        }

    def _generate_code_with_devin_node(self, state: ExecutorSubgraphState) -> dict:
        print("---ExecutorSubgraph---")
        print("generate_code_with_devin_node")
        experiment_session_id, experiment_devin_url = generate_code_with_devin(
            headers=self.headers,
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            new_method=state["new_method"],
            experiment_code=state["experiment_code"],
        )

        return {
            "fix_iteration_count": 0,
            "experiment_session_id": experiment_session_id,
            "branch_name": experiment_session_id,
            "experiment_devin_url": experiment_devin_url,
        }

    def _check_devin_completion_node(self, state: ExecutorSubgraphState) -> dict:
        time.sleep(120)
        print("check_devin_completion_node")
        check_devin_completion(
            headers=self.headers,
            session_id=state["experiment_session_id"],
        )
        return {
            "devin_completion": True,
        }

    def _execute_github_actions_workflow_node(
        self, state: ExecutorSubgraphState
    ) -> dict:
        print("execute_github_actions_workflow_node")
        workflow_run_id = execute_github_actions_workflow(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=state["branch_name"],
        )
        return {
            "workflow_run_id": workflow_run_id,
        }

    def _retrieve_github_actions_artifacts_node(
        self, state: ExecutorSubgraphState
    ) -> dict:
        print("retrieve_github_actions_artifacts_node")
        output_text_data, error_text_data = retrieve_github_actions_artifacts(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            workflow_run_id=state["workflow_run_id"],
            save_dir=self.save_dir,
            fix_iteration_count=state["fix_iteration_count"],
        )
        return {
            "output_text_data": output_text_data,
            "error_text_data": error_text_data,
        }

    def _llm_decide_node(self, state: ExecutorSubgraphState) -> dict:
        print("llm_decide_node")
        judgment_result = llm_decide(
            llm_name="gpt-4o-mini-2024-07-18",
            output_text_data=state["output_text_data"],
            error_text_data=state["error_text_data"],
        )
        return {
            "judgment_result": judgment_result,
        }

    def _fix_code_with_devin_node(self, state: ExecutorSubgraphState) -> dict:
        print("fix_code_with_devin_node")
        fix_iteration_count = fix_code_with_devin(
            headers=self.headers,
            session_id=state["experiment_session_id"],
            output_text_data=state["output_text_data"],
            error_text_data=state["error_text_data"],
            fix_iteration_count=state["fix_iteration_count"],
        )
        return {
            "fix_iteration_count": fix_iteration_count,
        }

    def iteration_function(self, state: ExecutorSubgraphState):
        if state["judgment_result"] is True:
            return "finish"
        else:
            if state["fix_iteration_count"] < self.max_code_fix_iteration:
                return "correction"
            else:
                return "finish"

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(ExecutorSubgraphState)
        # make nodes
        graph_builder.add_node(
            "generate_code_with_devin_node", self._generate_code_with_devin_node
        )
        graph_builder.add_node(
            "check_devin_completion_node", self._check_devin_completion_node
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
            "generate_code_with_devin_node", "check_devin_completion_node"
        )
        graph_builder.add_edge(
            "check_devin_completion_node",
            "execute_github_actions_workflow_node",
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
            "fix_code_with_devin_node", "check_devin_completion_node"
        )
        graph_builder.add_edge(
            "check_devin_completion_node", "execute_github_actions_workflow_node"
        )
        return graph_builder.compile()


if __name__ == "__main__":
    graph = ExecutorSubgraph(
        github_owner="auto-res2",
        repository_name="auto-research",
        save_dir="/workspaces/researchgraph/data",
        max_code_fix_iteration=3,
    ).build_graph()

    for event in graph.stream(executor_subgraph_input_data, stream_mode="updates"):
        # print(node)
        node_name = list(event.keys())[0]
        print(node_name)
        print(event[node_name])

    # executor_subgraph.output_mermaid
    # result = graph.invoke(executor_subgraph_input_data)
