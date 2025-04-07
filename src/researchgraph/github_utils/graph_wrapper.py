from typing import Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from researchgraph.github_utils.github_file_io import github_input_node, github_output_node
from researchgraph.utils.execution_timers import time_node

class GraphWrapper:
    def __init__(
        self,
        subgraph: CompiledGraph,
        github_owner: str,
        repository_name: str,
        branch_name: str,
        input_paths: dict[str, str],
        output_paths: dict[str, str],
    ):
        self.subgraph = subgraph
        self.github_owner = github_owner
        self.repository_name = repository_name
        self.branch_name = branch_name
        self.input_paths = input_paths
        self.output_paths = output_paths

    @time_node("wrapper", "github_input_node")
    def _github_input_node(self, state: dict) -> dict[str, Any]:
        return github_input_node(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=self.branch_name,
            input_paths=self.input_paths,
        )

    @time_node("wrapper", "github_output_node")
    def _github_output_node(self, state: dict) -> dict[str, bool]:
        result = github_output_node(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=self.branch_name,
            output_paths=self.output_paths,
            state=state,
        )
        return {"github_upload_success": result}

    def build_graph(self) -> CompiledGraph:
        wrapper = StateGraph(dict)
        wrapper.add_node("github_input_node", self._github_input_node)
        wrapper.add_node("run_subgraph", self.subgraph)
        wrapper.add_node("github_output_node", self._github_output_node)
        
        wrapper.add_edge(START, "github_input_node")
        wrapper.add_edge("github_input_node", "run_subgraph")
        wrapper.add_edge("run_subgraph", "github_output_node")
        wrapper.add_edge("github_output_node", END)

        return wrapper.compile()
