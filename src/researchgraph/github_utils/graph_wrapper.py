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
        input_branch_name: str | None = None, 
        input_paths: dict[str, str] | None = None,
        output_branch_name: str | None = None,
        output_paths: dict[str, str] | None = None,
    ):
        self.subgraph = subgraph
        self.github_owner = github_owner
        self.repository_name = repository_name
        self.input_branch_name = input_branch_name
        self.input_paths = input_paths
        self.output_branch_name = output_branch_name
        self.output_paths = output_paths

    @time_node("wrapper", "github_input_node")
    def _github_input_node(self, state: dict[str, Any]) -> dict[str, Any]:
        return github_input_node(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=self.input_branch_name,
            input_paths=self.input_paths,
        )

    @time_node("wrapper", "github_output_node")
    def _github_output_node(self, state: dict[str, Any]) -> dict[str, bool]:
        result = github_output_node(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=self.output_branch_name,
            output_paths=self.output_paths,
            state=state,
        )
        return {"github_upload_success": result}

    def build_graph(self) -> CompiledGraph:
        wrapper = StateGraph(dict[str, Any])
        prev = START

        if self.input_paths and self.input_branch_name:
            wrapper.add_node("github_input_node", self._github_input_node)
            wrapper.add_edge(prev, "github_input_node")
            prev = "github_input_node"

        wrapper.add_node("run_subgraph", self.subgraph)
        wrapper.add_edge(prev, "run_subgraph")
        prev = "run_subgraph"

        if self.output_paths and self.output_branch_name:
            wrapper.add_node("github_output_node", self._github_output_node)
            wrapper.add_edge(prev, "github_output_node")
            prev = "github_output_node"

        wrapper.add_edge(prev, END)
        return wrapper.compile()
