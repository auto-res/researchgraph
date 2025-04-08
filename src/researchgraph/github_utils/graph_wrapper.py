from typing import Any, Protocol, TypeVar, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from researchgraph.github_utils.github_file_io import download_from_github, upload_to_github
from researchgraph.utils.execution_timers import time_node


# class GraphWrapperState(TypedDict):
#     github_upload_success: bool


class GraphWrapper:
    def __init__(
        self,
        subgraph: CompiledGraph,
        github_owner: str,
        repository_name: str,
        input_branch_name: str | None = None, 
        input_path: str | None = None,
        output_branch_name: str | None = None,
        output_path: str | None = None,
        upload_key: str | None = None, 
    ):
        self.subgraph = subgraph
        self.github_owner = github_owner
        self.repository_name = repository_name
        self.input_branch_name = input_branch_name
        self.input_path = input_path
        self.output_branch_name = output_branch_name
        self.output_path = output_path
        self.upload_key = upload_key


    def _call_api(self) -> None:
        pass
    
    @time_node("wrapper", "download_from_github")
    def _download_from_github(self, state: dict[str, Any]) -> dict[str, Any]:
        return download_from_github(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=self.input_branch_name,
            input_path=self.input_path,
        )
    
    @time_node("wrapper", "run_subgraph")
    def _run_subgraph(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.subgraph.invoke(state)


    @time_node("wrapper", "upload_to_github")
    def _upload_to_github(self, state: dict[str, Any]) -> dict[str, bool]:
        result = upload_to_github(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=self.output_branch_name,
            output_path=self.output_path,
            state=state,
            upload_key=self.upload_key, 
        )
        return {"github_upload_success": result}

    def build_graph(self) -> CompiledGraph:
        wrapper = StateGraph(dict)
        prev = START

        if self.input_path and self.input_branch_name:
            wrapper.add_node("download_from_github", self._download_from_github)
            wrapper.add_edge(prev, "download_from_github")
            prev = "download_from_github"

        wrapper.add_node("run_subgraph", self.subgraph)
        wrapper.add_edge(prev, "run_subgraph")
        prev = "run_subgraph"

        if self.output_path and self.output_branch_name:
            wrapper.add_node("upload_to_github", self._upload_to_github)
            wrapper.add_edge(prev, "upload_to_github")
            prev = "upload_to_github"

        wrapper.add_edge(prev, END)
        return wrapper.compile()


class BuildableSubgraph(Protocol):
    def build_graph(self) -> CompiledGraph: ...

T = TypeVar("T", bound=BuildableSubgraph)


# TODO: Add support for subgraph API invocation
def create_wrapped_subgraph(
    subgraph_cls: type[T],
    github_owner: str, 
    repository_name: str,
    input_branch_name: str | None = None,
    input_path: str | None = None,
    output_branch_name: str | None = None,
    output_path: str | None = None,
    upload_key: str | None = None, 
    *args: Any, 
    **kwargs: Any,
) -> CompiledGraph:

    subgraph = subgraph_cls(
        *args, 
        **kwargs
    ).build_graph()

    if input_path or output_path:
        return GraphWrapper(
            subgraph=subgraph,
            github_owner=github_owner,
            repository_name=repository_name,
            input_branch_name=input_branch_name,
            input_path=input_path,
            output_branch_name=output_branch_name,
            output_path=output_path,
            upload_key=upload_key, 
        ).build_graph()

    return subgraph
