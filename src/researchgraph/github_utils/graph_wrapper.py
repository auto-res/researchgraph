import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from researchgraph.utils.logging_utils import setup_logging
from researchgraph.github_utils.github_file_io import download_from_github, upload_to_github, create_branch_on_github
from researchgraph.utils.execution_timers import time_node

setup_logging()
logger = logging.getLogger(__name__)

# class GraphWrapperState(TypedDict):
#     github_upload_success: bool

    
class GraphWrapper:
    def __init__(
        self,
        subgraph: CompiledGraph,
        output_state: type[TypedDict], 
        github_owner: str,
        repository_name: str,
        input_branch_name: str | None = None,  
        input_path: str | None = None,
        output_branch_name: str | None = None,
        output_path: str | None = None,
        extra_files: list[tuple[str, str, list[str]]] | None = None, 
    ):
        self.subgraph = subgraph
        self.output_state = output_state
        self.github_owner = github_owner
        self.repository_name = repository_name
        self.input_branch_name = input_branch_name
        self.input_path = input_path
        self.output_branch_name = output_branch_name
        self.output_path = output_path
        self.extra_files = extra_files

        self.subgraph_name = getattr(subgraph, "__source_subgraph_name__", "subgraph").lower()
        self.output_state_keys = (
            list(output_state.__annotations__.keys()) if output_state else []
        )

    def _deep_merge(self, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(old)
        for k, v in new.items():
            if (
                k in result
                and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = deepcopy(v)
        return result
        
    def _create_branch_name(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        new_branch = f"{self.output_branch_name}-{self.subgraph_name}-{ts}"
    
        create_branch_on_github(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            new_branch_name=new_branch,
            base_branch_name=self.output_branch_name
        )

        logger.info(f"Created new GitHub branch: {new_branch}")
        return new_branch
    
    def _format_extra_files(self, branch_name: str) -> list[tuple[str, str, list[str]]] | None:
        if self.extra_files is None:
            return None
        
        formatted_files = []
        for target_branch, target_path, file_paths in self.extra_files:
            tb = target_branch.replace("{{ branch_name }}", branch_name)
            tp = target_path.replace("{{ branch_name }}", branch_name)
            fps = [fp.replace("{{ branch_name }}", branch_name) for fp in file_paths]
            formatted_files.append((tb, tp, fps))

        return formatted_files

    def _call_api(self) -> None:
        pass
    
    @time_node("wrapper", "download_from_github")
    def _download_from_github(self, state: dict[str, Any]) -> dict[str, Any]:
        original_state = download_from_github(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=self.input_branch_name,
            input_path=self.input_path,
        )
        return {
            "original_state": deepcopy(original_state)  
        }
    
    @time_node("wrapper", "run_subgraph")
    def _run_subgraph(self, state: dict[str, Any]) -> dict[str, Any]:
        original_state = state.get("original_state")
        output_state = self.subgraph.invoke(original_state)
        return {
            "original_state": original_state, 
            "output_state": output_state, 
        }

    @time_node("wrapper", "upload_to_github")
    def _upload_to_github(self, state: dict[str, Any]) -> dict[str, bool]:

        original_state = state.get("original_state", {})
        raw_output_state = state.get("output_state", {})
        output_state = {k: raw_output_state[k] for k in self.output_state_keys if k in raw_output_state}
        merged_state = self._deep_merge(original_state, output_state)

        key_conflict = bool(set(original_state) & set(output_state))
        if key_conflict:
            final_branch = self._create_branch_name()
            logger.info(f"Key conflict detected. Created new branch: {final_branch}")
        else:
            final_branch = self.output_branch_name
            logger.info(f"No key conflict. Using existing branch: {final_branch}")

        formatted_extra_files = self._format_extra_files(final_branch)

        result = upload_to_github(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=final_branch,
            output_path=self.output_path,
            state=merged_state,
            extra_files=formatted_extra_files, 
            commit_message=f"Update by subgraph: {self.subgraph_name}"
        )
        return {"github_upload_success": result}

    def build_graph(self) -> CompiledGraph:
        wrapper = StateGraph(dict[str, Any])
        prev = START

        if self.input_path and self.input_branch_name:
            wrapper.add_node("download_from_github", self._download_from_github)
            wrapper.add_edge(prev, "download_from_github")
            prev = "download_from_github"

        wrapper.add_node("run_subgraph", self._run_subgraph)
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
    subgraph: type[T],
    output_state: type[TypedDict], 
    github_owner: str, 
    repository_name: str,
    input_branch_name: str,
    input_path: str,
    output_branch_name: str,
    output_path: str,
    extra_files: list[tuple[str, str, list[str]]] | None = None, 
    *args: Any, 
    **kwargs: Any,
) -> CompiledGraph:

    subgraph_instance = subgraph(*args, **kwargs)
    compiled_subgraph = subgraph_instance.build_graph()
    setattr(compiled_subgraph, "__source_subgraph_name__", subgraph_instance.__class__.__name__)

    if input_path or output_path:
        return GraphWrapper(
            subgraph=compiled_subgraph,
            output_state=output_state, 
            github_owner=github_owner,
            repository_name=repository_name,
            input_branch_name=input_branch_name,
            input_path=input_path,
            output_branch_name=output_branch_name,
            output_path=output_path,
            extra_files = extra_files, 
        ).build_graph()

    return compiled_subgraph
