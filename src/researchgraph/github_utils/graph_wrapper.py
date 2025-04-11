import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from researchgraph.utils.logging_utils import setup_logging
from researchgraph.utils.execution_timers import time_node
from researchgraph.github_utils.github_file_io import (
    download_from_github, 
    upload_to_github, 
    create_branch_on_github, 
    ExtraFileConfig, 
)

setup_logging()
logger = logging.getLogger(__name__)

# class GraphWrapperState(TypedDict):
#     github_upload_success: bool

class GithubGraphWrapper:
    def __init__(
        self,
        subgraph: CompiledGraph,
        output_state: type[TypedDict], 
        github_repository: str, 
        branch_name: str,  
        research_file_path: str = ".research/research_history.json", 
        extra_files: list[ExtraFileConfig] | None = None, 
        perform_download: bool = True, 
        perform_upload: bool = True, 
        public_branch: str = "gh-pages", 
    ):
        self.subgraph = subgraph
        self.output_state = output_state
        self.github_repository = github_repository
        self.branch_name = branch_name
        self.research_file_path = research_file_path
        self.extra_files = extra_files
        self.perform_download = perform_download
        self.perform_upload = perform_upload
        self.public_branch = public_branch

        self.subgraph_name = getattr(subgraph, "__source_subgraph_name__", "subgraph").lower()
        self.output_state_keys = (
            list(output_state.__annotations__.keys()) if output_state else []
        )
        try:
            owner, repository = github_repository.split("/", 1)
        except ValueError:
            raise ValueError("Repo string must be in the format 'owner/repository'")
        self.github_owner = owner
        self.repository_name = repository

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
        new_branch = f"{self.branch_name}-{self.subgraph_name}-{ts}"
    
        create_branch_on_github(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            new_branch_name=new_branch,
            base_branch_name=self.branch_name
        )

        logger.info(f"Created new GitHub branch: {new_branch}")
        return new_branch
    
    def _format_extra_files(self, branch_name: str) -> list[ExtraFileConfig] | None:
        if self.extra_files is None:
            return None
        
        formatted_files = []
        for file_config in self.extra_files:
            tb = file_config["upload_branch"].replace("{{ branch_name }}", branch_name)
            tp = file_config["upload_dir"].replace("{{ branch_name }}", branch_name)
            fps = [fp.replace("{{ branch_name }}", branch_name) for fp in file_config["local_file_paths"]]
            formatted_files.append({
                "upload_branch": tb, 
                "upload_dir": tp, 
                "local_file_paths": fps, 
            })

        return formatted_files

    def _call_api(self) -> None:
        pass
    
    @time_node("wrapper", "download_from_github")
    def _download_from_github(self, state: dict[str, Any]) -> dict[str, Any]:
        original_state = download_from_github(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=self.branch_name,
            input_path=self.research_file_path,
        )
        return {
            "original_state": deepcopy(original_state)  
        }
    
    @time_node("wrapper", "run_subgraph")
    def _run_subgraph(self, state: dict[str, Any]) -> dict[str, Any]:
        original_state = state.get("original_state") or {}
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
            final_branch = self.branch_name
            logger.info(f"No key conflict. Using existing branch: {final_branch}")

        formatted_extra_files = self._format_extra_files(final_branch)

        result = upload_to_github(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=final_branch,
            output_path=self.research_file_path,
            state=merged_state,
            extra_files=formatted_extra_files, 
            commit_message=f"Update by subgraph: {self.subgraph_name}"
        )
        if formatted_extra_files is not None:
            for file_config in formatted_extra_files:
                if file_config["upload_branch"].lower() == self.public_branch.lower():
                    target_path = file_config["upload_dir"]
                    if not target_path.endswith("/"):
                        target_path += "/"
                    github_pages_url = f"https://{self.github_owner}.github.io/{self.repository_name}/{target_path}"
                    logger.info(f"Uploaded HTML available at: {github_pages_url}")
                    break

        return {"github_upload_success": result}

    def build_graph(self) -> CompiledGraph:
        wrapper = StateGraph(dict[str, Any])
        prev = START

        if self.perform_download:
            wrapper.add_node("download_from_github", self._download_from_github)
            wrapper.add_edge(prev, "download_from_github")
            prev = "download_from_github"

        wrapper.add_node("run_subgraph", self._run_subgraph)
        wrapper.add_edge(prev, "run_subgraph")
        prev = "run_subgraph"

        if self.perform_upload:
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
) -> type:
    
    class GithubGraphRunner(GithubGraphWrapper):
        def __init__(
            self, 
            github_repository: str,
            branch_name: str,
            research_file_path: str = ".research/research_history.json",
            extra_files: list[ExtraFileConfig] | None = None,
            perform_download: bool = True, 
            perform_upload: bool = True, 
            public_branch: str = "gh-pages", 
            *args: Any,
            **kwargs: Any,
        ):
            
            subgraph_instance = subgraph(*args, **kwargs)
            compiled_subgraph = subgraph_instance.build_graph()
            setattr(compiled_subgraph, "__source_subgraph_name__", subgraph_instance.__class__.__name__)
            super().__init__(
                subgraph=compiled_subgraph,
                output_state=output_state,
                github_repository=github_repository, 
                branch_name=branch_name,
                research_file_path=research_file_path,
                extra_files=extra_files,
                perform_download=perform_download, 
                perform_upload=perform_upload, 
                public_branch=public_branch, 
            )

        def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
            graph = self.build_graph()
            return graph.invoke(inputs)
        
    return GithubGraphRunner