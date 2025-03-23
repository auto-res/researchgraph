import os
import json

from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.upload_subgraph.nodes.github_upload import github_upload
from researchgraph.upload_subgraph.input_data import upload_subgraph_input_data


class UploadSubgraphInputState(TypedDict):
    paper_content: dict
    output_text_data: str
    branch_name: str
    experiment_devin_url: str
    base_method_text: str
    execution_logs: dict


class UploadSubgraphOutputState(TypedDict):
    completion: bool


class UploadSubgraphState(UploadSubgraphInputState, UploadSubgraphOutputState):
    pass


class UploadSubgraph:
    def __init__(
        self,
        github_owner: str,
        repository_name: str,
        save_dir: str,
    ) -> None:
        self.github_owner = github_owner
        self.repository_name = repository_name
        self.save_dir = save_dir
        self.pdf_file_path = os.path.join(self.save_dir, "paper.pdf")

    def _github_upload_node(self, state: UploadSubgraphState) -> dict:
        completion = github_upload(
            pdf_file_path=self.pdf_file_path,
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=state["branch_name"],
            title=state["paper_content"]["Title"],
            abstract=state["paper_content"]["Abstract"],
            base_paper_url=json.loads(state["base_method_text"])["arxiv_url"],
            experimental_results=state["output_text_data"],
            devin_url=state["experiment_devin_url"],
            all_logs=state["execution_logs"],
        )
        return {"completion": completion}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(UploadSubgraphState)
        # make nodes
        graph_builder.add_node("github_upload_node", self._github_upload_node)
        # make edges
        graph_builder.add_edge(START, "github_upload_node")
        graph_builder.add_edge("github_upload_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    subgraph = UploadSubgraph(
        github_owner="auto-res2",
        repository_name="auto-research",
        save_dir="/workspaces/researchgraph/data",
    ).build_graph()
    result = subgraph.invoke(upload_subgraph_input_data)
