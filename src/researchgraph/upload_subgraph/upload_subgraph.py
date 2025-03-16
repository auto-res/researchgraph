from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.upload_subgraph.nodes.github_upload import github_upload
from researchgraph.upload_subgraph.input_data import upload_subgraph_input_data


class UploadSubgraphInputState(TypedDict):
    paper_content: dict
    branch_name: str
    devin_url: str


class UploadSubgraphOutputState(TypedDict):
    completion: bool


class UploadSubgraphState(UploadSubgraphInputState, UploadSubgraphOutputState):
    pass


class UploadSubgraph:
    def __init__(
        self,
        github_owner: str,
        repository_name: str,
        pdf_file_path: str,
    ) -> None:
        self.github_owner = github_owner
        self.repository_name = repository_name
        self.pdf_file_path = pdf_file_path

    def _github_upload_node(self, state: UploadSubgraphState) -> dict:
        completion = github_upload(
            pdf_file_path=self.pdf_file_path,
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            branch_name=state["branch_name"],
            title=state["paper_content"]["Title"],
            abstract=state["paper_content"]["Abstract"],
            devin_url=state["devin_url"],
            all_logs=dict(state),
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
        pdf_file_path="/workspaces/researchgraph/data/test_output.pdf",
    ).build_graph()
    result = subgraph.invoke(upload_subgraph_input_data)
