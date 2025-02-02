import os
import base64

from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node
from researchgraph.nodes.utils.api_request_handler import fetch_api_data, retry_request

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


class State(BaseModel):
    pdf_file_path: str = Field(default="")
    github_owner: str = Field(default="")
    repository_name: str = Field(default="")
    branch_name: str = Field(default="")
    completion: bool = Field(default=False)


class GithubUploadNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
    ):
        super().__init__(input_key, output_key)
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request_github_file_upload(
        self,
        github_owner: str,
        repository_name: str,
        branch_name: str,
        encoded_pdf_data: str,
    ):
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/contents/paper/paper.pdf"
        data = {
            "message": "Research paper uploaded.",
            "branch": f"{branch_name}",
            "content": encoded_pdf_data,
        }
        return retry_request(
            fetch_api_data, url, headers=self.headers, data=data, method="PUT"
        )

    @staticmethod
    def _encoded_pdf_file(pdf_file_path: str):
        with open(pdf_file_path, "rb") as pdf_file:
            encoded_pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")
        return encoded_pdf_data

    def execute(self, state: State):
        pdf_file_path = getattr(state, self.input_key[0])
        github_owner = getattr(state, self.input_key[1])
        repository_name = getattr(state, self.input_key[2])
        branch_name = getattr(state, self.input_key[3])
        encoded_pdf_data = GithubUploadNode._encoded_pdf_file(pdf_file_path)
        self._request_github_file_upload(
            github_owner, repository_name, branch_name, encoded_pdf_data
        )
        return {self.output_key[0]: True}


if __name__ == "__main__":
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "GithubUploadNode",
        GithubUploadNode(
            input_key=[
                "pdf_file_path",
                "github_owner",
                "repository_name",
                "branch_name",
            ],
            output_key=["completion"],
        ),
    )
    graph_builder.add_edge(START, "GithubUploadNode")
    graph_builder.add_edge("GithubUploadNode", END)
    graph = graph_builder.compile()
    state = {
        "pdf_file_path": "/workspaces/researchgraph/data/test_output.pdf",
        "github_owner": "auto-res",
        "repository_name": "experimental-script",
        "branch_name": "devin/1738251222-learnable-gated-pooling",
    }
    graph.invoke(state, debug=True)
