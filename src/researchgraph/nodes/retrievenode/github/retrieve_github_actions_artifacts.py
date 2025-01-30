import os
import zipfile

from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node

from researchgraph.nodes.utils.api_request_handler import fetch_api_data, retry_request


GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


class State(BaseModel):
    github_owner: str = Field(default="")
    repository_name: str = Field(default="")
    workflow_run_id: int
    save_dir: str = Field(default="")
    num_iterations: int
    output_file_path: str = Field(default="")
    error_file_path: str = Field(default="")


class RetrieveGithubActionsArtifactsNode(Node):
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

    def _request_github_actions_artifacts(
        self, github_owner: str, repository_name: str
    ):
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/actions/artifacts"
        return retry_request(fetch_api_data, url, headers=self.headers, method="GET")

    def _parse_artifacts_info(self, artifacts_infos: dict, workflow_run_id: str):
        artifacts_redirect_url_dict = {}
        for artifacts_info in artifacts_infos["artifacts"]:
            if artifacts_info["workflow_run"]["id"] == workflow_run_id:
                artifacts_redirect_url_dict[artifacts_info["name"]] = artifacts_info[
                    "archive_download_url"
                ]
        return artifacts_redirect_url_dict

    def _request_download_artifacts(
        self, artifacts_redirect_url_dict: dict, iteration_save_dir: str
    ):
        for key, url in artifacts_redirect_url_dict.items():
            response = retry_request(
                fetch_api_data, url, headers=self.headers, method="GET", stream=True
            )
            self._zip_to_txt(response, iteration_save_dir, key)

    def _zip_to_txt(self, response, iteration_save_dir, key):
        zip_file_path = os.path.join(iteration_save_dir, f"{key}.zip")
        with open(zip_file_path, "wb") as f:
            f.write(response)
        print(f"Downloaded artifact saved to: {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(iteration_save_dir)
        print(f"Extracted artifact to: {iteration_save_dir}")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"ZIP file deleted: {zip_file_path}")

    def execute(self, state: State) -> dict:
        github_owner = getattr(state, self.input_key[0])
        repository_name = getattr(state, self.input_key[1])
        workflow_run_id = getattr(state, self.input_key[2])
        save_dir = getattr(state, self.input_key[3])
        num_iterations = getattr(state, self.input_key[4])
        iteration_save_dir = save_dir + f"/iteration_{num_iterations}"
        os.makedirs(iteration_save_dir, exist_ok=True)
        response_artifacts_infos = self._request_github_actions_artifacts(
            github_owner, repository_name
        )
        if response_artifacts_infos:
            print("Successfully retrieved artifacts information.")
        else:
            print("Failure to retrieve artifacts information.")
        get_artifacts_redirect_url_dict = self._parse_artifacts_info(
            response_artifacts_infos, workflow_run_id
        )
        self._request_download_artifacts(
            get_artifacts_redirect_url_dict, iteration_save_dir
        )
        return {
            self.output_key[0]: os.path.join(iteration_save_dir, "output.txt"),
            self.output_key[1]: os.path.join(iteration_save_dir, "error.txt"),
        }


if __name__ == "__main__":
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "retrieve_github_actions_artifacts",
        RetrieveGithubActionsArtifactsNode(
            input_key=[
                "github_owner",
                "repository_name",
                "workflow_run_id",
                "save_dir",
                "num_iterations",
            ],
            output_key=["output_file_path", "error_file_path"],
        ),
    )
    graph_builder.add_edge(START, "retrieve_github_actions_artifacts")
    graph_builder.add_edge("retrieve_github_actions_artifacts", END)
    graph = graph_builder.compile()
    state = {
        "github_owner": "auto-res",
        "repository_name": "experimental-script",
        "workflow_run_id": 13055964079,
        "save_dir": "/workspaces/researchgraph/data",
        "num_iterations": 1,
    }
    graph.invoke(state, debug=True)
