import os
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
import zipfile

from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node

from pprint import pprint

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

    # artifactsの情報を取得
    def _get_github_actions_artifacts(self, github_owner: str, repository_name: str):
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/actions/artifacts"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except RequestException as req_err:
            print(f"An error occurred: {req_err}")
        except Exception as err:
            print(f"An unexpected error occurred: {err}")
        return None

    # artifactsの情報をパース
    def _parse_artifacts_info(self, artifacts_infos: dict, workflow_run_id: str):
        artifacts_redirect_url_dict = {}
        for artifacts_info in artifacts_infos["artifacts"]:
            if artifacts_info["workflow_run"]["id"] == workflow_run_id:
                artifacts_redirect_url_dict[artifacts_info["name"]] = artifacts_info[
                    "archive_download_url"
                ]
        return artifacts_redirect_url_dict

    # artifactsをダウンロード
    def _download_artifacts(
        self, artifacts_redirect_url_dict: dict, iteration_save_dir: str
    ):
        for key, value in artifacts_redirect_url_dict.items():
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            try:
                response = requests.get(value, headers=headers)
                response.raise_for_status()
                if response.status_code == 200:
                    self._zip_to_txt(response, iteration_save_dir, key)
            except HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
            except ConnectionError as conn_err:
                print(f"Connection error occurred: {conn_err}")
            except Timeout as timeout_err:
                print(f"Timeout error occurred: {timeout_err}")
            except RequestException as req_err:
                print(f"An error occurred: {req_err}")
            except Exception as err:
                print(f"An unexpected error occurred: {err}")
                pprint(response)
        return

    def _zip_to_txt(self, response, iteration_save_dir, key):
        zip_file_path = os.path.join(iteration_save_dir, f"{key}.zip")
        with open(zip_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(iteration_save_dir)
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"ZIP file deleted: {zip_file_path}")
        else:
            print(f"Tried to delete, but file not found: {zip_file_path}")
        return

    def execute(self, state: State) -> dict:
        github_owner = getattr(state, self.input_key[0])
        repository_name = getattr(state, self.input_key[1])
        workflow_run_id = getattr(state, self.input_key[2])
        save_dir = getattr(state, self.input_key[3])
        num_iterations = getattr(state, self.input_key[4])
        iteration_save_dir = save_dir + f"iteration_{num_iterations}"
        os.makedirs(iteration_save_dir, exist_ok=True)
        artifacts_infos = self._get_github_actions_artifacts(
            github_owner, repository_name
        )
        get_artifacts_redirect_url_dict = self._parse_artifacts_info(
            artifacts_infos, workflow_run_id
        )
        self._download_artifacts(get_artifacts_redirect_url_dict, iteration_save_dir)
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
        "github_owner": "fuyu-quant",
        "repository_name": "experimental-script",
        "workflow_run_id": 12975375976,
        "save_dir": "/workspaces/researchgraph/data",
        "num_iterations": 1,
    }
    graph.invoke(state, debug=True)
