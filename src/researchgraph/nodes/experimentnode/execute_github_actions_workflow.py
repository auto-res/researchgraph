import os
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException

from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node
import time
from datetime import datetime, timezone

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


class State(BaseModel):
    github_owner: str = Field(default="")
    repository_name: str = Field(default="")
    branch_name: str = Field(default="")
    workflow_run_id: int = Field(default=0)


class ExecuteGithubActionsWorkflowNode(Node):
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

    def _execute_github_actions_workflow(
        self, github_owner: str, repository_name: str, branch_name: str
    ):
        workflow_file_name = "run_experiment.yml"
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/actions/workflows/{workflow_file_name}/dispatches"
        data = {
            "ref": f"{branch_name}",
        }
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            print(
                f"Workflow dispatch triggered successfully. Response: {response.status_code}"
            )
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}. Response: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
        except Exception as err:
            print(f"An unexpected error occurred: {err}")
        return

    def _count_github_actions_workflow_runs(self, response: dict):
        num_workflow_runs = len(response["workflow_runs"])
        return num_workflow_runs

    def _get_github_actions_workflow_info(
        self, github_owner: str, repository_name: str, branch_name: str
    ):
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/actions/runs"
        params = {
            "branch": f"{branch_name}",
            "event": "workflow_dispatch",
            # "status": "completed",
            # "status": "in_progress",
        }
        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )
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

    def _parse_workflow_run_id(self, response: dict):
        workflow_timestamp_dict = {}
        latest_timestamp = datetime.min.replace(tzinfo=timezone.utc)
        for res in response["workflow_runs"]:
            created_at = datetime.fromisoformat(
                res["created_at"].replace("Z", "+00:00")
            )
            workflow_timestamp_dict[created_at] = res["id"]
            if created_at > latest_timestamp:
                latest_timestamp = created_at
        print(workflow_timestamp_dict)
        return workflow_timestamp_dict[latest_timestamp]

    def _check_confirmation_of_execution_completion(self, response: dict):
        status_list = []
        for res in response["workflow_runs"]:
            status_list.append(res["status"])
        return all(item == "completed" for item in status_list)

    def execute(self, state):
        github_owner = getattr(state, "github_owner")
        repository_name = getattr(state, "repository_name")
        branch_name = getattr(state, "branch_name")

        # NOTE: Check the number of runs before executing workflow
        response_before_execution = self._get_github_actions_workflow_info(
            github_owner, repository_name, branch_name
        )
        num_workflow_runs_before_execution = self._count_github_actions_workflow_runs(
            response_before_execution
        )
        print(
            f"Number of workflow runs before execution:{num_workflow_runs_before_execution}"
        )

        # Execute the workflow
        print("Running the Experiment")
        self._execute_github_actions_workflow(
            github_owner, repository_name, branch_name
        )
        time.sleep(60)

        # NOTE:The number of runs is increased by one to confirm that execution is complete.
        num_workflow_runs_after_execution = num_workflow_runs_before_execution
        while True:
            response_after_execution = self._get_github_actions_workflow_info(
                github_owner, repository_name, branch_name
            )
            num_workflow_runs_after_execution = (
                self._count_github_actions_workflow_runs(response_after_execution)
            )
            if (
                num_workflow_runs_after_execution
                == num_workflow_runs_before_execution + 1
            ) & self._check_confirmation_of_execution_completion(
                response_after_execution
            ):
                break
            time.sleep(10)
        workflow_run_id = self._parse_workflow_run_id(response_after_execution)

        return {self.output_key[0]: workflow_run_id}


if __name__ == "__main__":
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "retrieve_github_actions_artifacts",
        ExecuteGithubActionsWorkflowNode(
            input_key=["github_owner", "repository_name", "branch_name"],
            output_key=["workflow_run_id"],
        ),
    )
    graph_builder.add_edge(START, "retrieve_github_actions_artifacts")
    graph_builder.add_edge("retrieve_github_actions_artifacts", END)
    graph = graph_builder.compile()
    state = {
        "github_owner": "fuyu-quant",
        "repository_name": "experimental-script",
        "branch_name": "devin/1737913235-learnable-gated-pooling",
    }
    graph.invoke(state, debug=True)
