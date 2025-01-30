import os

from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node

from datetime import datetime, timezone

from researchgraph.nodes.utils.api_request_handler import fetch_api_data, retry_request

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

    def _request_github_actions_workflow_execution(
        self, github_owner: str, repository_name: str, branch_name: str
    ):
        workflow_file_name = "run_experiment.yml"
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/actions/workflows/{workflow_file_name}/dispatches"
        data = {
            "ref": f"{branch_name}",
        }
        return retry_request(
            fetch_api_data, url, headers=self.headers, data=data, method="POST"
        )

    def _request_github_actions_workflow_info_before_execution(
        self, github_owner: str, repository_name: str, branch_name: str
    ):
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/actions/runs"
        params = {
            "branch": f"{branch_name}",
            "event": "workflow_dispatch",
        }
        return retry_request(
            fetch_api_data, url, headers=self.headers, params=params, method="GET"
        )

    def _request_github_actions_workflow_info_after_execution(
        self,
        github_owner: str,
        repository_name: str,
        branch_name: str,
        num_workflow_runs_before_execution: int,
    ):
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/actions/runs"
        params = {
            "branch": f"{branch_name}",
            "event": "workflow_dispatch",
        }

        # NOTE:The number of runs is increased by one to confirm that execution is complete.
        def should_retry(response) -> bool:
            # Describe the process so that it is True if you want to retry
            num_workflow_runs_after_execution = (
                self._count_github_actions_workflow_runs(response)
            )
            return not (
                (
                    num_workflow_runs_after_execution
                    == num_workflow_runs_before_execution + 1
                )
                and self._check_confirmation_of_execution_completion(response)
            )

        return retry_request(
            fetch_api_data,
            url,
            headers=self.headers,
            params=params,
            method="GET",
            check_condition=should_retry,
        )

    def _count_github_actions_workflow_runs(self, response: dict):
        num_workflow_runs = len(response["workflow_runs"])
        return num_workflow_runs

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

        # Check the number of runs before executing workflow
        response_before_execution = (
            self._request_github_actions_workflow_info_before_execution(
                github_owner, repository_name, branch_name
            )
        )
        if response_before_execution:
            print(
                "Successfully retrieved information on Github actions prior to execution of workflow."
            )
        else:
            print(
                "Failure to retrieve information on Github actions before executing workflow."
            )
        num_workflow_runs_before_execution = self._count_github_actions_workflow_runs(
            response_before_execution
        )
        print(
            f"Number of workflow runs before execution:{num_workflow_runs_before_execution}"
        )

        # Execute the workflow
        self._request_github_actions_workflow_execution(
            github_owner, repository_name, branch_name
        )

        # Check the number of runs after executing workflow
        response_after_execution = (
            self._request_github_actions_workflow_info_after_execution(
                github_owner,
                repository_name,
                branch_name,
                num_workflow_runs_before_execution,
            )
        )
        if response_before_execution:
            print(
                "Successfully retrieved information on Github actions after execution of workflow."
            )
        else:
            print(
                "Failure to retrieve information on Github actions after executing workflow."
            )

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
        "github_owner": "auto-res",
        "repository_name": "experimental-script",
        "branch_name": "devin/1738251222-learnable-gated-pooling",
    }
    graph.invoke(state, debug=True)
