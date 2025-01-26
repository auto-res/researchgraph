import os
import requests
import time
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node


API_KEY = os.getenv("DEVIN_API_KEY")


class State(BaseModel):
    session_id: str = Field(default="")
    output_file_path: str = Field(default="")
    error_file_path: str = Field(default="")
    num_iterations: int = Field(default=1)


class FixCodeWithDevinNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
    ):
        super().__init__(input_key, output_key)
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

    def _revision_request_to_devin(
        self, session_id: str, output_str: str, error_str: str
    ):
        url = f"https://api.devin.ai/v1/session/{session_id}/message"
        data = {
            "message": f"""
Please correct the following error output.The standard output is attached for reference.
# Error
{error_str}
# Standard Output
{output_str}
""",
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            print("Revision request sent successfully")
            return
        else:
            print("Failed:", response.status_code, response.text)
            return

    def _execute_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def _get_devin_response(self, session_id):
        get_url = f"https://api.devin.ai/v1/session/{session_id}"
        backoff = 1
        max_attempts = 20
        attempts = 0
        while attempts < max_attempts:
            print(f"Attempt {attempts + 1}")
            response = requests.get(get_url, headers=self.headers)
            if response.status_code != 200:
                print(
                    f"Failed to fetch session status: {response.status_code}, {response.text}"
                )
                break
            response_json = response.json()
            if response_json["status_enum"] in ["blocked", "stopped"]:
                break
            time.sleep(min(backoff, 60))
            backoff = min(backoff * 3, 60)
            attempts += 1

    def execute(self, state: State) -> dict:
        session_id = getattr(state, self.input_key[0])
        output_file_path = getattr(state, self.input_key[1])
        error_file_path = getattr(state, self.input_key[2])
        num_iterations = getattr(state, self.input_key[3])
        output_str = self._execute_file(output_file_path)
        error_str = self._execute_file(error_file_path)
        self._revision_request_to_devin(session_id, output_str, error_str)
        time.sleep(300)
        self._get_devin_response(session_id)
        return {self.output_key[0]: num_iterations + 1}


if __name__ == "__main__":
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "FixCodeWithDevinNode",
        FixCodeWithDevinNode(
            input_key=[
                "session_id",
                "output_file_path",
                "error_file_path",
                "num_iterations",
            ],
            output_key=["num_iterations"],
        ),
    )
    graph_builder.add_edge(START, "FixCodeWithDevinNode")
    graph_builder.add_edge("FixCodeWithDevinNode", END)
    graph = graph_builder.compile()
    state = {
        "session_id": "devin-694c2bca7e9f45e38d7abbf99ed21867",
        "output_file_path": "/workspaces/researchgraph/data/iteration_1/output.txt",
        "error_file_path": "/workspaces/researchgraph/data/iteration_1/error.txt",
        "num_iterations": 1,
    }
    graph.invoke(state, debug=True)
