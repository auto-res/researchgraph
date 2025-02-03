import os
import time
from typing import TypedDict

from researchgraph.nodes.utils.api_request_handler import fetch_api_data, retry_request

API_KEY = os.getenv("DEVIN_API_KEY")


class State(TypedDict):
    session_id: str
    output_file_path: str
    error_file_path: str
    fix_iterations: int


class FixCodeWithDevinNode:
    def __init__(
        self,
    ):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

    def _request_revision_to_devin(
        self, session_id: str, output_text_data: str, error_text_data: str
    ):
        url = f"https://api.devin.ai/v1/session/{session_id}/message"
        data = {
            "message": f"""
When the code was executed, the error described below occurred.
Please correct the code and push the corrected code to the remote repository.
Please correct the following error output.The standard output is attached for reference.
# Error
{error_text_data}
# Standard Output
{output_text_data}
""",
        }
        return retry_request(
            fetch_api_data, url, headers=self.headers, data=data, method="POST"
        )

    # def _execute_file(self, file_path):
    #     with open(file_path, "r", encoding="utf-8") as file:
    #         content = file.read()
    #     return content

    def _get_devin_response(self, session_id):
        url = f"https://api.devin.ai/v1/session/{session_id}"

        def should_retry(response):
            # Describe the process so that it is True if you want to retry
            return response.get("status_enum") not in ["blocked", "stopped"]

        return retry_request(
            fetch_api_data,
            url,
            headers=self.headers,
            method="GET",
            check_condition=should_retry,
        )

    def execute(
        self,
        session_id: str,
        output_text_data: str,
        error_text_data: str,
        fix_iteration_count: int,
    ) -> int:
        # session_id = getattr(state, self.input_key[0])
        # output_file_path = getattr(state, self.input_key[1])
        # error_file_path = getattr(state, self.input_key[2])
        # num_iterations = getattr(state, self.input_key[3])
        print("Execute code fixes in Devin")
        self._request_revision_to_devin(session_id, output_text_data, error_text_data)
        time.sleep(60)
        print("Check to see if Devin execution is complete")
        self._get_devin_response(session_id)
        return fix_iteration_count + 1


# if __name__ == "__main__":
#     graph_builder = StateGraph(State)
#     graph_builder.add_node(
#         "FixCodeWithDevinNode",
#         FixCodeWithDevinNode(
#             input_key=[
#                 "session_id",
#                 "output_file_path",
#                 "error_file_path",
#                 "fix_iterations",
#             ],
#             output_key=["num_iterations"],
#         ),
#     )
#     graph_builder.add_edge(START, "FixCodeWithDevinNode")
#     graph_builder.add_edge("FixCodeWithDevinNode", END)
#     graph = graph_builder.compile()
#     state = {
#         "session_id": "devin-a3c0741bce344b93a704277a6fec63d9",
#         "output_file_path": "/workspaces/researchgraph/data/iteration_1/output.txt",
#         "error_file_path": "/workspaces/researchgraph/data/iteration_1/error.txt",
#         "fix_iterations": 1,
#     }
#     graph.invoke(state, debug=True)
