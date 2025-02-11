import os
import time

from researchgraph.utils.api_request_handler import fetch_api_data, retry_request

API_KEY = os.getenv("DEVIN_API_KEY")


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
# Instruction
The following error occurred when executing the code in main.py. Please modify the code and push the modified code to the remote repository.
Also, if there is no or little content in “Standard Output”, please modify main.py to make the standard output content richer.
- "Error” contains errors that occur when main.py is run.
- "Standard Output” contains the standard output of the main.py run.
# Error
{error_text_data}
# Standard Output
{output_text_data}""",
        }

        def should_retry(response):
            # Describe the process so that it is True if you want to retry
            return response is not None

        # TODO:RUNNINGならリクエスを送らないようにする
        return retry_request(
            fetch_api_data,
            url,
            headers=self.headers,
            data=data,
            method="POST",
            check_condition=should_retry,
        )

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
        print("Execute code fixes in Devin")
        self._request_revision_to_devin(session_id, output_text_data, error_text_data)
        time.sleep(60)
        print("Check to see if Devin execution is complete")
        self._get_devin_response(session_id)
        return fix_iteration_count + 1


if __name__ == "__main__":
    node = FixCodeWithDevinNode()
    session_id = "devin-d4ba0d7ed2c54a29bac8cfb2dc610d55"
    output_text_data = ""
    error_text_data = "python: can't open file '/home/runner/work/experimental-script/experimental-script/src/main.py': [Errno 2] No such file or directory"
    fix_iteration_count = 1
    node.execute(session_id, output_text_data, error_text_data, fix_iteration_count)
