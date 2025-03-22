import os
import time

from researchgraph.utils.api_request_handler import fetch_api_data, retry_request


API_KEY = os.getenv("DEVIN_API_KEY")


class RetrieveCodeWithDevinNode:
    def __init__(
        self,
    ):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

    def _request_create_session(self, github_url, add_method_text):
        url = "https://api.devin.ai/v1/sessions"
        data = {
            "prompt": f"""
Extract the code related to the contents of the “Description of Methodology” given below from the repository at the “GitHub Repository URL”.
Be sure to make the extracted code available as “extracted_code”.
If there is no code, output “No applicable code”.
# Description of Methodology
{add_method_text}
# GitHub Repository URL
{github_url}""",
            "idempotent": True,
        }
        return retry_request(
            fetch_api_data, url, headers=self.headers, data=data, method="POST"
        )

    def _request_devin_output(self, session_id):
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

    def execute(self, github_url: str, add_method_text: str) -> str:
        create_session_response = self._request_create_session(
            github_url, add_method_text
        )
        if create_session_response:
            print("Successfully created Devin session.")
            session_id = create_session_response["session_id"]
        else:
            print("Failed to create Devin session.")

        time.sleep(120)
        devin_output_response = self._request_devin_output(session_id)
        print(devin_output_response)
        if devin_output_response["structured_output"] is None:
            print("Failed to retrieve Devin output. Response is None.")
            return ""

        print("Successfully retrieved Devin output.")
        extracted_code = devin_output_response["structured_output"].get(
            "extracted_code", ""
        )
        return extracted_code
