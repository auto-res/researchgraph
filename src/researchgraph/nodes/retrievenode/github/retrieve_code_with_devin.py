import os
import requests
import time
from pydantic import BaseModel, Field
from researchgraph.core.node import Node

API_KEY = os.getenv("DEVIN_API_KEY")

class State(BaseModel):
    github_url: str = Field(default="")
    add_method_text: str = Field(default="")
    extracted_code: str = Field(default="")


class RetrieveCodeWithDevinNode(Node):
    def __init__(
        self, 
        input_key: list[str], 
        output_key: list[str],
    ):
        super().__init__(input_key, output_key)
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
            }
        
    def _create_session(self, github_url, add_method_text):
        url = "https://api.devin.ai/v1/sessions"
        data = {
            "prompt": f"""
Extract the code related to the contents of the “Description of Methodology” given below from the repository at the “GitHub Repository URL”.
The extracted code and description should be output as “extracted_code”.
If there is no code, output “No applicable code”.
            # Description of Methodology
            {add_method_text}
            # GitHub Repository url
            {github_url}""",
            "idempotent": True
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Failed:", response.status_code, response.text)
        session_data = response.json()
        session_id = session_data["session_id"]
        return session_id
        
    def _get_devin_response(self, session_id):
        get_url = f"https://api.devin.ai/v1/session/{session_id}"
        backoff = 1
        max_attempts = 20
        attempts = 0
        while attempts < max_attempts:
            print(f"Attempt {attempts + 1}")
            response = requests.get(get_url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to fetch session status: {response.status_code}, {response.text}")
                return ""
            response_json = response.json()
            if response_json["status_enum"] in ["blocked", "stopped"]:
                return response_json["structured_output"].get("extracted_code", "")
            time.sleep(min(backoff, 60))
            backoff = min(backoff * 3, 60)
            attempts += 1

    def execute(self, state: State) -> dict:
        github_url = getattr(state, self.input_key[0])
        add_method_text = getattr(state, self.input_key[1])
        session_id = self._create_session(github_url, add_method_text)
        time.sleep(120)
        extracted_code = self._get_devin_response(session_id)
        return {
            self.output_key[0]: extracted_code
        }
