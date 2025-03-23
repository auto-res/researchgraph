from researchgraph.utils.api_request_handler import fetch_api_data, retry_request


def _request_create_session(headers, github_url, base_method_text):
    url = "https://api.devin.ai/v1/sessions"
    data = {
        "prompt": f"""
# Instructions
The GitHub repository provided in the "GitHub Repository URL" corresponds to the implementation used in the research described in "Description of Methodology". Please extract the information according to the following rules.
# Rules
- If a machine learning model is used in the implementation, extract its details and the relevant code.
- If a dataset is used in the implementation, extract its details and the relevant code.
- If there are configuration files for experiments, extract all their contents.
- If there is an implementation corresponding to the "Description of Methodology", extract its details.
- If there is information about required Python packages, extract that information.
- If there is information related to the experiments in files such as README.md, extract that information.
- The extracted information should be made available as extracted_info.
# Description of Methodology
{base_method_text}
# GitHub Repository URL
{github_url}""",
        "idempotent": True,
    }
    return retry_request(fetch_api_data, url, headers=headers, data=data, method="POST")


def retrieve_code_with_devin(
    headers: dict, github_url: str, base_method_text: str
) -> tuple[str | None, str | None]:
    response = _request_create_session(headers, github_url, base_method_text)
    if response:
        print("Successfully created Devin session.")
        retrieve_session_id = response["session_id"]
        retrieve_devin_url = response["url"]
        print("Devin URL: ", retrieve_devin_url)
        return retrieve_session_id, retrieve_devin_url
    else:
        print("Failed to create Devin session.")
        return None, None
