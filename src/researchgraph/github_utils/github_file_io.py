import os
import base64
import json
import logging
from typing import Any
from researchgraph.utils.api_request_handler import fetch_api_data, retry_request

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


def _build_headers():
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _download_file_from_github(
    github_owner: str,
    repository_name: str,
    branch_name: str,
    repository_path: str,
) -> bytes | None:
    url = f"https://api.github.com/repos/{github_owner}/{repository_name}/contents/{repository_path}"
    params = {"ref": branch_name}
    response = retry_request(fetch_api_data, url, headers=_build_headers(), params=params, method="GET")
    if response and "content" in response:
        return base64.b64decode(response["content"])
    return None


def _upload_file_to_github(
    github_owner: str,
    repository_name: str,
    branch_name: str,
    repository_path: str,
    file_content: bytes,
    commit_message: str = "Upload file via ResearchGraph",
) -> bool:
    url = f"https://api.github.com/repos/{github_owner}/{repository_name}/contents/{repository_path}"
    existing = retry_request(fetch_api_data, url, headers=_build_headers(), params={"ref": branch_name}, method="GET")
    sha = existing["sha"] if existing and "sha" in existing else None

    data = {
        "message": commit_message,
        "branch": branch_name,
        "content": base64.b64encode(file_content).decode("utf-8"),
    }
    if sha:
        data["sha"] = sha

    retry_request(fetch_api_data, url, headers=_build_headers(), data=data, method="PUT")
    return True


def github_input_node(
    github_owner: str,
    repository_name: str,
    branch_name: str,
    input_paths: dict[str, str],
) -> dict[str, Any]:
    input_data = {}
    for state_key, github_path in input_paths.items():
        logger.info(f"[GitHub I/O] Downloading input from: {github_path}")
        file_bytes = _download_file_from_github(
            github_owner, 
            repository_name, 
            branch_name, 
            github_path, 
        )
        if not file_bytes:
            logger.error(f"GitHub file not found: {github_path}")
            raise FileNotFoundError(f"Required GitHub input not found: {github_path}")
        try:
            input_data[state_key] = json.loads(file_bytes.decode("utf-8"))
        except Exception as e:
            logger.warning(f"Could not parse {github_path} as JSON. Using raw string. Error: {e}")
            try:
                input_data[state_key] = file_bytes.decode("utf-8") 
            except Exception as e2:
                logger.error(f"Failed to decode {github_path} as UTF-8 string: {e2}")
                raise ValueError(f"Failed to decode content of {github_path}")
    return input_data


def github_output_node(
    github_owner: str,
    repository_name: str,
    branch_name: str,
    output_paths: dict[str, str],
    state: dict[str, Any],
) -> bool:
    completion = True
    for state_key, github_path in output_paths.items():
        logger.info(f"[GitHub I/O] Uploading {state_key} to: {github_path}")
        try:
            value = state[state_key]
            file_bytes = _encode_content(value)
            _upload_file_to_github(
                github_owner, 
                repository_name, 
                branch_name, 
                github_path, 
                file_bytes
            )
        except Exception as e:
            logger.warning(f"Failed to upload {state_key} to {github_path}: {e}", exc_info=True)
            completion = False
    return completion

def _encode_content(value: Any) -> bytes:
    if isinstance(value, str) and os.path.isfile(value):
        with open(value, "rb") as f:
            return f.read()
    elif isinstance(value, bytes):
        return value
    elif isinstance(value, str):
        return value.encode("utf-8")
    else:
        return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")