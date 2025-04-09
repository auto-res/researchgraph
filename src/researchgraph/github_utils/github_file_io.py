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


def _download_file_bytes_from_github(
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


def _upload_file_bytes_to_github(
    github_owner: str,
    repository_name: str,
    branch_name: str,
    repository_path: str,
    file_content: bytes,
    commit_message: str,
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


def download_from_github(
    github_owner: str,
    repository_name: str,
    branch_name: str,
    input_path: str,
) -> dict[str, Any]:
    logger.info(f"[GitHub I/O] Downloading input from: {input_path}")
    file_bytes = _download_file_bytes_from_github(
        github_owner, 
        repository_name, 
        branch_name, 
        input_path, 
    )
    if not file_bytes:
        logger.error(f"GitHub file not found: {input_path}")
        raise FileNotFoundError(f"Required GitHub input not found: {input_path}")
    try:
        decoded = json.loads(file_bytes.decode("utf-8"))
        if not isinstance(decoded, dict):
            logger.error(f"Decoded input is not a dictionary: {input_path}")
            raise ValueError("Decoded input is not a dictionary.")
        return decoded
    except Exception as e:
        error_message = f"Failed to parse full-state JSON from {input_path}: {e}"
        logger.error(error_message)
        raise ValueError(error_message) from e


def upload_to_github(
    github_owner: str,
    repository_name: str,
    branch_name: str,
    output_path: str,
    state: dict[str, Any],
    extra_files: list[tuple[str, str, list[str]]] | None = None,  # [branch_name, repo_path, local_paths]
    commit_message: str = "Upload file via ResearchGraph",
) -> bool:
    logger.info(f"[GitHub I/O] Uploading state to: {output_path}")
    success = True

    try:
        file_bytes = _encode_content(state)
        _upload_file_bytes_to_github(
            github_owner, 
            repository_name, 
            branch_name, 
            output_path, 
            file_bytes, 
            commit_message=commit_message, 
        )
    except Exception as e:
        logger.warning(f"Failed to upload state to {output_path}: {e}", exc_info=True)
        success = False

    if extra_files:
        for extra_branch, repo_base_path, local_paths in extra_files:
            for local_path in local_paths:
                try:
                    filename = os.path.basename(local_path)
                    repo_path = os.path.join(repo_base_path, filename).replace("\\", "/")
                    with open(local_path, "rb") as f:
                        file_bytes = f.read()

                    _upload_file_bytes_to_github(
                        github_owner,
                        repository_name,
                        extra_branch,
                        repo_path,
                        file_bytes,
                        commit_message=commit_message,
                    )
                    logger.info(f"[GitHub I/O] Uploaded extra file: {local_path} to {repo_path}")
                except Exception as e:
                    logger.warning(f"Failed to upload extra file {local_path} to {repo_path}: {e}", exc_info=True)
                    success = False
    return success

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


def create_branch_on_github(
    github_owner: str,
    repository_name: str,
    new_branch_name: str,
    base_branch_name: str,
    ) -> None:
    ref_url = f"https://api.github.com/repos/{github_owner}/{repository_name}/git/ref/heads/{base_branch_name}"
    ref_response = retry_request(fetch_api_data, ref_url, headers=_build_headers(), method="GET")
    if not ref_response or "object" not in ref_response or "sha" not in ref_response["object"]:
        logger.error(f"Failed to get base branch '{base_branch_name}' SHA")
        raise ValueError(f"Failed to get base branch '{base_branch_name}' SHA")
    
    base_sha = ref_response["object"]["sha"]
    create_url = f"https://api.github.com/repos/{github_owner}/{repository_name}/git/refs"
    payload = {
        "ref": f"refs/heads/{new_branch_name}",
        "sha": base_sha,
    }
    try:
        retry_request(fetch_api_data, create_url, headers=_build_headers(), data=payload, method="POST")
        logger.info(f"Created new branch: {new_branch_name} based on {base_branch_name}")
    except Exception as e:
        logger.warning(f"[GitHub] Branch creation failed or already exists: {new_branch_name} — {e}", exc_info=True)
        raise
