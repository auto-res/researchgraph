import os
import time
import requests
from typing import Literal
from logging import getLogger

logger = getLogger(__name__)

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


DEVICETYPE = Literal["cpu", "gpu"]

# NOTE：API Documentation
# https://docs.github.com/ja/rest/repos/forks?apiVersion=2022-11-28#create-a-fork


def fork_repository(
    repository_name: str,
    # NOTE:Make it possible to respond simply by rewriting run_experiment.yml.
    device_type: DEVICETYPE = "cpu",
    organization: str = "",
    max_retries: int = 10,
) -> bool:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if device_type == "cpu":
        url = "https://api.github.com/repos/auto-res/cpu-repository/forks"
    elif device_type == "gpu":
        url = "https://api.github.com/repos/auto-res2/gpu-repository/forks"
    if organization == "":
        json = {
            "name": repository_name,
            "default_branch_only": "true",
        }
    else:
        json = {
            "organization": organization,
            "name": repository_name,
            "default_branch_only": "true",
        }
    retries = 0
    wait_time = 1

    while True:
        try:
            response = requests.post(
                url=url, headers=headers, json=json, timeout=10, stream=False
            )
            if response.status_code == 202:
                logger.info("Fork of the repository was successful.")
                return True
            elif response.status_code == 400:
                raise RuntimeError(
                    f"Error code：{response.status_code} Bad request: {url}"
                )
            elif response.status_code == 403:
                raise RuntimeError(
                    f"Error code：{response.status_code} The access token may be incorrect.: {url}"
                )
            elif response.status_code == 404:
                raise RuntimeError(
                    f"Error code：{response.status_code} Resource not found: {url}"
                )
            elif response.status_code == 422:
                raise RuntimeError(
                    f"Error code：{response.status_code} Validation failed, or the endpoint has been spammed: {url}"
                )
            else:
                logger.error(
                    f"Unhandled status code {response.status_code} for URL: {url}\n"
                )
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")

        retries += 1
        if retries >= max_retries:
            raise RuntimeError("Max retries exceeded. Forking failed.")

        logger.info(
            f"Retrying in {wait_time} seconds... (attempt {retries}/{max_retries})"
        )
        time.sleep(wait_time)


if __name__ == "__main__":
    # Example usage
    repository_name = "test-branch"
    device_type = "gpu"  # or "gpu"
    organization = "auto-res2"  # or "" for no organization
    fork_repository(repository_name, device_type, organization)
