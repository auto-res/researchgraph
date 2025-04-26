import os
import time
import requests
from typing import Literal
from logging import getLogger

logger = getLogger(__name__)

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


DEVICETYPE = Literal["cpu", "gpu"]

# NOTE：API Documentation
# https://docs.github.com/ja/rest/git/refs?apiVersion=2022-11-28#create-a-reference


def create_branch(
    github_owner: str,
    repository_name: str,
    branch_name: str,
    main_sha: str,
    max_retries: int = 10,
) -> bool:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{github_owner}/{repository_name}/git/refs"
    json = {
        "ref": f"refs/heads/{branch_name}",
        "sha": main_sha,
    }
    retries = 0
    wait_time = 1

    while True:
        try:
            response = requests.post(
                url=url, headers=headers, json=json, timeout=10, stream=False
            )
            if response.status_code == 201:
                logger.info("Create branch.")
                return True
            elif response.status_code == 409:
                raise RuntimeError(f"Conflict: {url}")
            elif response.status_code == 422:
                error_message = response.json()
                raise RuntimeError(
                    f"Validation failed, or the endpoint has been spammed.: {url}\n"
                    f"Error message: {error_message}"
                )
            else:
                response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Request failed: {e}")

        retries += 1
        if retries >= max_retries:
            raise RuntimeError("Max retries exceeded. Creating branch failed.")

        logger.info(
            f"Retrying in {wait_time} seconds... (attempt {retries}/{max_retries})"
        )
        time.sleep(wait_time)


if __name__ == "__main__":
    # Example usage
    github_owner = "auto-res2"
    repository_name = "test-branch"
    branch_name = "test"
    sha = "0b4ffd87d989e369a03fce523be014bc6cf75ea8"
    output = create_branch(
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        main_sha=sha,  # You need to provide the SHA of the commit you want to branch from
    )
    print(output)
