import os
import time
import requests
from typing import Literal
from logging import getLogger

logger = getLogger(__name__)

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


DEVICETYPE = Literal["cpu", "gpu"]

# NOTE：API Documentation
# https://docs.github.com/ja/rest/branches/branches?apiVersion=2022-11-28#get-a-branch


def check_branch_existence(
    github_owner: str, repository_name: str, branch_name: str, max_retries: int = 10
) -> str:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{github_owner}/{repository_name}/branches/{branch_name}"

    retries = 0
    wait_time = 1

    while True:
        try:
            response = requests.get(url=url, headers=headers, timeout=10, stream=False)
            if response.status_code == 200:
                logger.info("The specified branch exists.")
                response = response.json()
                return response["commit"]["sha"]
            elif response.status_code == 404:
                logger.info(f"Branch not found: {url}")
                return ""
            elif response.status_code == 301:
                raise RuntimeError(f"Moved permanently: {url}")
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
    github_owner = "auto-res2"
    repository_name = "test-branch"
    branch_name = "test"
    output = retrieve_branch_name(github_owner, repository_name, branch_name)
    print(output)
