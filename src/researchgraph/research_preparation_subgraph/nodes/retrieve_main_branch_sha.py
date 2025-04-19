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


def retrieve_main_branch_sha(
    github_owner: str, repository_name: str, max_retries: int = 10
) -> str:
    # time.sleep(3)
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{github_owner}/{repository_name}/branches/main"

    retries = 0
    wait_time = 1

    while True:
        try:
            response = requests.get(url=url, headers=headers, timeout=10, stream=False)
            if response.status_code == 200:
                logger.info("Retrieve main branch sha.")
                response = response.json()
                return response["commit"]["sha"]
            elif response.status_code == 404:
                # NOTE：If the API request is made too early, it may result in a 404 error, so retry processing is used.
                error_message = response.json()
                logger.error(f"error_message: {error_message}")
            elif response.status_code == 301:
                raise RuntimeError(f"Moved permanently: {url}")
            else:
                response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Request failed: {e}")

        retries += 1
        if retries >= max_retries:
            raise RuntimeError("Max retries exceeded. Forking failed.")

        logger.info(
            f"Retrying in {wait_time} seconds... (attempt {retries}/{max_retries})"
        )
        time.sleep(wait_time)


# if __name__ == "__main__":
#     # Example usage
#     github_owner = "auto-res2"
#     repository_name = "test-branch"
#     branch_name = "test"
#     output = retrieve_branch_name(github_owner, repository_name, branch_name)
#     print(output)
