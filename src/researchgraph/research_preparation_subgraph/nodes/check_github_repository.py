import os
import time
import requests
from logging import getLogger

logger = getLogger(__name__)

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


# NOTE：API Documentation
# https://docs.github.com/ja/rest/repos/repos?apiVersion=2022-11-28#get-a-repository


def check_github_repository(
    github_owner: str, repository_name: str, max_retries: int = 10
) -> bool | None:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{github_owner}/{repository_name}"
    retries = 0
    wait_time = 1

    while True:
        try:
            response = requests.get(url=url, headers=headers, timeout=10, stream=False)
            if response.status_code == 200:
                logger.info("A research repository exists.")
                return True
            elif response.status_code == 404:
                logger.info(f"Repository not found: {url}")
                return False
            elif response.status_code == 403:
                raise RuntimeError(
                    f"Access forbidden: {url}\n"
                    "The requested resource has been permanently moved to a new location."
                )
            elif response.status_code == 301:
                raise RuntimeError(
                    f"Access forbidden: {url}\n"
                    "You do not have permission to access this resource."
                )
            else:
                logger.error(
                    f"Unhandled status code {response.status_code} for URL: {url}"
                )
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")

        retries += 1
        if retries >= max_retries:
            raise RuntimeError(f"Max retries exceeded for URL: {url}")

        logger.info(
            f"Retrying in {wait_time} seconds... (attempt {retries}/{max_retries})"
        )
        time.sleep(wait_time)


if __name__ == "__main__":
    # Example usage
    github_owner = "auto-res2"
    repository_name = "gpu-repository"
    github_owner = "auto-res"
    repository_name = "cpu-repositor"
    get_github_repository(github_owner, repository_name)
