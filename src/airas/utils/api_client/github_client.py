import os
import requests
import logging

from airas.utils.logging_utils import setup_logging

setup_logging()

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


class GithubClient:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def get_repository_content(
        self, repository_owner: str, repository_name: str, file_path: str
    ) -> str | None:
        # https://docs.github.com/ja/rest/repos/contents?apiVersion=2022-11-28#get-repository-content
        # For public repositories, no access token is required.
        url = f"https://api.github.com/repos/{repository_owner}/{repository_name}/contents/{file_path}"
        headers = {
            "Accept": "application/vnd.github.raw+json",
            "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = requests.get(url=url, headers=headers, timeout=10, stream=False)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 302:
            self.logger.warning("Resource not found.")
            return None
        elif response.status_code == 304:
            self.logger.warning("Not Modified.")
            return None
        elif response.status_code == 403:
            self.logger.warning("Forbidden.")
            return None
        elif response.status_code == 404:
            self.logger.warning("Resource not found.")
            return None
        else:
            raise RuntimeError(
                f"Unhandled status code {response.status_code} for URL: {url}\n"
            )

    def get_a_tree(
        self, repository_owner: str, repository_name: str, tree_sha: str
    ) -> dict | None:
        # https://docs.github.com/ja/rest/git/trees?apiVersion=2022-11-28#get-a-tree
        # For public repositories, no access token is required.
        url = f"https://api.github.com/repos/{repository_owner}/{repository_name}/git/trees/{tree_sha}"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        params = {"recursive": "true"}
        response = requests.get(
            url=url, headers=headers, params=params, timeout=10, stream=False
        )
        if response.status_code == 200:
            response = response.json()
            return response
        elif response.status_code == 404:
            self.logger.warning("Resource not found.")
            return None
        elif response.status_code == 409:
            self.logger.warning("Conflict.")
            return None
        elif response.status_code == 422:
            self.logger.warning("Validation failed, or the endpoint has been spammed.")
            return None
        else:
            raise RuntimeError(
                f"Unhandled status code {response.status_code} for URL: {url}\n"
            )


if __name__ == "__main__":
    # repository_owner="fuyu-quant"
    # repository_name="iblm"
    repository_owner = "auto-res"
    repository_name = "airas"
    file_path = "README.md"
    github_client = GithubClient()
    response = github_client.get_repository_content(
        repository_owner=repository_owner,
        repository_name=repository_name,
        file_path=file_path,
    )
    print(response)

    # response = github_client.get_a_tree(
    #     repository_owner=repository_owner,
    #     repository_name=repository_name,
    #     tree_sha="main",
    # )
    # print(response)
