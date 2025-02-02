import os
import base64
import json

from researchgraph.nodes.utils.api_request_handler import fetch_api_data, retry_request

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


class GithubUploadNode:
    def __init__(
        self,
    ):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request_get_github_content(
        self,
        github_owner: str,
        repository_name: str,
        branch_name: str,
        repository_path: str,
    ):
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/contents/{repository_path}"
        params = {
            "ref": f"{branch_name}",
        }
        return retry_request(
            fetch_api_data, url, headers=self.headers, params=params, method="GET"
        )

    def _request_github_file_upload(
        self,
        github_owner: str,
        repository_name: str,
        branch_name: str,
        repository_path: str,
        encoded_data: str,
        sha: str = None,
    ):
        url = f"https://api.github.com/repos/{github_owner}/{repository_name}/contents/{repository_path}"
        data = {
            "message": "Research paper uploaded.",
            "branch": f"{branch_name}",
            "content": encoded_data,
            # "sha": sha,
        }
        if sha is not None:
            data["sha"] = sha
        return retry_request(
            fetch_api_data, url, headers=self.headers, data=data, method="PUT"
        )

    @staticmethod
    def _encoded_pdf_file(pdf_file_path: str):
        with open(pdf_file_path, "rb") as pdf_file:
            encoded_pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")
        return encoded_pdf_data

    @staticmethod
    def _encoded_markdown_data(
        title, abstract, add_github_url, base_github_url, devin_url
    ):
        markdown_text = """
# Automated Research
## Research Title
{title}

## Abstract
{abstract}

## Referenced Research
- {add_github_url}
- {base_github_url}

## Devin Execution Log
{devin_url}""".format(
            title=title,
            abstract=abstract,
            add_github_url=add_github_url,
            base_github_url=base_github_url,
            devin_url=devin_url,
        )
        encoded_markdown_data = base64.b64encode(markdown_text.encode("utf-8")).decode(
            "utf-8"
        )
        return encoded_markdown_data

    def _encodeing_all_data(all_data):
        json_data = json.dumps(all_data, indent=2, ensure_ascii=False)
        encoded_all_data = base64.b64encode(json_data.encode("utf-8")).decode("utf-8")
        return encoded_all_data

    def execute(
        self,
        pdf_file_path: str,
        github_owner: str,
        repository_name: str,
        branch_name: str,
        title: str,
        abstract: str,
        add_github_url: str,
        base_github_url: str,
        devin_url: str,
        all_logs: dict,
    ) -> dict:
        encoded_pdf_data = GithubUploadNode._encoded_pdf_file(pdf_file_path)
        encoded_markdown_data = GithubUploadNode._encoded_markdown_data(
            title, abstract, add_github_url, base_github_url, devin_url
        )
        encoded_all_logs = GithubUploadNode._encodeing_all_data(all_logs)

        print("Paper Upload")
        # response_paper = self._request_get_github_content(
        #     github_owner=github_owner,
        #     repository_name=repository_name,
        #     branch_name=branch_name,
        #     repository_path="paper/paper.pdf"
        # )
        self._request_github_file_upload(
            github_owner=github_owner,
            repository_name=repository_name,
            branch_name=branch_name,
            encoded_data=encoded_pdf_data,
            repository_path="paper/paper.pdf",
            # sha = response_paper["sha"]
        )

        print("Markdown Upload")
        response_readme = self._request_get_github_content(
            github_owner=github_owner,
            repository_name=repository_name,
            branch_name=branch_name,
            repository_path="README.md",
        )

        self._request_github_file_upload(
            github_owner=github_owner,
            repository_name=repository_name,
            branch_name=branch_name,
            encoded_data=encoded_markdown_data,
            repository_path="README.md",
            sha=response_readme["sha"],
        )

        print("All Data Upload")
        # response_all_log = self._request_get_github_content(
        #     github_owner=github_owner,
        #     repository_name=repository_name,
        #     branch_name=branch_name,
        #     repository_path="logs/all_data.json"
        # )
        self._request_github_file_upload(
            github_owner=github_owner,
            repository_name=repository_name,
            branch_name=branch_name,
            encoded_data=encoded_all_logs,
            repository_path="logs/all_logs.json",
            # sha = response_all_log["sha"]
        )
        return True


if __name__ == "__main__":
    node = GithubUploadNode()
    node.execute(
        pdf_file_path="/workspaces/researchgraph/data/test_output.pdf",
        github_owner="auto-res",
        repository_name="experimental-script",
        branch_name="devin/1738495156-learnable-gated-pooling",
        title="Test",
        abstract="Test",
        add_github_url="aaa",
        base_github_url="bbb",
        devin_url="ccc",
        all_logs={
            "objective": "Researching optimizers for fine-tuning LLMs.",
            "base_method_text": "Baseline method description...",
            "add_method_text": "Added method description...",
            "new_method_text": ["New combined method description..."],
            "base_method_code": "def base_method(): pass",
            "add_method_code": "def add_method(): pass",
        },
    )
