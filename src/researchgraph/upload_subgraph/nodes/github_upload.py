import os
import base64
import json

from researchgraph.utils.api_request_handler import fetch_api_data, retry_request

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


def _request_get_github_content(
    headers: dict,
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
        fetch_api_data, url, headers=headers, params=params, method="GET"
    )


def _request_github_file_upload(
    headers: dict,
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
    return retry_request(fetch_api_data, url, headers=headers, data=data, method="PUT")


def _encoded_pdf_file(pdf_file_path: str):
    with open(pdf_file_path, "rb") as pdf_file:
        encoded_pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")
    return encoded_pdf_data


def _encoded_markdown_data(title: str, abstract: str, devin_url: str):
    markdown_text = """
# Automated Research
## Research Title
{title}

## Abstract
{abstract}

## Devin Execution Log
{devin_url}""".format(
        title=title,
        abstract=abstract,
        devin_url=devin_url,
    )
    encoded_markdown_data = base64.b64encode(markdown_text.encode("utf-8")).decode(
        "utf-8"
    )
    return encoded_markdown_data


def _encoding_all_data(all_data: dict):
    json_data = json.dumps(all_data, indent=2, ensure_ascii=False)
    encoded_all_data = base64.b64encode(json_data.encode("utf-8")).decode("utf-8")
    return encoded_all_data


def github_upload(
    pdf_file_path: str,
    github_owner: str,
    repository_name: str,
    branch_name: str,
    title: str,
    abstract: str,
    devin_url: str,
    all_logs: dict,
) -> bool:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    encoded_pdf_data = _encoded_pdf_file(pdf_file_path)
    encoded_markdown_data = _encoded_markdown_data(title, abstract, devin_url)
    encoded_all_logs = _encoding_all_data(all_logs)

    print("Paper Upload")
    # response_paper = _request_get_github_content(
    #     github_owner=github_owner,
    #     repository_name=repository_name,
    #     branch_name=branch_name,
    #     repository_path="paper/paper.pdf"
    # )
    _request_github_file_upload(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        encoded_data=encoded_pdf_data,
        repository_path="paper/paper.pdf",
        # sha = response_paper["sha"]
    )

    print("Markdown Upload")
    response_readme = _request_get_github_content(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        repository_path="README.md",
    )

    _request_github_file_upload(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        encoded_data=encoded_markdown_data,
        repository_path="README.md",
        sha=response_readme["sha"],
    )

    print("All Data Upload")
    _request_github_file_upload(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        encoded_data=encoded_all_logs,
        repository_path="logs/all_logs.json",
    )
    return True


if __name__ == "__main__":
    branch_name = "devin-047ba743189c49f59469287c0103d1b8"
    completion = github_upload(
        pdf_file_path="/workspaces/researchgraph/data/test_output.pdf",
        github_owner="auto-res2",
        repository_name="auto-research",
        branch_name=branch_name,
        title="Test Title",
        abstract="Test Abstract",
        devin_url="https://app.devin.ai/sessions/29ebce5ed6f247a4a5bb0d109ecb2f9b",
        all_logs={"test": "test"},
    )
