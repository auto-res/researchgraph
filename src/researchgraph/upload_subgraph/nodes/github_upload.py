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
) -> dict | None:
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
    sha: str | None,
):
    url = f"https://api.github.com/repos/{github_owner}/{repository_name}/contents/{repository_path}"
    data = {
        "message": "Research paper uploaded.",
        "branch": f"{branch_name}",
        "content": encoded_data,
    }
    if sha is not None:
        data["sha"] = sha
    return retry_request(fetch_api_data, url, headers=headers, data=data, method="PUT")


def _encoded_pdf_file(pdf_file_path: str):
    with open(pdf_file_path, "rb") as pdf_file:
        encoded_pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")
    return encoded_pdf_data


def _encoded_markdown_data(
    title: str,
    abstract: str,
    paper_url: str,
    base_paper_url: str,
    research_graph_execution_log: str,
    devin_url: str,
):
    markdown_text = f"""
# {title}
> ⚠️ **NOTE:** This research is an automatic research using Research Graph.
## Abstract
{abstract}

- [Full paper]({paper_url})
- [Related work]({base_paper_url})
- [Research Graph execution log]({research_graph_execution_log})
- [Devin execution log]({devin_url})"""
    encoded_markdown_data = base64.b64encode(markdown_text.encode("utf-8")).decode(
        "utf-8"
    )
    return encoded_markdown_data


def _encoding_all_data(all_data: dict):
    json_data = json.dumps(all_data, indent=2, ensure_ascii=False)
    encoded_all_data = base64.b64encode(json_data.encode("utf-8")).decode("utf-8")
    return encoded_all_data


def _encoding_experimental_results_data(experimental_results: str):
    encoded_experimental_results = base64.b64encode(
        experimental_results.encode("utf-8")
    ).decode("utf-8")
    return encoded_experimental_results


def github_upload(
    pdf_file_path: str,
    github_owner: str,
    repository_name: str,
    branch_name: str,
    title: str,
    abstract: str,
    base_paper_url: str,
    experimental_results: str,
    devin_url: str,
    all_logs: dict,
) -> bool:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    encoded_pdf_data = _encoded_pdf_file(pdf_file_path)
    paper_url = f"https://github.com/{github_owner}/{repository_name}/blob/{branch_name}/paper/paper.pdf"
    research_graph_execution_log = f"https://github.com/{github_owner}/{repository_name}/blob/{branch_name}/logs/research_graph_log.json"
    encoded_markdown_data = _encoded_markdown_data(
        title,
        abstract,
        paper_url,
        base_paper_url,
        research_graph_execution_log,
        devin_url,
    )
    encoded_experimental_results = _encoding_experimental_results_data(
        experimental_results
    )
    encoded_research_graph_log = _encoding_all_data(all_logs)

    print("Paper Upload")
    paper_path = "paper/paper.pdf"
    response_paper = _request_get_github_content(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        repository_path=paper_path,
    )
    _request_github_file_upload(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        encoded_data=encoded_pdf_data,
        repository_path=paper_path,
        sha=response_paper["sha"] if response_paper is not None else None,
    )

    print("Experiment log upload")
    experiment_log_path = "logs/experiment_log.txt"
    response_experiment_log = _request_get_github_content(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        repository_path=experiment_log_path,
    )
    _request_github_file_upload(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        encoded_data=encoded_experimental_results,
        repository_path=experiment_log_path,
        sha=response_experiment_log["sha"]
        if response_experiment_log is not None
        else None,
    )

    print("README upload")
    readme_path = "README.md"
    response_readme = _request_get_github_content(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        repository_path=readme_path,
    )

    _request_github_file_upload(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        encoded_data=encoded_markdown_data,
        repository_path=readme_path,
        sha=response_readme["sha"] if response_readme is not None else None,
    )

    print("Research Graph log upload")
    research_graph_log_path = "logs/research_graph_log.json"
    response_research_graph_log = _request_get_github_content(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        repository_path=research_graph_log_path,
    )
    _request_github_file_upload(
        headers=headers,
        github_owner=github_owner,
        repository_name=repository_name,
        branch_name=branch_name,
        encoded_data=encoded_research_graph_log,
        repository_path=research_graph_log_path,
        sha=response_research_graph_log["sha"]
        if response_research_graph_log is not None
        else None,
    )
    return True


if __name__ == "__main__":
    branch_name = "devin-b5ffa5ceb46e4562a62da3cef2715742"
    completion = github_upload(
        pdf_file_path="/workspaces/researchgraph/data/paper.pdf",
        github_owner="auto-res2",
        repository_name="auto-research",
        branch_name=branch_name,
        title="Test Title ver2",
        abstract="Test Abstract",
        base_paper_url="https://arxiv.org/abs/2106.01484",
        experimental_results="Test Experimental Results",
        devin_url="https://app.devin.ai/sessions/29ebce5ed6f247a4a5bb0d109ecb2f9b",
        all_logs={"test": "test"},
    )
