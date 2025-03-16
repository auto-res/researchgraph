import os
import zipfile

from researchgraph.utils.api_request_handler import fetch_api_data, retry_request

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


def _request_github_actions_artifacts(
    headers: dict, github_owner: str, repository_name: str
):
    url = f"https://api.github.com/repos/{github_owner}/{repository_name}/actions/artifacts"
    return retry_request(fetch_api_data, url, headers=headers, method="GET")


def _parse_artifacts_info(artifacts_infos: dict, workflow_run_id: str):
    artifacts_redirect_url_dict = {}
    for artifacts_info in artifacts_infos["artifacts"]:
        if artifacts_info["workflow_run"]["id"] == workflow_run_id:
            artifacts_redirect_url_dict[artifacts_info["name"]] = artifacts_info[
                "archive_download_url"
            ]
    return artifacts_redirect_url_dict


def _request_download_artifacts(
    headers: dict, artifacts_redirect_url_dict: dict, iteration_save_dir: str
):
    for key, url in artifacts_redirect_url_dict.items():
        response = retry_request(
            fetch_api_data, url, headers=headers, method="GET", stream=True
        )
        _zip_to_txt(response, iteration_save_dir, key)


def _zip_to_txt(response, iteration_save_dir, key):
    zip_file_path = os.path.join(iteration_save_dir, f"{key}.zip")
    with open(zip_file_path, "wb") as f:
        f.write(response)
    print(f"Downloaded artifact saved to: {zip_file_path}")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(iteration_save_dir)
    print(f"Extracted artifact to: {iteration_save_dir}")
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)
        print(f"ZIP file deleted: {zip_file_path}")


def retrieve_github_actions_artifacts(
    github_owner,
    repository_name,
    workflow_run_id,
    save_dir,
    fix_iteration_count,
) -> tuple[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PERSONAL_ACCESS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    iteration_save_dir = save_dir + f"/iteration_{fix_iteration_count}"
    os.makedirs(iteration_save_dir, exist_ok=True)
    response_artifacts_infos = _request_github_actions_artifacts(
        headers, github_owner, repository_name
    )
    if response_artifacts_infos:
        print("Successfully retrieved artifacts information.")
    else:
        print("Failure to retrieve artifacts information.")
    get_artifacts_redirect_url_dict = _parse_artifacts_info(
        response_artifacts_infos, workflow_run_id
    )
    _request_download_artifacts(
        headers, get_artifacts_redirect_url_dict, iteration_save_dir
    )
    with open(os.path.join(iteration_save_dir, "output.txt"), "r") as f:
        output_text_data = f.read()
    with open(os.path.join(iteration_save_dir, "error.txt"), "r") as f:
        error_text_data = f.read()
    return (
        output_text_data,
        error_text_data,
    )


# if __name__ == "__main__":
#     graph_builder = StateGraph(State)
#     graph_builder.add_node(
#         "retrieve_github_actions_artifacts",
#         RetrieveGithubActionsArtifactsNode(
#             input_key=[
#                 "github_owner",
#                 "repository_name",
#                 "workflow_run_id",
#                 "save_dir",
#                 "num_iterations",
#             ],
#             output_key=["output_file_path", "error_file_path"],
#         ),
#     )
#     graph_builder.add_edge(START, "retrieve_github_actions_artifacts")
#     graph_builder.add_edge("retrieve_github_actions_artifacts", END)
#     graph = graph_builder.compile()
#     state = {
#         "github_owner": "auto-res",
#         "repository_name": "experimental-script",
#         "workflow_run_id": 13055964079,
#         "save_dir": "/workspaces/researchgraph/data",
#         "num_iterations": 1,
#     }
#     graph.invoke(state, debug=True)
