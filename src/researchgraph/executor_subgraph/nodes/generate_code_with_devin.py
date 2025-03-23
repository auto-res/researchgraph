import os
from researchgraph.utils.api_request_handler import fetch_api_data, retry_request

DEVIN_API_KEY = os.getenv("DEVIN_API_KEY")


def _request_create_session(
    headers: dict,
    repository_url: str,
    new_method: str,
    experiment_code: str,
):
    url = "https://api.devin.ai/v1/sessions"
    data = {
        "prompt": f"""
The “New Method” and “Experiment Code” sections contain ideas for new machine learning research and the code associated with those ideas. 
The name of the new branch should be exactly the same as the session ID, starting with “devin-”.
Please follow the “Rules” section to create an experimental script to conduct this research.
Please create code that can run on NVIDIA Tesla T4 · 16 GB VRAM.
# Rules
- Create and implement a new branch in the repository given in “Repository URL”. 
- Experimental scripts should be given a simple test run to make sure they work. The test run should not be too long.
- Install and use the necessary python packages as needed.
- Please also list the python packages required for the experiment in the requirements.txt file.
- The roles of directories and scripts are listed below. Follow the roles to complete your implementation.
- .github/workflows/run_experiment.yml...Under no circumstances should the contents or folder structure of the “run_experiment.yml” file be altered. This rule must be observed.
- config...If you want to set parameters for running the experiment, place the file that completes the parameters under this directory.
- data...This directory is used to store data used for model training and evaluation.
- models...This directory is used to store pre-trained and trained models.
- paper...Do not change anything in this directory.
- src
    - train.py...Scripts for training models. Implement the code to train the models.
    - evaluate.py...Script to evaluate the model. Implement the code to evaluate the model.
    - preprocess.py...Script for preprocessing data. Implement the code necessary for data preprocessing.
    - main.py...Scripts for running the experiment, using train.py, evaluate.py, and preprocess.py to implement the entire process from model training to evaluation.
                The script should be implemented in such a way that the results of the experiment can be seen in detail on the standard output.
- requirements.txt...Please list the python packages required to run the model.        

# New Method
----------------------------------------
{new_method}
----------------------------------------
# Experiment Code
----------------------------------------
{experiment_code}
----------------------------------------
# Repository URL
{repository_url}
""",
        "idempotent": True,
    }
    return retry_request(fetch_api_data, url, headers=headers, data=data, method="POST")


# def _request_devin_output(headers, session_id):
#     url = f"https://api.devin.ai/v1/session/{session_id}"

#     def should_retry(response):
#         # Describe the process so that it is True if you want to retry
#         return response.get("status_enum") not in ["blocked", "stopped"]

#     return retry_request(
#         fetch_api_data,
#         url,
#         headers=headers,
#         method="GET",
#         check_condition=should_retry,
#     )


def generate_code_with_devin(
    headers: dict,
    github_owner: str,
    repository_name: str,
    new_method: str,
    experiment_code: str,
) -> tuple[str | None, str | None]:
    repository_url = f"https://github.com/{github_owner}/{repository_name}"
    response = _request_create_session(
        headers=headers,
        repository_url=repository_url,
        new_method=new_method,
        experiment_code=experiment_code,
    )
    if response:
        print("Successfully created Devin session.")
        experiment_session_id = response["session_id"]
        experiment_devin_url = response["url"]
        print("Devin URL: ", experiment_devin_url)
        # NOTE: Devin takes a while to complete its execution, so it does not send unnecessary requests.
        # time.sleep(120)
        # response = _request_devin_output(headers, session_id)
        # print(response)
        # branch_name = session_id
        return (
            experiment_session_id,
            experiment_devin_url,
        )
    else:
        print("Failed to create Devin session.")
        return (
            None,
            None,
        )
