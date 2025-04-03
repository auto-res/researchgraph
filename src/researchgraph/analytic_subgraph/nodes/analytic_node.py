import json
from pydantic import BaseModel
from researchgraph.utils.openai_client import openai_client
from logging import getLogger

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    analysis_report: str
    analysis_results: bool


def analytic_node(
    llm_name: str,
    new_method: str,
    verification_policy: str,
    experiment_code: str,
    output_text_data: str,
) -> tuple[str | None, bool | None]:
    prompt = f"""\
You are an expert in machine learning research.
- In order to demonstrate the usefulness of the new method described in "New Method",
you conducted an experiment using the policy described in "Verification Policy" . The experimental code was based on the code described in "Experiment Code" . The experimental results are described in "Experimental results" .
- Please summarize the results in detail as an "analysis_report", based on the experimental setup and outcomes. Also, include whether the new method demonstrates a clear advantage.
- Output "analysis_results" as True if the new method shows sufficient superiority, or False otherwise.
# New Method
---------------------------------
{new_method}
---------------------------------
# Verification Policy
---------------------------------
{verification_policy}
---------------------------------
# Experiment Code
---------------------------------
{experiment_code}
---------------------------------
# Experimental results
---------------------------------
{output_text_data}
---------------------------------"""
    messages = [
        {"role": "user", "content": f"{prompt}"},
    ]
    response = openai_client(model_name=llm_name, message=messages)
    if response is None:
        logger.error("Error: No response from LLM.")
        return None, None
    else:
        response = json.loads(response)
        analysis_report = response.get("analysis_report")
        analysis_results = response.get("analysis_results")

        if analysis_report is None:
            logger.error("No 'analysis_report' found in the response.")
        if analysis_results is None:
            logger.error("No 'analysis_results' found in the response.")

        return analysis_report, analysis_results
