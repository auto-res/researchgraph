import json
from pydantic import BaseModel
from jinja2 import Environment

from airas.utils.openai_client import openai_client
from airas.analytic_subgraph.prompt.analytic_node_prompt import (
    analytic_node_prompt,
)
from logging import getLogger

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    analysis_report: str


def analytic_node(
    llm_name: str,
    new_method: str,
    verification_policy: str,
    experiment_code: str,
    output_text_data: str,
) -> str | None:
    env = Environment()
    template = env.from_string(analytic_node_prompt)
    data = {
        "new_method": new_method,
        "verification_policy": verification_policy,
        "experiment_code": experiment_code,
        "output_text_data": output_text_data,
    }
    prompt = template.render(data)

    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai_client(
        model_name=llm_name, message=messages, data_model=LLMOutput
    )
    if response is None:
        logger.error("Error: No response from LLM.")
        return None
    else:
        response = json.loads(response)
        analysis_report = response.get("analysis_report")

        if analysis_report is None:
            logger.error("No 'analysis_report' found in the response.")

        return analysis_report
