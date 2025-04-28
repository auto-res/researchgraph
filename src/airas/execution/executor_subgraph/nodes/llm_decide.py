import requests
from pydantic import BaseModel
from jinja2 import Environment
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from airas.utils.api_client.openai_client import OpenAIClient, OPENAI_MODEL
from airas.execution.executor_subgraph.prompt.llm_decide import (
    llm_decide_prompt,
)
from logging import getLogger

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    judgment_result: bool


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
)
def llm_decide(
    llm_name: OPENAI_MODEL,
    output_text_data: str,
    error_text_data: str,
    prompt_template: str = llm_decide_prompt,
) -> bool | None:
    openai_client = OpenAIClient()

    data = {"output_text_data": output_text_data, "error_text_data": error_text_data}

    env = Environment()
    template = env.from_string(prompt_template)
    messages = template.render(data)
    output, cost = openai_client.structured_outputs(
        model_name=llm_name,
        message=messages,
        data_model=LLMOutput,
    )
    if "judgment_result" in output:
        judgment_result = output["judgment_result"]
        return judgment_result
    else:
        raise ValueError("Error: No response from LLM in llm_decide.")


if __name__ == "__main__":
    llm_name = "gpt-4o-mini-2024-07-18"
    output_text_data = "No error"
    error_text_data = "Error"
    result = llm_decide(llm_name, output_text_data, error_text_data)
    print(result)
