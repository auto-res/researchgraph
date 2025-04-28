from jinja2 import Environment
from logging import getLogger
from pydantic import BaseModel

import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from airas.utils.api_client.openai_client import OpenAIClient, OPENAI_MODEL
from airas.publication.html_subgraph.prompt.convert_to_html_prompt import (
    convert_to_html_prompt,
)

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    generated_html_text: str


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
)
def convert_to_html(
    llm_name: OPENAI_MODEL,
    paper_content: dict[str, str],
) -> str:
    openai_client = OpenAIClient()
    data = {
        "sections": [
            {"name": section, "content": paper_content[section]}
            for section in paper_content.keys()
        ]
    }

    env = Environment()
    template = env.from_string(convert_to_html_prompt)
    messages = template.render(data)
    output, cost = openai_client.structured_outputs(
        model_name=llm_name,
        message=messages,
        data_model=LLMOutput,
    )
    if "generated_html_text" in output:
        generated_html_text = output["generated_html_text"]
        return generated_html_text
    else:
        raise ValueError("Error: No response from the model in convert_to_html.")
