import json
from jinja2 import Environment
from logging import getLogger
from pydantic import BaseModel

from researchgraph.utils.openai_client import openai_client
from researchgraph.html_subgraph.prompt.convert_to_html_prompt import (
    convert_to_html_prompt,
)

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    generated_html_text: str


def convert_to_html(
    llm_name: str,
    prompt_template: str,
    paper_content: dict[str, str],
) -> str:
    data = {
        "sections": [
            {"name": section, "content": paper_content[section]}
            for section in paper_content.keys()
        ]
    }

    env = Environment()
    template = env.from_string(convert_to_html_prompt)
    prompt = template.render(data)

    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai_client(llm_name, message=messages, data_model=LLMOutput)
    if response is None:
        raise ValueError("Error: No response from the model in convert_to_html.")
    response = json.loads(response)
    return response["generated_html_text"]