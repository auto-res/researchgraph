import json
from jinja2 import Environment
from logging import getLogger
from pydantic import BaseModel

from airas.utils.openai_client import openai_client
from airas.html_subgraph.prompt.convert_to_html_prompt import (
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
    raw_response = openai_client(llm_name, message=messages, data_model=LLMOutput)
    if not raw_response:
        raise ValueError("Error: No response from the model in convert_to_html.")

    try:
        response = json.loads(raw_response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        raise ValueError("Error: Invalid JSON response from model in convert_to_html.")

    html = response.get("generated_html_text", "")
    if not html:
        raise ValueError("Error: Empty HTML content from model in convert_to_html.")

    return html
