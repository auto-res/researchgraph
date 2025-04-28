import requests
from jinja2 import Environment
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from airas.utils.api_client.openai_client import OpenAIClient, OPENAI_MODEL
from airas.create.create_method_subgraph.prompt.generator_node_prompt import (
    generator_node_prompt,
)
from logging import getLogger

logger = getLogger(__name__)


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
)
def generator_node(
    llm_name: OPENAI_MODEL,
    base_method_text: str,
    add_method_text_list: list[str],
) -> str:
    openai_client = OpenAIClient()

    add_method_text = "---".join(add_method_text_list)
    env = Environment()
    template = env.from_string(generator_node_prompt)
    data = {
        "base_method_text": base_method_text,
        "add_method_text": add_method_text,
    }
    messages = template.render(data)
    output, cost = openai_client.generate(
        model_name=llm_name,
        message=messages,
    )
    if output is None:
        raise ValueError("Error: No response from the model in generator_node.")
    return output
