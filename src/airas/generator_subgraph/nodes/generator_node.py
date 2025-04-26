from jinja2 import Environment
from airas.utils.openai_client import openai_client
from airas.generator_subgraph.prompt.generator_node_prompt import (
    generator_node_prompt,
)
from logging import getLogger

logger = getLogger(__name__)


def generator_node(
    llm_name: str,
    base_method_text: str,
    add_method_text_list: list[str],
) -> str:
    add_method_text = "---".join(add_method_text_list)
    env = Environment()
    template = env.from_string(generator_node_prompt)
    data = {
        "base_method_text": base_method_text,
        "add_method_text": add_method_text,
    }
    prompt = template.render(data)
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai_client(model_name=llm_name, message=messages)
    if response is None:
        raise ValueError("Error: No response from the model in generator_node.")
    return response
