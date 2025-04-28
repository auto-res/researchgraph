from pydantic import BaseModel
from jinja2 import Environment
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from airas.utils.api_client.openai_client import OpenAIClient, OPENAI_MODEL


from logging import getLogger

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    generated_query_1: str
    generated_query_2: str
    generated_query_3: str
    generated_query_4: str
    generated_query_5: str


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
)
def generate_queries_node(
    llm_name: OPENAI_MODEL,
    prompt_template: str,
    selected_base_paper_info: str,
    queries: list,
) -> list[str]:
    data = {"selected_base_paper_info": selected_base_paper_info, "queries": queries}

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    output, cost = OpenAIClient().structured_outputs(
        llm_name, message=prompt, data_model=LLMOutput
    )
    if output is None:
        raise ValueError("Error: No response from the model in generate_queries_node.")
    else:
        return [
            output["generated_query_1"],
            output["generated_query_2"],
            output["generated_query_3"],
            output["generated_query_4"],
            output["generated_query_5"],
        ]
