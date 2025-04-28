from pydantic import BaseModel
from jinja2 import Environment
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from airas.utils.api_client.google_genai_client import GoogelGenAIClient, VERTEXAI_MODEL
from airas.retrieve.retrieve_code_subgraph.prompt.extract_experimental_info_prompt import (
    extract_experimental_info_prompt,
)


class LLMOutput(BaseModel):
    extract_code: str
    extract_info: str


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
)
def extract_experimental_info(
    model_name: VERTEXAI_MODEL, method_text: str, repository_content_str
) -> tuple[str, str]:
    genai_client = GoogelGenAIClient()
    env = Environment()
    template = env.from_string(extract_experimental_info_prompt)
    data = {
        "method_text": method_text,
        "repository_content_str": repository_content_str,
    }
    messages = template.render(data)
    output, cost = genai_client.structured_outputs(
        model_name=model_name,
        message=messages,
        data_model=LLMOutput,
    )
    if output is None:
        raise RuntimeError("Failed to get response from Vertex AI.")
    else:
        extract_code = output["extract_code"]
        extract_info = output["extract_info"]
        return extract_code, extract_info


if __name__ == "__main__":
    method_text = "This is a test method."
    repository_content_str = "This is a test repository content."
    extract_code, extract_info = extract_experimental_info(
        model_name="gemini-2.0-flash-001",
        method_text=method_text,
        repository_content_str=repository_content_str,
    )
    print("Extracted Code:", extract_code)
    print("Extracted Info:", extract_info)
