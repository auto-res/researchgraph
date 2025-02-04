from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
import ast


class LLMOutput(BaseModel):
    new_method_text: str
    new_method_code: str


def execute_llm(
    llm_name: str,
    prompt_template: str,
    objective: str,
    base_method_code: str,
    base_method_text: str,
    add_method_code: str,
    add_method_text: str,
) -> tuple[str, str]:
    data = {
        "objective": objective,
        "base_method_code": base_method_code,
        "base_method_text": base_method_text,
        "add_method_code": add_method_code,
        "add_method_text": add_method_text,
    }

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    response = completion(
        model=llm_name,
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        response_format=LLMOutput,
    )
    output = response.choices[0].message.content
    output_dict = ast.literal_eval(output)
    new_method_text = output_dict["new_method_text"]
    new_method_code = output_dict["new_method_code"]
    return new_method_text, new_method_code
