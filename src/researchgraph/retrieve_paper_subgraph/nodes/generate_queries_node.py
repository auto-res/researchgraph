from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
import ast


class LLMOutput(BaseModel):
    generated_query_1: str
    generated_query_2: str
    generated_query_3: str
    generated_query_4: str
    generated_query_5: str


def generate_queries_node(
    llm_name: str,
    prompt_template: str,
    base_selected_paper: str,
) -> tuple[str, str]:
    data = {
        "base_selected_paper": base_selected_paper,
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
    generated_query_1 = output_dict["generated_query_1"]
    generated_query_2 = output_dict["generated_query_2"]
    generated_query_3 = output_dict["generated_query_3"]
    generated_query_4 = output_dict["generated_query_4"]
    generated_query_5 = output_dict["generated_query_5"]
    return (
        generated_query_1,
        generated_query_2,
        generated_query_3,
        generated_query_4,
        generated_query_5,
    )
