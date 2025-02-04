from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
import ast


class LLMOutput(BaseModel):
    main_contributions: str
    methodology: str
    experimental_setup: str
    limitations: str
    future_research_directions: str


def summarize_paper_node(
    llm_name: str,
    prompt_template: str,
    paper_text: str,
) -> tuple[str, str]:
    data = {
        "paper_text": paper_text,
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
    main_contributions = output_dict["main_contributions"]
    methodology = output_dict["methodology"]
    experimental_setup = output_dict["experimental_setup"]
    limitations = output_dict["limitations"]
    future_research_directions = output_dict["future_research_directions"]
    return (
        main_contributions,
        methodology,
        experimental_setup,
        limitations,
        future_research_directions,
    )
