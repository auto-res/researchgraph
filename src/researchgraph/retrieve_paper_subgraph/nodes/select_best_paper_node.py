from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
import ast


class LLMOutput(BaseModel):
    selected_arxiv_id: str


def select_best_paper_node(
    llm_name: str,
    prompt_template: str,
    candidate_papers,
    base_selected_paper=None,
) -> tuple[str, str]:
    if base_selected_paper is None:
        data = {
            "candidate_papers": candidate_papers,
        }
    else:
        data = {
            "candidate_papers": candidate_papers,
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
    selected_arxiv_id = output_dict["selected_arxiv_id"]
    return selected_arxiv_id
