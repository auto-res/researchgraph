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
    selected_base_paper_info: str,
) -> list[str]:
    data = {
        "selected_base_paper_info": selected_base_paper_info,
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
    return [
        generated_query_1,
        generated_query_2,
        generated_query_3,
        generated_query_4,
        generated_query_5,
    ]


generate_queries_prompt_add = """
You are an expert research assistant tasked with generating search queries for finding relevant research papers.
Your goal is to create a set of well-structured queries that can be used with a research paper search API 
to retrieve papers that are conceptually or methodologically related to a given foundational paper (Research A).

**Research A (Base Paper):**
- **Title:** {{ selected_base_paper_info.title }}
- **Authors:** {{ selected_base_paper_info.authors | join(', ') }}
- **Publication Date:** {{ selected_base_paper_info.publication_date }}
- **Journal/Conference:** {{ selected_base_paper_info.journal }}
- **DOI:** {{ selected_base_paper_info.doi }}
- **Main Contributions:** {{ selected_base_paper_info.main_contributions }}
- **Methodology:** {{ selected_base_paper_info.methodology }}
- **Experimental Setup:** {{ selected_base_paper_info.experimental_setup }}
- **Limitations:** {{ selected_base_paper_info.limitations }}
- **Future Research Directions:** {{ selected_base_paper_info.future_research_directions }}
---

**Instructions (Important!):**
1. Analyze the provided Research A details.
2. Generate exactly 5 **short** search queries (ideally 1-5 words each).
3. **Output must be a valid Python dictionary literal that can be parsed by `ast.literal_eval`.**
    - The dictionary must have exactly five keys: `"generated_query_1"`, `"generated_query_2"`, `"generated_query_3"`, `"generated_query_4"`, `"generated_query_5"`.
    - Each key's value must be a string representing a search query.
    - Example:
    ```python
    {"generated_query_1": "robust matrix completion",
    "generated_query_2": "low-rank data recovery",
    "generated_query_3": "sparse optimization methods",
    "generated_query_4": "convex relaxation techniques",
    "generated_query_5": "compressed sensing applications"}
    ```
4. **No extra text, no triple backticks, no markdown.** Output ONLY the dictionary.
5. If you are unsure, only output valid Python dictionary syntax with double quotes for strings.

**Output Format Example**:
{"generated_query_1": "robust matrix completion", "generated_query_2": "low-rank data recovery", "generated_query_3": "sparse optimization methods", "generated_query_4": "convex relaxation techniques", "generated_query_5": "compressed sensing applications"}

Now, output the dictionary literal in one single line (no additional commentary):
"""
