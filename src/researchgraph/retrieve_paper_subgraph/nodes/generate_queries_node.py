from pydantic import BaseModel
from openai import OpenAI
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
    queries: list, 
) -> list[str]:
    data = {
        "selected_base_paper_info": selected_base_paper_info,
        "queries": queries
    }

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    client = OpenAI()
    response = client.chat.completions.create(
        model=llm_name,
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        temperature=0.2, 
        response_format={"type": "json_object"},
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

User Query: {{ queries }}

**Instructions (Important!):**
1. Analyze the provided Research A details.
2. Maintain topic relevance**: The generated queries should closely relate to the user's original query.
3. Generate exactly **short** search queries (ideally 1-3 words each) that are commonly user in AI research.
4. **No applied domains**: Do not generate queries related to industry applications, business applications, healthcare, finance, medical research, or market trends.
5. **Instead, focus on core theoretical concepts, mathematical principles, and model advancements** rather than how they are used in real-world industries.

**Format**
1. **Output must be a valid Python dictionary literal that can be parsed by `ast.literal_eval`.**
    - The dictionary must have exactly five keys: `"generated_query_1"`, `"generated_query_2"`, `"generated_query_3"`, `"generated_query_4"`, `"generated_query_5"`.
    - Each key's value must be a string representing a search query.
    - Example:
    ```python
    {
    "generated_query_1": "<query_1>", 
    "generated_query_2": "<query_2>", 
    "generated_query_3": "<query_3>", 
    "generated_query_4": "<query_4>", 
    "generated_query_5": "<query_5>", 
    }
    ```
2. **No extra text, no triple backticks, no markdown.** Output ONLY the dictionary.
3. If you are unsure, only output valid Python dictionary syntax with double quotes for strings.

Now, output the dictionary literal in one single line (no additional commentary):
"""
