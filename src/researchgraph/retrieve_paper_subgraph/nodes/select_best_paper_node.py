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
    selected_base_paper_info=None,
) -> tuple[str, str]:
    if selected_base_paper_info is None:
        data = {
            "candidate_papers": candidate_papers,
        }
    else:
        data = {
            "candidate_papers": candidate_papers,
            "selected_base_paper": selected_base_paper_info,
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


select_base_paper_prompt = """
You are an expert research assistant tasked with selecting the most relevant and high-quality paper from a list of research papers. 
Your goal is to identify the paper that best aligns with the research topic and provides the most value as a foundational base for further research.

Below is a list of papers with their details:

{% for paper in candidate_papers %}
**Paper (ID: {{ paper.arxiv_id }})**
- **Title:** {{ paper.title }}
- **Authors:** {{ paper.authors | join(', ') }}
- **Publication Date:** {{ paper.publication_date }}
- **Journal/Conference:** {{ paper.journal }}
- **DOI:** {{ paper.doi }}
- **arXiv URL:** {{ paper.arxiv_url }}
- **GitHub URL:** {{ paper.github_urls }}
- **Main Contributions:** {{ paper.main_contributions }}
- **Methodology:** {{ paper.methodology }}
- **Experimental Setup:** {{ paper.experimental_setup }}
- **Limitations:** {{ paper.limitations }}
- **Future Research Directions:** {{ paper.future_research_directions }}
{% endfor %}

**Selection Criteria:**
1. **Relevance:** The paper should directly address the research topic or provide foundational insights. Exclude papers that are only tangentially related.
2. **Quality:** Prioritize papers published in reputable journals or conferences. Evaluate the clarity of the abstract and the soundness of the methodology.
3. **Future Potential:** Consider whether the paper introduces innovative methods or theories that could be foundational for future research.
4. **Code Availability:** Papers with accessible and functional GitHub repositories are preferred. Evaluate the quality of the code and documentation if available.

**Instructions:**
1. Carefully review the details of each paper.
2. Evaluate each paper based on the selection criteria above.
3. Select the **single most relevant and high-quality paper** that best serves as a foundational base for further research.
4. Return the **arxiv_id** of the selected paper.
5. Provide your response in the following JSON format:
```json
{
  "selected_arxiv_id": "{arxiv_id}"
}
```

"""


select_add_paper_prompt = """
You are an expert research assistant tasked with selecting the most relevant and high-quality research paper from a list of candidate papers. 
Your goal is to identify a paper (Research B) that can be effectively synthesized with a given foundational paper (Research A) to create a novel and non-trivial research direction.

**Research A (Base Paper):**
- **Title:** {{ selected_base_paper.title }}
- **Authors:** {{ selected_base_paper.authors | join(', ') }}
- **Publication Date:** {{ selected_base_paper.publication_date }}
- **Journal/Conference:** {{ selected_base_paper.journal }}
- **DOI:** {{ selected_base_paper.doi }}
- **arXiv URL:** {{ selected_base_paper.arxiv_url }}
- **GitHub URL:** {{ selected_base_paper.github_urls }}
- **Main Contributions:** {{ selected_base_paper.main_contributions }}
- **Methodology:** {{ selected_base_paper.methodology }}
- **Experimental Setup:** {{ selected_base_paper.experimental_setup }}
- **Limitations:** {{ selected_base_paper.limitations }}
- **Future Research Directions:** {{ selected_base_paper.future_research_directions }}

Below is a list of candidate papers (Research B candidates):

{% for paper in candidate_papers %}
**Paper (ID: {{ paper.arxiv_id }})**
- **Title:** {{ paper.title }}
- **Authors:** {{ paper.authors | join(', ') }}
- **Publication Date:** {{ paper.publication_date }}
- **Journal/Conference:** {{ paper.journal }}
- **DOI:** {{ paper.doi }}
- **arXiv URL:** {{ paper.arxiv_url }}
- **GitHub URL:** {{ paper.github_urls }}
- **Technical Summary:** {{ paper.technical_summary }}
{% endfor %}

**Selection Criteria:**
1. **Synthesis Potential:** The selected paper should complement Research A in a way that leads to a novel and non-trivial research direction when combined.
2. **Relevance:** The paper should have a meaningful conceptual or methodological connection with Research A, avoiding those that are only loosely related.
3. **Quality:** Prioritize papers published in reputable journals or conferences. Evaluate the clarity of the abstract and the soundness of the methodology.
4. **Future Potential:** Consider whether the paper introduces innovative methods or theories that could provide a strong foundation for new research.
5. **Code Availability:** Papers with accessible and functional GitHub repositories are preferred. Evaluate the quality of the code and documentation if available.

**Instructions:**
1. Carefully review the details of each candidate paper.
2. Evaluate each paper based on the selection criteria above.
3. Select the **single most relevant and high-quality paper (Research B)** that, when synthesized with Research A, enables a non-trivial and novel research direction.
4. Return the **arxiv_id** of the selected paper.
5. Provide your response in the following JSON format:
```json
{
"selected_arxiv_id": "{arxiv_id}"
}
```
"""
