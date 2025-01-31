ai_integrator_v3_generate_queries_prompt = """
You are an expert research assistant tasked with generating search queries for finding relevant research papers.
Your goal is to create a set of well-structured queries that can be used with a research paper search API 
to retrieve papers that are conceptually or methodologically related to a given foundational paper (Research A).

**Research A (Base Paper):**
- **Title:** {{ base_selected_paper.title }}
- **Technical Summary:** {{ base_selected_paper.technical_summary }}

---

**Instructions (Important!):**
1. Analyze the provided Research A details.
2. Generate a set of 5-10 **short** search queries (ideally 1-5 words each).
3. **Output must be a valid Python dictionary literal that can be parsed by `ast.literal_eval`.
    - The dictionary must have exactly one key: "queries"
    - That key's value must be a list of one or more strings.
    - Example: {"queries": ["Query 1", "Query 2"]}.
4. **No extra text, no triple backticks, no markdown.** Output ONLY the dictionary.
5. If you are unsure, only output valid Python dictionary syntax with double quotes for strings.

**Output Format Example**:
{"queries": ["robust matrix completion", "low-rank data recovery"]}

Now, output the dictionary literal in one single line(no additional commentary):
"""

ai_integrator_v3_summarize_paper_prompt = """
You are an expert research assistant responsible for summarizing a research paper that will be integrated into an existing body of knowledge (Research B).

Your task is to generate a structured summary of this research paper with a focus on:
- **Technical Contributions**: Identify the main research problem and key findings.
- **Methodology**: Describe the techniques, models, or algorithms used.
- **Experimental Setup**: Outline the datasets, benchmarks, and validation methods.
- **Limitations**: Highlight any weaknesses, constraints, or assumptions.
- **Relevance to Existing Research**: Explain how this paper contributes to or extends prior work (e.g., Research A).
- **Potential Applications**: Identify practical applications of the findings.

Below is the full text of the research paper:

```
{{ paper_text }}
```

## **Instructions:**
1. Analyze the paper based on the categories listed below.
2. Your response **must be a valid JSON object** that can be directly parsed using `json.loads()`.
3. Do not include any extra text, explanations, or formatting outside of the JSON object.
4. **If a field has no available information, set its value to `"Not mentioned"` instead of leaving it empty.**
5. Ensure that the JSON format is correct, including the use of **double quotes (`"`) for all keys and values.**

## **Output Format (JSON)**:
```json
{
    "technical_summary": {
        "main_contributions": "<Concise description of the main research problem and contributions>",
        "methodology": "<Brief explanation of the key techniques, models, or algorithms>",
        "experimental_setup": "<Description of datasets, benchmarks, and validation methods>",
        "limitations": "<Summary of weaknesses, constraints, or assumptions>",
        "relevance_to_existing_research": "<Explanation of how this paper contributes to or extends Research A>",
        "potential_applications": "<Practical applications of this research>"
    }
}
```
"""


ai_integrator_v3_select_paper_prompt = """
You are an expert research assistant tasked with selecting the most relevant and high-quality research paper from a list of candidate papers. 
Your goal is to identify a paper (Research B) that can be effectively synthesized with a given foundational paper (Research A) to create a novel and non-trivial research direction.

**Research A (Base Paper):**
- **Title:** {{ base_selected_paper.title }}
- **Authors:** {{ base_selected_paper.authors | join(', ') }}
- **Publication Date:** {{ base_selected_paper.publication_date }}
- **Journal/Conference:** {{ base_selected_paper.journal }}
- **DOI:** {{ base_selected_paper.doi }}
- **arXiv URL:** {{ base_selected_paper.arxiv_url }}
- **GitHub URL:** {{ base_selected_paper.github_urls }}
- **Technical Summary:** {{ base_selected_paper.technical_summary }}

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
