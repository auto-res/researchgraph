ai_integrator_v3_summarize_paper_prompt = """
You are an expert research assistant responsible for summarizing a research paper that will serve as the foundation (Research A) for further exploration and integration.

Your task is to generate a structured summary of the given research paper with a focus on:
- **Technical Contributions**: Identify the main research problem and key findings.
- **Methodology**: Describe the techniques, models, or algorithms used.
- **Experimental Setup**: Outline the datasets, benchmarks, and validation methods.
- **Limitations**: Highlight any weaknesses, constraints, or assumptions.
- **Future Research Directions**: Suggest possible extensions or new areas for research.

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
        "future_research_directions": "<Potential areas for extending this research>"
    }
}
```
"""

ai_integrator_v3_select_paper_prompt = """
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
- **Technical Summary:** {{ paper.technical_summary }}
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
