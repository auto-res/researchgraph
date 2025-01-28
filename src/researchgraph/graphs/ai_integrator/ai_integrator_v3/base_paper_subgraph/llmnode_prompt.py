ai_integrator_v3_select_paper_prompt = """
You are an expert research assistant tasked with selecting the most relevant and high-quality paper from a list of research papers. 
Your goal is to identify the paper that best aligns with the research topic and provides the most value as a foundational base for further research.

Below is a list of papers with their details:

{% for paper in base_paper_data %}
**Paper {{ loop.index }}:**
- **Title:** {{ paper.title }}
- **Authors:** {{ paper.authors | join(', ') }}
- **Publication Date:** {{ paper.publication_date }}
- **Journal/Conference:** {{ paper.journal }}
- **DOI:** {{ paper.doi }}
- **arXiv URL:** {{ paper.arxiv_url }}
- **GitHub URL:** {{ paper.github_url }}
- **Abstract/Text Excerpt:** {{ paper.paper_text[:500] }}... (truncated)
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
4. Return the selected paper **exactly as it appears in the input data**, without modifying its structure or content.
5. Provide your response in the following JSON format:
```json
{
  "base_selected_paper": {
    "title": "Title of the selected paper",
    "authors": ["Author 1", "Author 2", ...],
    "publication_date": "Publication date of the selected paper",
    "journal": "Journal or conference name",
    "doi": "DOI of the selected paper",
    "arxiv_url": "arXiv URL of the selected paper",
    "github_url": "GitHub URL of the selected paper (if available)",
    "paper_text": "Full text or excerpt of the selected paper"
  }
}
"""