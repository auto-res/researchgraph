from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
import ast
from typing import List, Optional

from researchgraph.retrieve_paper_subgraph.nodes.recursive_paper_search import (
    CandidatePaperInfo,
)


class LLMOutput(BaseModel):
    selected_arxiv_id: str
    reasoning: str


def select_best_paper_with_context(
    llm_name: str,
    candidate_papers: List[CandidatePaperInfo],
    learnings: List[str],
    base_paper: Optional[CandidatePaperInfo] = None,
) -> CandidatePaperInfo:
    """
    コンテキスト（学習内容）を考慮して最適な論文を選択する

    Args:
        llm_name: 使用するLLMの名前
        candidate_papers: 候補論文のリスト
        learnings: 探索で得られた学習内容のリスト
        base_paper: ベース論文（追加論文選択時のみ使用）

    Returns:
        選択された論文情報
    """
    print(f"\n{'='*50}")
    print(f"SELECTING BEST PAPER WITH CONTEXT")
    print(f"{'='*50}")

    if base_paper:
        print(f"Selection mode: Additional paper (with base paper)")
        print(f"Base paper: {base_paper.title} (ID: {base_paper.arxiv_id})")
    else:
        print(f"Selection mode: Base paper")

    print(f"Number of candidate papers: {len(candidate_papers)}")
    print(f"Number of learnings from research: {len(learnings)}")

    if not candidate_papers:
        print("Warning: No candidate papers provided. Creating a default paper.")
        # 候補論文がない場合はデフォルトの論文を作成
        default_paper = CandidatePaperInfo(
            arxiv_id="default-paper-id",
            arxiv_url="https://arxiv.org/abs/default",
            title="No Papers Found for the Given Query",
            authors=["System"],
            published_date="",
            summary="No papers were found that match the search criteria with GitHub repositories.",
            github_url="",
            main_contributions="",
            methodology="",
            experimental_setup="",
            limitations="",
            future_research_directions=""
        )
        return default_paper

    # 学習内容をテキストに変換
    print("Converting learnings to text format...")
    learnings_text = "\n".join([f"- {learning}" for learning in learnings])

    # 候補論文の情報をテキストに変換
    print("Converting paper information to text format...")
    papers_text = ""
    print("Candidate papers:")
    for i, paper in enumerate(candidate_papers):
        print(f"  {i+1}. {paper.title} (ID: {paper.arxiv_id})")
        papers_text += f"""
**Paper {i+1} (ID: {paper.arxiv_id})**
- **Title:** {paper.title}
- **Authors:** {', '.join(paper.authors)}
- **Publication Date:** {paper.published_date}
- **Journal/Conference:** {paper.journal}
- **DOI:** {paper.doi}
- **arXiv URL:** {paper.arxiv_url}
- **GitHub URL:** {paper.github_url}
- **Main Contributions:** {paper.main_contributions}
- **Methodology:** {paper.methodology}
- **Experimental Setup:** {paper.experimental_setup}
- **Limitations:** {paper.limitations}
- **Future Research Directions:** {paper.future_research_directions}

"""

    # プロンプトテンプレートを選択
    if base_paper:
        # 追加論文選択用のプロンプト
        prompt_template = """
You are an expert research assistant tasked with selecting the most relevant and high-quality research paper from a list of candidate papers.
Your goal is to identify a paper (Research B) that can be effectively synthesized with a given foundational paper (Research A) to create a novel and non-trivial research direction.

**Research A (Base Paper):**
- **Title:** {{ base_paper.title }}
- **Authors:** {{ base_paper.authors | join(', ') }}
- **Publication Date:** {{ base_paper.published_date }}
- **Journal/Conference:** {{ base_paper.journal }}
- **DOI:** {{ base_paper.doi }}
- **arXiv URL:** {{ base_paper.arxiv_url }}
- **GitHub URL:** {{ base_paper.github_url }}
- **Main Contributions:** {{ base_paper.main_contributions }}
- **Methodology:** {{ base_paper.methodology }}
- **Experimental Setup:** {{ base_paper.experimental_setup }}
- **Limitations:** {{ base_paper.limitations }}
- **Future Research Directions:** {{ base_paper.future_research_directions }}

**Additional Context from Research:**
{{ learnings_text }}

Below is a list of candidate papers (Research B candidates):

{{ papers_text }}

**Selection Criteria:**
1. **Synthesis Potential:** The selected paper should complement Research A in a way that leads to a novel and non-trivial research direction when combined.
2. **Relevance:** The paper should have a meaningful conceptual or methodological connection with Research A, avoiding those that are only loosely related.
3. **Quality:** Prioritize papers published in reputable journals or conferences. Evaluate the clarity of the abstract and the soundness of the methodology.
4. **Future Potential:** Consider whether the paper introduces innovative methods or theories that could provide a strong foundation for new research.
5. **Code Availability:** Papers with accessible and functional GitHub repositories are preferred. Evaluate the quality of the code and documentation if available.
6. **Context Alignment:** Consider how well the paper aligns with the additional context from research.
7. **Avoid domain-specific applications:** Do not select papers that focus on specific industries, healthcare, finance, biology, or business applications. Instead, prioritize fundamental advancements in AI/ML theory.

**Instructions:**
1. Carefully review the details of each candidate paper.
2. Evaluate each paper based on the selection criteria above.
3. Select the **single most relevant and high-quality paper (Research B)** that, when synthesized with Research A, enables a non-trivial and novel research direction.
4. Return the **arxiv_id** of the selected paper and your reasoning.
5. Provide your response in the following JSON format:
```json
{
"selected_arxiv_id": "{arxiv_id}",
"reasoning": "{your detailed reasoning for selecting this paper}"
}
```
"""
    else:
        # ベース論文選択用のプロンプト
        prompt_template = """
You are an expert research assistant tasked with selecting the most relevant and high-quality paper from a list of research papers.
Your goal is to identify the paper that best aligns with the research topic and provides the most value as a foundational base for further research.

**Context from Research:**
{{ learnings_text }}

Below is a list of papers with their details:

{{ papers_text }}

**Selection Criteria:**
1. **Relevance:** The paper should directly address the research topic or provide foundational insights. Exclude papers that are only tangentially related.
2. **Quality:** Prioritize papers published in reputable journals or conferences. Evaluate the clarity of the abstract and the soundness of the methodology.
3. **Future Potential:** Consider whether the paper introduces innovative methods or theories that could be foundational for future research.
4. **Code Availability:** Papers with accessible and functional GitHub repositories are preferred. Evaluate the quality of the code and documentation if available.
5. **Context Alignment:** Consider how well the paper aligns with the context from research.

**Instructions:**
1. Carefully review the details of each paper.
2. Evaluate each paper based on the selection criteria above.
3. Select the **single most relevant and high-quality paper** that best serves as a foundational base for further research.
4. Return the **arxiv_id** of the selected paper and your reasoning.
5. Provide your response in the following JSON format:
```json
{
  "selected_arxiv_id": "{arxiv_id}",
  "reasoning": "{your detailed reasoning for selecting this paper}"
}
```
"""

    # テンプレートにデータを適用
    data = {
        "papers_text": papers_text,
        "learnings_text": learnings_text,
    }

    if base_paper:
        data["base_paper"] = base_paper

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    # LLMを使用して最適な論文を選択
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
    reasoning = output_dict["reasoning"]

    print(f"Selected paper: {selected_arxiv_id}")
    print(f"Reasoning: {reasoning}")

    # 選択された論文を返す
    for paper in candidate_papers:
        if paper.arxiv_id == selected_arxiv_id:
            return paper

    # 見つからない場合は最初の論文を返す（通常はここに到達しない）
    print("Warning: Selected paper not found in candidates. Returning first paper.")
    return candidate_papers[0]


if __name__ == "__main__":
    # テスト用のダミーデータ
    from pydantic import BaseModel

    # ダミーの候補論文
    candidate_papers = [
        CandidatePaperInfo(
            arxiv_id="2101.12345",
            arxiv_url="https://arxiv.org/abs/2101.12345",
            title="Deep Learning for Natural Language Processing",
            authors=["John Smith", "Jane Doe"],
            published_date="2021-01-15",
            summary="This paper presents a novel approach to NLP using deep learning.",
            github_url="https://github.com/example/nlp-deep-learning",
            main_contributions="Improved accuracy in language translation tasks.",
            methodology="Transformer-based architecture with attention mechanisms.",
            experimental_setup="Tested on multiple datasets including GLUE and SQuAD.",
            limitations="High computational requirements.",
            future_research_directions="Exploring more efficient training methods.",
        ),
        CandidatePaperInfo(
            arxiv_id="2102.54321",
            arxiv_url="https://arxiv.org/abs/2102.54321",
            title="Efficient Transformers for Language Understanding",
            authors=["Alice Johnson", "Bob Brown"],
            published_date="2021-02-20",
            summary="This paper introduces efficiency improvements for transformer models.",
            github_url="https://github.com/example/efficient-transformers",
            main_contributions="Reduced training time by 40% while maintaining accuracy.",
            methodology="Sparse attention patterns and parameter sharing.",
            experimental_setup="Evaluated on GLUE benchmark and machine translation tasks.",
            limitations="Slight decrease in performance on very long sequences.",
            future_research_directions="Combining with other efficiency techniques.",
        ),
    ]

    # ダミーの学習内容
    learnings = [
        "Transformer models have become the standard for NLP tasks.",
        "Efficiency is a major concern for deploying large language models.",
        "Attention mechanisms are computationally expensive but crucial for performance.",
    ]

    # 最適な論文を選択
    selected_paper = select_best_paper_with_context(
        llm_name="gpt-4o-mini-2024-07-18",
        candidate_papers=candidate_papers,
        learnings=learnings,
    )

    print(f"Selected paper: {selected_paper.title}")
