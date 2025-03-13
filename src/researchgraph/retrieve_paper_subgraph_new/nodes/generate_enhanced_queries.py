from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
import ast
from typing import List

from researchgraph.retrieve_paper_subgraph_new.nodes.recursive_paper_search import (
    CandidatePaperInfo,
)


class LLMOutput(BaseModel):
    generated_query_1: str
    generated_query_2: str
    generated_query_3: str
    generated_query_4: str
    generated_query_5: str


def generate_enhanced_queries(
    llm_name: str,
    base_paper: CandidatePaperInfo,
    learnings: List[str],
) -> List[str]:
    """
    ベース論文と学習内容を考慮して、より洗練された検索クエリを生成する

    Args:
        llm_name: 使用するLLMの名前
        base_paper: 選択されたベース論文
        learnings: 探索で得られた学習内容のリスト

    Returns:
        生成された検索クエリのリスト
    """
    print(f"\n{'='*50}")
    print(f"GENERATING ENHANCED QUERIES")
    print(f"{'='*50}")

    print(f"Base paper: {base_paper.title} (ID: {base_paper.arxiv_id})")
    print(f"Number of learnings from research: {len(learnings)}")

    # 学習内容をテキストに変換
    print("Converting learnings to text format...")
    learnings_text = "\n".join([f"- {learning}" for learning in learnings])

    # プロンプトテンプレート
    prompt_template = """
You are an expert research assistant tasked with generating search queries for finding relevant research papers.
Your goal is to create a set of well-structured queries that can be used with a research paper search API
to retrieve papers that are conceptually or methodologically related to a given foundational paper (Research A).

**Research A (Base Paper):**
- **Title:** {{ base_paper.title }}
- **Authors:** {{ base_paper.authors | join(', ') }}
- **Publication Date:** {{ base_paper.published_date }}
- **Journal/Conference:** {{ base_paper.journal }}
- **DOI:** {{ base_paper.doi }}
- **Main Contributions:** {{ base_paper.main_contributions }}
- **Methodology:** {{ base_paper.methodology }}
- **Experimental Setup:** {{ base_paper.experimental_setup }}
- **Limitations:** {{ base_paper.limitations }}
- **Future Research Directions:** {{ base_paper.future_research_directions }}

**Additional Context from Research:**
{{ learnings_text }}

---

**Instructions (Important!):**
1. Analyze the provided Research A details and the additional context from research.
2. Generate exactly 5 **short** search queries (ideally 1-5 words each) that are:
   - Focused on finding papers that could complement or extend Research A
   - Informed by the additional context from research
   - Diverse in their focus to cover different aspects of the research area
   - Specific enough to yield relevant results
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

    # テンプレートにデータを適用
    data = {
        "base_paper": base_paper,
        "learnings_text": learnings_text,
    }

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    # LLMを使用してクエリを生成
    response = completion(
        model=llm_name,
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        response_format=LLMOutput,
    )

    output = response.choices[0].message.content
    output_dict = ast.literal_eval(output)

    # 生成されたクエリをリストに変換
    generated_queries = [
        output_dict["generated_query_1"],
        output_dict["generated_query_2"],
        output_dict["generated_query_3"],
        output_dict["generated_query_4"],
        output_dict["generated_query_5"],
    ]

    # 生成されたクエリを表示
    print("\nGenerated enhanced queries:")
    for i, query in enumerate(generated_queries, 1):
        print(f"  {i}. {query}")
    print()

    return generated_queries


if __name__ == "__main__":
    # テスト用のダミーデータ
    base_paper = CandidatePaperInfo(
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
    )

    learnings = [
        "Transformer models have become the standard for NLP tasks.",
        "Efficiency is a major concern for deploying large language models.",
        "Attention mechanisms are computationally expensive but crucial for performance.",
    ]

    queries = generate_enhanced_queries(
        llm_name="gpt-4o-mini-2024-07-18",
        base_paper=base_paper,
        learnings=learnings,
    )

    print("Generated queries:")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
