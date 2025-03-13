from typing import List, Dict, Any
from datetime import datetime
from researchgraph.retrieve_paper_subgraph_new.nodes.recursive_paper_search import (
    CandidatePaperInfo,
)


def generate_markdown_report(
    base_paper: CandidatePaperInfo,
    add_paper: CandidatePaperInfo,
    base_learnings: List[str],
    add_learnings: List[str],
    base_visited_urls: List[str],
    add_visited_urls: List[str],
    max_learnings: int = 10,  # 学習内容の最大数を制限
) -> dict:
    """
    ベース論文と追加論文の情報、および学習内容からMarkdownレポートを生成する

    Args:
        base_paper: ベース論文情報
        add_paper: 追加論文情報
        base_learnings: ベース論文検索から得られた学習内容
        add_learnings: 追加論文検索から得られた学習内容
        base_visited_urls: ベース論文検索で訪問したURL
        add_visited_urls: 追加論文検索で訪問したURL

    Returns:
        Markdownフォーマットのレポート
    """
    # 現在の日時
    now = datetime.utcnow().isoformat() + "Z"

    # 全てのURLを結合して重複を削除
    all_urls = list(dict.fromkeys(base_visited_urls + add_visited_urls))

    # 全ての学習内容を結合して重複を削除
    all_learnings = list(dict.fromkeys(base_learnings + add_learnings))

    # URLをソースとして番号付け
    sources = {}
    for i, url in enumerate(all_urls, 1):
        sources[url] = i

    # 論文のURLもソースに追加
    if base_paper.arxiv_url and base_paper.arxiv_url not in sources:
        sources[base_paper.arxiv_url] = len(sources) + 1
    if add_paper.arxiv_url and add_paper.arxiv_url not in sources:
        sources[add_paper.arxiv_url] = len(sources) + 1
    if base_paper.github_url and base_paper.github_url not in sources:
        sources[base_paper.github_url] = len(sources) + 1
    if add_paper.github_url and add_paper.github_url not in sources:
        sources[add_paper.github_url] = len(sources) + 1

    # レポートのタイトル
    title = f"# Research Report: {base_paper.title} and {add_paper.title}"

    # 概要セクション
    summary = f"""
## Summary

This report presents a comprehensive analysis of two research papers:

1. **{base_paper.title}** by {', '.join(base_paper.authors)} [[{sources.get(base_paper.arxiv_url, '?')}]](#ref{sources.get(base_paper.arxiv_url, '?')})
2. **{add_paper.title}** by {', '.join(add_paper.authors)} [[{sources.get(add_paper.arxiv_url, '?')}]](#ref{sources.get(add_paper.arxiv_url, '?')})

The research explores the relationship between these papers and identifies potential areas for integration and further research.
"""

    # ベース論文セクション
    base_paper_section = f"""
## Base Paper: {base_paper.title}

**Authors:** {', '.join(base_paper.authors)}
**Published:** {base_paper.published_date}
**arXiv URL:** [{base_paper.arxiv_url}]({base_paper.arxiv_url}) [[{sources.get(base_paper.arxiv_url, '?')}]](#ref{sources.get(base_paper.arxiv_url, '?')})
**GitHub Repository:** [{base_paper.github_url}]({base_paper.github_url}) [[{sources.get(base_paper.github_url, '?')}]](#ref{sources.get(base_paper.github_url, '?')})

### Main Contributions

{base_paper.main_contributions}

### Methodology

{base_paper.methodology}

### Experimental Setup

{base_paper.experimental_setup}

### Limitations

{base_paper.limitations}

### Future Research Directions

{base_paper.future_research_directions}
"""

    # 追加論文セクション
    add_paper_section = f"""
## Additional Paper: {add_paper.title}

**Authors:** {', '.join(add_paper.authors)}
**Published:** {add_paper.published_date}
**arXiv URL:** [{add_paper.arxiv_url}]({add_paper.arxiv_url}) [[{sources.get(add_paper.arxiv_url, '?')}]](#ref{sources.get(add_paper.arxiv_url, '?')})
**GitHub Repository:** [{add_paper.github_url}]({add_paper.github_url}) [[{sources.get(add_paper.github_url, '?')}]](#ref{sources.get(add_paper.github_url, '?')})

### Main Contributions

{add_paper.main_contributions}

### Methodology

{add_paper.methodology}

### Experimental Setup

{add_paper.experimental_setup}

### Limitations

{add_paper.limitations}

### Future Research Directions

{add_paper.future_research_directions}
"""

    # 学習内容セクション
    learnings_section = """
## Key Learnings from Research

The following insights were gathered during the research process:

"""

    # 学習内容を制限して箇条書きで追加
    limited_learnings = all_learnings[:max_learnings] if len(all_learnings) > max_learnings else all_learnings
    for i, learning in enumerate(limited_learnings, 1):
        learnings_section += f"{i}. {learning}\n"

    # ソースセクション
    sources_section = """
## Sources

"""

    # ソースを番号付きリストで追加
    sorted_sources = sorted(sources.items(), key=lambda x: x[1])
    for url, idx in sorted_sources:
        sources_section += f"<a id=\"ref{idx}\"></a>[{idx}] [{url}]({url})\n\n"

    # 生成日時
    footer = f"\n\n*Report generated on {now}*"

    # 全てのセクションを結合
    markdown_report = f"{title}\n{summary}\n{base_paper_section}\n{add_paper_section}\n{learnings_section}\n{sources_section}\n{footer}"

    # 構造化されたデータも返す
    report_data = {
        "markdown": markdown_report,
        "base_paper": {
            "title": base_paper.title,
            "authors": base_paper.authors,
            "arxiv_url": base_paper.arxiv_url,
            "github_url": base_paper.github_url,
            "main_contributions": base_paper.main_contributions,
            "methodology": base_paper.methodology,
            "experimental_setup": base_paper.experimental_setup,
            "limitations": base_paper.limitations,
            "future_research_directions": base_paper.future_research_directions
        },
        "add_paper": {
            "title": add_paper.title,
            "authors": add_paper.authors,
            "arxiv_url": add_paper.arxiv_url,
            "github_url": add_paper.github_url,
            "main_contributions": add_paper.main_contributions,
            "methodology": add_paper.methodology,
            "experimental_setup": add_paper.experimental_setup,
            "limitations": add_paper.limitations,
            "future_research_directions": add_paper.future_research_directions
        },
        "learnings": limited_learnings,
        "sources": {url: idx for url, idx in sorted_sources},
        "generated_at": now
    }

    return report_data


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

    add_paper = CandidatePaperInfo(
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
    )

    base_learnings = [
        "Transformer models have become the standard for NLP tasks.",
        "Attention mechanisms are computationally expensive but crucial for performance.",
    ]

    add_learnings = [
        "Efficiency is a major concern for deploying large language models.",
        "Sparse attention patterns can significantly reduce computational requirements.",
    ]

    base_visited_urls = [
        "https://example.com/transformers",
        "https://example.org/nlp-research",
    ]

    add_visited_urls = [
        "https://example.com/efficient-ml",
        "https://example.net/sparse-attention",
    ]

    report = generate_markdown_report(
        base_paper=base_paper,
        add_paper=add_paper,
        base_learnings=base_learnings,
        add_learnings=add_learnings,
        base_visited_urls=base_visited_urls,
        add_visited_urls=add_visited_urls,
    )

    print(report)
