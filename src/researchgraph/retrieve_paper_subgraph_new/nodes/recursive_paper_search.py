from typing import TypedDict, Optional, List, Dict, Any
import asyncio
from pydantic import BaseModel
import os

from researchgraph.deep_research_subgraph.nodes.generate_queries import (
    generate_queries,
    QueryInfo,
)
from researchgraph.deep_research_subgraph.nodes.request_firecrawl_api import (
    request_firecrawl_api,
)
from researchgraph.deep_research_subgraph.nodes.process_serp_result import (
    process_serp_result,
)
from researchgraph.retrieve_paper_subgraph.nodes.search_api.arxiv_api_node import (
    ArxivNode,
)
from researchgraph.retrieve_paper_subgraph_new.nodes.retrieve_arxiv_text_node_new import (
    RetrievearXivTextNodeNew,
)
from researchgraph.retrieve_paper_subgraph.nodes.extract_github_url_node import (
    ExtractGithubUrlNode,
)
from researchgraph.retrieve_paper_subgraph.nodes.summarize_paper_node import (
    summarize_paper_node,
    summarize_paper_prompt_base,
)


class CandidatePaperInfo(BaseModel):
    arxiv_id: str
    arxiv_url: str
    title: str
    authors: list[str]
    published_date: str
    journal: str = ""
    doi: str = ""
    summary: str
    github_url: str = ""
    main_contributions: str = ""
    methodology: str = ""
    experimental_setup: str = ""
    limitations: str = ""
    future_research_directions: str = ""


class ResearchResult(TypedDict):
    learnings: list[str]
    visited_urls: list[str]
    paper_candidates: list[CandidatePaperInfo]


async def recursive_paper_search(
    llm_name: str,
    queries: list[str],
    breadth: int,
    depth: int,
    save_dir: str,
    previous_learnings: list[str] = None,
) -> ResearchResult:
    """
    再帰的に論文を検索し、関連する知識を収集する

    1. クエリから複数の検索クエリを生成
    2. 各クエリでウェブ検索（FireCrawl API）と論文検索（arXiv API）を実行
    3. 結果を処理して学習内容を抽出
    4. フォローアップ質問を生成して再帰的に探索
    5. 見つかった論文候補と学習内容を返す
    """
    print(f"\n{'='*50}")
    print(f"STARTING RECURSIVE PAPER SEARCH (depth={depth})")
    print(f"Initial queries: {queries}")
    print(f"{'='*50}\n")

    if previous_learnings is None:
        previous_learnings = []

    # 結果を格納する変数を初期化
    all_learnings = previous_learnings.copy()
    all_visited_urls = []
    all_paper_candidates = []

    # 各クエリを処理
    for i, query in enumerate(queries):
        print(f"\n{'-'*50}")
        print(f"Processing query {i+1}/{len(queries)}: '{query}'")
        print(f"{'-'*50}")

        # 1. クエリから複数の検索クエリを生成
        print(f"Generating search queries from: '{query}'")
        serp_queries_list = await generate_queries(
            llm_name=llm_name,
            query=query,
            num_queries=breadth,
            learnings=previous_learnings,
        )
        print(f"Generated {len(serp_queries_list.queries_list)} search queries:")
        for i, q in enumerate(serp_queries_list.queries_list):
            print(f"  {i+1}. {q.query} (Goal: {q.research_goal[:50]}...)")

        # 2. 各検索クエリを処理
        for j, serp_query in enumerate(serp_queries_list.queries_list):
            print(f"\nProcessing search query {j+1}/{len(serp_queries_list.queries_list)}: '{serp_query.query}'")

            # ウェブ検索（FireCrawl API）
            try:
                print(f"  Searching web with FireCrawl API...")
                # FireCrawl APIを使用してウェブ検索を実行
                print(f"  Executing FireCrawl API search...")
                search_result = await request_firecrawl_api(serp_query.query)

                # 検索結果からURLを抽出
                new_urls = [item.url for item in search_result if item.url]
                print(f"  Found {len(new_urls)} web results")

                # 検索結果のタイトルを表示
                if search_result:
                    print(f"  Search result titles:")
                    for i, item in enumerate(search_result[:3], 1):  # 最初の3件のみ表示
                        title = item.title if item.title else "No title"
                        print(f"    {i}. {title[:50]}...")
                    if len(search_result) > 3:
                        print(f"    ... and {len(search_result) - 3} more results")

                all_visited_urls.extend(new_urls)

                # 結果を処理して学習内容を抽出
                print(f"  Processing search results to extract learnings...")
                processed_result = await process_serp_result(
                    llm_name=llm_name,
                    query=serp_query.query,
                    result=search_result,
                )
                print(f"  Extracted {len(processed_result.learnings)} learnings")
                all_learnings.extend(processed_result.learnings)

                # 論文検索（arXiv API）- 一流学会誌を優先
                print(f"  Searching papers on arXiv...")
                # 一流学会誌を含むクエリを作成
                academic_query = f"{serp_query.query} (NeurIPS OR ICML OR AAAI OR ICLR OR CVPR OR ACL)"
                print(f"  Enhanced academic query: '{academic_query}'")

                paper_search = ArxivNode(
                    num_retrieve_paper=5,
                    period_days=365,  # 検索期間を1年に拡大
                )
                paper_results = paper_search.execute([academic_query])
                print(f"  Found {len(paper_results)} papers on arXiv")

                # 各論文を処理
                paper_count = 0
                github_count = 0
                for k, paper_info in enumerate(paper_results):
                    try:
                        paper_id = paper_info.get("arxiv_id", f"unknown-{k}")
                        paper_title = paper_info.get("title", "Untitled")
                        print(f"  Processing paper {k+1}/{len(paper_results)}: {paper_id} - {paper_title[:50]}...")

                        # 論文の全文を取得
                        arxiv_url = paper_info["arxiv_url"]
                        print(f"    Retrieving full text from {arxiv_url}...")
                        paper_full_text = RetrievearXivTextNodeNew(
                            save_dir=save_dir,
                        ).execute(arxiv_url=arxiv_url)
                        print(f"    Retrieved {len(paper_full_text)} characters of text")

                        # GitHub URLを抽出
                        print(f"    Extracting GitHub URL...")
                        github_url = ExtractGithubUrlNode(
                            llm_name=llm_name,
                        ).execute(
                            paper_full_text=paper_full_text,
                            paper_summary=paper_info["summary"],
                        )

                        # GitHub URLの有無に関わらず論文を要約
                        if github_url:
                            print(f"    Found GitHub URL: {github_url}")
                            github_count += 1
                        else:
                            print(f"    No GitHub URL found, using empty URL")
                            github_url = ""  # 空のURLを使用

                        # 論文を要約
                        print(f"    Summarizing paper...")
                        (
                            main_contributions,
                            methodology,
                            experimental_setup,
                            limitations,
                            future_research_directions,
                        ) = summarize_paper_node(
                            llm_name=llm_name,
                            prompt_template=summarize_paper_prompt_base,
                            paper_text=paper_full_text,
                        )
                        print(f"    Paper summarized successfully")

                        # 候補論文情報を作成
                        candidate_paper = CandidatePaperInfo(
                            arxiv_id=paper_info["arxiv_id"],
                            arxiv_url=paper_info["arxiv_url"],
                            title=paper_info.get("title", ""),
                            authors=paper_info.get("authors", []),
                            published_date=paper_info.get("published_date", ""),
                            journal=paper_info.get("journal", ""),
                            doi=paper_info.get("doi", ""),
                            summary=paper_info.get("summary", ""),
                            github_url=github_url,
                            main_contributions=main_contributions,
                            methodology=methodology,
                            experimental_setup=experimental_setup,
                            limitations=limitations,
                            future_research_directions=future_research_directions,
                        )
                        all_paper_candidates.append(candidate_paper)
                        paper_count += 1
                        print(f"    Added paper to candidates list")
                    except Exception as e:
                        print(f"    Error processing paper {paper_info.get('arxiv_id', 'unknown')}: {str(e)}")

                print(f"  Processed {len(paper_results)} papers, found {github_count} with GitHub URLs, added {paper_count} to candidates")

                # 再帰的に探索（深さが残っている場合）
                if depth > 1:
                    # フォローアップ質問を使用して再帰的に探索
                    new_breadth = max(1, breadth // 2)
                    new_depth = depth - 1

                    print(f"\n  Starting recursive search with follow-up questions:")
                    for q_idx, q in enumerate(processed_result.followup_questions):
                        print(f"    {q_idx+1}. {q}")

                    # 再帰呼び出し
                    recursive_result = await recursive_paper_search(
                        llm_name=llm_name,
                        queries=processed_result.followup_questions,
                        breadth=new_breadth,
                        depth=new_depth,
                        save_dir=save_dir,
                        previous_learnings=all_learnings,
                    )

                    # 結果を統合
                    print(f"  Recursive search complete, integrating results...")
                    all_learnings.extend(recursive_result["learnings"])
                    all_visited_urls.extend(recursive_result["visited_urls"])
                    all_paper_candidates.extend(recursive_result["paper_candidates"])
                    print(f"  Added {len(recursive_result['learnings'])} learnings, {len(recursive_result['visited_urls'])} URLs, and {len(recursive_result['paper_candidates'])} paper candidates from recursive search")

            except Exception as e:
                print(f"  Error processing query '{serp_query.query}': {str(e)}")

    print(f"\n{'-'*50}")
    print(f"FINALIZING RESULTS")
    print(f"{'-'*50}")

    # 重複を削除
    print(f"Removing duplicates from {len(all_learnings)} learnings and {len(all_visited_urls)} URLs...")
    unique_learnings = list(dict.fromkeys(all_learnings))
    unique_visited_urls = list(dict.fromkeys(all_visited_urls))
    print(f"After deduplication: {len(unique_learnings)} unique learnings and {len(unique_visited_urls)} unique URLs")

    # 論文候補の重複を削除（arxiv_idで判断）
    print(f"Removing duplicate papers from {len(all_paper_candidates)} candidates...")
    unique_paper_candidates = []
    seen_arxiv_ids = set()
    for paper in all_paper_candidates:
        if paper.arxiv_id not in seen_arxiv_ids:
            unique_paper_candidates.append(paper)
            seen_arxiv_ids.add(paper.arxiv_id)
    print(f"After deduplication: {len(unique_paper_candidates)} unique paper candidates")

    # 候補論文の情報を表示
    if unique_paper_candidates:
        print("\nCandidate papers:")
        for i, paper in enumerate(unique_paper_candidates):
            print(f"  {i+1}. {paper.title} (ID: {paper.arxiv_id})")
            print(f"     GitHub: {paper.github_url}")

    return {
        "learnings": unique_learnings,
        "visited_urls": unique_visited_urls,
        "paper_candidates": unique_paper_candidates,
    }


if __name__ == "__main__":
    async def main():
        save_dir = "/workspaces/researchgraph/data"
        llm_name = "gpt-4o-mini-2024-07-18"

        result = await recursive_paper_search(
            llm_name=llm_name,
            queries=["deep learning"],
            breadth=2,
            depth=1,
            save_dir=save_dir,
        )

        print(f"Learnings: {len(result['learnings'])}")
        print(f"Visited URLs: {len(result['visited_urls'])}")
        print(f"Paper candidates: {len(result['paper_candidates'])}")

    asyncio.run(main())
