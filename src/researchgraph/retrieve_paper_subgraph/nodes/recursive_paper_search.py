import asyncio
from typing import TypedDict, Optional
from pydantic import BaseModel, Field

from researchgraph.deep_research_subgraph.nodes.generate_queries import generate_queries, QueryInfo
from researchgraph.deep_research_subgraph.nodes.request_firecrawl_api import request_firecrawl_api
from researchgraph.deep_research_subgraph.nodes.process_serp_result import process_serp_result
from researchgraph.retrieve_paper_subgraph.nodes.search_api.arxiv_api_node import ArxivNode
from researchgraph.retrieve_paper_subgraph.nodes.retrieve_arxiv_text_node import RetrieveArxivTextNode
from researchgraph.retrieve_paper_subgraph.nodes.extract_github_url_node import ExtractGithubUrlNode
from researchgraph.retrieve_paper_subgraph.nodes.summarize_paper_node import (
    summarize_paper_node,
    summarize_paper_prompt_base,
)

class CandidatePaperInfo(BaseModel):
    arxiv_id: str = Field(..., description="arXivの識別ID")
    arxiv_url: str = Field(..., description="arXivのURL")
    title: str = Field(..., description="論文のタイトル")
    authors: list[str] = Field(..., description="論文の著者リスト")
    published_date: str = Field(..., description="公開日")
    journal: str = Field("", description="掲載ジャーナル名")
    doi: str = Field("", description="DOI番号")
    summary: str = Field(..., description="論文の要約")
    github_url: str = Field("", description="関連するGitHub URL")
    main_contributions: str = Field("", description="主な貢献内容")
    methodology: str = Field("", description="研究手法")
    experimental_setup: str = Field("", description="実験設定")
    limitations: str = Field("", description="研究の限界点")
    future_research_directions: str = Field("", description="将来的な研究方向性")


class ResearchResult(TypedDict):
    learnings: list[str]
    visited_urls: list[str]
    paper_candidates: list[CandidatePaperInfo]


class RecursivePaperSearchNode:
    def __init__(self, llm_name: str, breadth: int, depth: int, save_dir: str):
        self.llm_name = llm_name
        self.breadth = breadth
        self.depth = depth
        self.save_dir = save_dir

    async def _recursive_search(self, queries: list, previous_learnings: Optional[list] = None) -> tuple[list, list, list]:
        if previous_learnings is None:
            previous_learnings = []

        all_learnings = previous_learnings.copy()
        all_visited_urls = []
        all_paper_candidates = []

        for query in queries:
            serp_queries = await self._generate_serp_queries(query, previous_learnings)

            for serp_query_info in serp_queries:
                urls, processed_result = await self._perform_web_search(serp_query_info.query)
                all_visited_urls.extend(urls)
                all_learnings.extend(processed_result.learnings)

                papers = self._perform_arxiv_search(serp_query_info.query)
                for paper_info in papers:
                    try:
                        candidate_paper = self._process_paper(paper_info)
                        all_paper_candidates.append(candidate_paper)
                    except Exception as e:
                        print(f"Error processing paper {paper_info.get('arxiv_id', 'unknown')}: {e}")

                if self.depth > 1:
                    self.depth -= 1
                    recursive_learnings, recursive_visited_urls, recursive_paper_candidates = await self._recursive_search(
                        processed_result.followup_questions, all_learnings
                    )
                    all_learnings.extend(recursive_learnings)
                    all_visited_urls.extend(recursive_visited_urls)
                    all_paper_candidates.extend(recursive_paper_candidates)
        
        return all_learnings, all_visited_urls, all_paper_candidates
                    

    async def _generate_serp_queries(self, query: str, previous_learnings: list) -> list[QueryInfo]:
        serp_queries_list = await generate_queries(
            llm_name=self.llm_name, 
            query=query, 
            num_queries=self.breadth, 
            learnings=previous_learnings, 
        )
        return serp_queries_list.queries_list
    
    async def _perform_web_search(self, serp_query: str) -> tuple:
        search_result = await request_firecrawl_api(serp_query)
        urls = [item.url for item in search_result if item.url]
        processed_result = await process_serp_result(
            llm_name=self.llm_name, query=serp_query, result=search_result
        )
        return urls, processed_result
    
    def _perform_arxiv_search(self, serp_query: str) -> list:
        academic_query = f"{serp_query} (NeurIPS OR ICML OR AAAI OR ICLR OR CVPR OR ACL)" #TODO: クエリを変更する
        paper_search = ArxivNode(num_retrieve_paper=5, period_days=365) #TODO: インスタンス引数にする
        return paper_search.execute([academic_query])
    
    def _process_paper(self, paper_info: dict) -> CandidatePaperInfo:
        arxiv_url = paper_info["arxiv_url"]
        paper_full_text =  RetrieveArxivTextNode(save_dir=self.save_dir).execute(arxiv_url=arxiv_url)
        github_url = ExtractGithubUrlNode(llm_name=self.llm_name).execute(
            paper_full_text=paper_full_text, paper_summary=paper_info["summary"]
        )
        (
            main_contributions, 
            methodology, 
            experimental_setup, 
            limitations, 
            future_research_directions, 
        ) = summarize_paper_node(
            llm_name=self.llm_name, 
            prompt_template=summarize_paper_prompt_base, 
            paper_text=paper_full_text        
        )

        candidate_paper = CandidatePaperInfo(
            arxiv_id=paper_info["arxiv_id"],
            arxiv_url=arxiv_url,
            title=paper_info.get("title", ""),
            authors=paper_info.get("authors", []),
            published_date=paper_info.get("published_date", ""),
            journal=paper_info.get("journal", ""),
            doi=paper_info.get("doi", ""),
            summary=paper_info.get("summary", ""),
            github_url=github_url or "",
            main_contributions=main_contributions,
            methodology=methodology,
            experimental_setup=experimental_setup,
            limitations=limitations,
            future_research_directions=future_research_directions,
        )
        return candidate_paper
    
    def finalize_results(self, learnings: list, visited_urls: list, paper_candidates: list) -> ResearchResult:
        unique_learnings = list(dict.fromkeys(learnings))
        unique_visited_urls = list(dict.fromkeys(visited_urls))

        seen_ids = set()
        unique_papers = []
        for paper in paper_candidates:
            if paper.arxiv_id not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper.arxiv_id)

        return ResearchResult(
            learnings=unique_learnings, 
            visited_urls=unique_visited_urls, 
            paper_candidates=unique_papers
        )

    async def execute(self, initial_queries: list, previous_learnings: Optional[list] = []) -> ResearchResult:
        all_learnings, all_visited_urls, all_paper_candidates = await self._recursive_search(initial_queries, previous_learnings)
        return self.finalize_results(all_learnings, all_visited_urls, all_paper_candidates)


if __name__ == "__main__":
    async def main():
        save_dir = "/workspaces/researchgraph/data"
        llm_name = "gpt-4o-mini-2024-07-18"

        recursive_paper_search = RecursivePaperSearchNode(
            llm_name=llm_name,
            breadth=1,
            depth=1,
            save_dir=save_dir,
        )
        result = await recursive_paper_search.execute(initial_queries=["deep learning"])
        print(f"Learnings: {len(result['learnings'])}")
        print(f"Visited URLs: {len(result['visited_urls'])}")
        print(f"Paper candidates: {len(result['paper_candidates'])}")
        
    asyncio.run(main())
