import asyncio
import json
from typing import TypedDict, Optional
from pydantic import BaseModel, Field
from litellm import completion

from researchgraph.deep_research_subgraph.nodes.generate_queries import generate_queries, QueryInfo
from researchgraph.deep_research_subgraph.nodes.perform_web_scrape import perform_web_scrape 
from researchgraph.deep_research_subgraph.nodes.process_serp_result import process_serp_result
from researchgraph.retrieve_paper_subgraph.nodes.search_api.arxiv_api_node import ArxivNode
from researchgraph.retrieve_paper_subgraph.nodes.retrieve_arxiv_text_node import RetrieveArxivTextNode
from researchgraph.retrieve_paper_subgraph.nodes.extract_github_url_node import ExtractGithubUrlNode
from researchgraph.deep_research_subgraph.nodes.request_firecrawl_api import SearchResponseItem
from researchgraph.retrieve_paper_subgraph.nodes.summarize_paper_node import (
    summarize_paper_node,
    summarize_paper_prompt_base,
)


class LLMOutput(BaseModel):
    paper_titles: str


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
    paper_candidates: list[CandidatePaperInfo]


class RecursivePaperSearchNode:
    def __init__(
        self, 
        llm_name: str, 
        breadth: int, 
        depth: int, 
        save_dir: str, 
        scrape_urls: list, 
        arxiv_query_batch_size: int = 10, 
        arxiv_num_retrieve_paper: int = 1, 
        arxiv_period_days: Optional[int] = None, 
    ):
        self.llm_name = llm_name
        self.breadth = breadth
        self.depth = depth
        self.save_dir = save_dir
        self.scrape_urls = scrape_urls
        self.arxiv_query_batch_size = arxiv_query_batch_size
        self.arxiv_num_retrieve_paper = arxiv_num_retrieve_paper
        self.arxiv_period_days = arxiv_period_days        

    async def _recursive_search(self, queries: list, previous_learnings: Optional[list] = None) -> tuple[list, list]:
        if previous_learnings is None:
            previous_learnings = []

        all_learnings = previous_learnings.copy()
        all_paper_candidates = []

        for query in queries:
            serp_queries = await self._generate_serp_queries(query, previous_learnings) 

            for serp_query_info in serp_queries:
                paper_titles, processed_result = await self._perform_web_scrape(
                    serp_query_info.query,
                    query, 
                    previous_learnings
                )
                all_learnings.extend(processed_result.learnings)

                papers = self._perform_arxiv_search(paper_titles[:min(len(paper_titles), self.arxiv_query_batch_size)])
                for paper_info in papers:
                    try:
                        candidate_paper = self._process_paper(paper_info)
                        all_paper_candidates.append(candidate_paper)
                    except Exception as e:
                        print(f"Error processing paper {paper_info.get('arxiv_id', 'unknown')}: {e}")

                if self.depth > 1:
                    self.depth -= 1
                    recursive_learnings, recursive_paper_candidates = await self._recursive_search(
                        processed_result.followup_questions, all_learnings
                    )
                    all_learnings.extend(recursive_learnings)
                    all_paper_candidates.extend(recursive_paper_candidates)
        
        return all_learnings, all_paper_candidates
                    

    async def _generate_serp_queries(self, query: str, previous_learnings: list) -> list[QueryInfo]:
        serp_queries_list = await generate_queries(
            llm_name=self.llm_name, 
            query=query, 
            num_queries=self.breadth, 
            learnings=previous_learnings, 
        )
        return serp_queries_list.queries_list

    async def _perform_web_scrape(self, serp_query: str, query: str, previous_learnings: list[str]) -> tuple[Optional[list[str]], ResearchResult]:
        max_query_retries = 2
        for attempt in range(max_query_retries + 1):
            result = await perform_web_scrape(serp_query, self.scrape_urls)
            paper_titles =  self._extract_paper_title(
                result=result, 
                query=serp_query, 
                previous_learnings=previous_learnings, 
            )
            if paper_titles:
                break
            else:
                print(f"Paper titles are empty for query: {serp_query}. Regenerating query...")
                new_serp_queries = await self._generate_serp_queries(query, previous_learnings)
                if new_serp_queries and len(new_serp_queries) > 0:
                    serp_query = new_serp_queries[0].query
                else:
                    print("Failed to re-generate query.")
                    break

        processed_result = await process_serp_result(
            llm_name=self.llm_name, query=serp_query, result=result
        )
        return paper_titles, processed_result
    
    def _extract_paper_title(
        self,
        result: list[SearchResponseItem], 
        query: str, 
        previous_learnings: list[str], 
        max_retries: int = 3
    ) -> Optional[list[str]]:
        
        learnings_str = "; ".join(previous_learnings) if previous_learnings else "No previous learnings"
        prompt = f"""
        You are an expert at extracting research paper titles from web content. 
        We have the following information:
            - Query: {query}
            - Previous learnings or context: {learnings_str}

        Below is a block of markdown content from a research paper listing page. 

        Your tasks are:
        1. Identify only the titles of research papers within the markdown. 
        These may appear as the text inside markdown links (for example, text enclosed in ** or within [ ] if the link text represents a title).
        2. Order the extracted titles in descending order of relevance to the Query (i.e., most relevant first).
        3. Output the extracted titles as a single plain text string, with each title separated by a newline character.
        4. Return your answer in JSON format with a single key "paper_titles". Do not include any additional commentary or formatting.

        Content:
        <result>
        {result}
        </result>
        """

        for attempt in range(max_retries):
            try:
                response = completion(
                    model=self.llm_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format=LLMOutput,
                )
                structured_output = json.loads(response.choices[0].message.content)
                titles_str = structured_output["paper_titles"]
                titles_list = [title.strip() for title in titles_str.split('\n') if title.strip()]
                return titles_list
            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_retries}] Error calling LLM: {e}")
        print("Exceeded maximum retries for LLM call.")
        return None
    
    def _perform_arxiv_search(self, paper_titles: Optional[list]) -> list:
        if not paper_titles:
            print("No paper titles provided to _perform_arxiv_search. Skipping Arxiv search.")
            return []
        
        print(f"Query to arXiv: {paper_titles}")
        papers = ArxivNode(
            num_retrieve_paper=self.arxiv_num_retrieve_paper, 
            period_days=self.arxiv_period_days
        ).execute(paper_titles)
        print(f"Papers from arXiv: {papers}")
        return papers
    
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
    
    def finalize_results(self, learnings: list, paper_candidates: list) -> ResearchResult:
        unique_learnings = list(dict.fromkeys(learnings))

        seen_ids = set()
        unique_papers = []
        for paper in paper_candidates:
            if paper.arxiv_id not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper.arxiv_id)

        return ResearchResult(
            learnings=unique_learnings, 
            paper_candidates=unique_papers
        )

    async def execute(self, initial_queries: list, previous_learnings: Optional[list] = []) -> ResearchResult:
        all_learnings, all_paper_candidates = await self._recursive_search(initial_queries, previous_learnings)
        return self.finalize_results(all_learnings, all_paper_candidates)


if __name__ == "__main__":
    async def main():
        save_dir = "/workspaces/researchgraph/data"
        llm_name = "gpt-4o-mini-2024-07-18"
        scrape_urls = [
            "https://icml.cc/virtual/2024/papers.html?filter=titles", 
            "https://iclr.cc/virtual/2024/papers.html?filter=titles", 
            # "https://nips.cc/virtual/2024/papers.html?filter=titles", 
            # "https://cvpr.thecvf.com/virtual/2024/papers.html?filter=titles", 
        ]

        recursive_paper_search = RecursivePaperSearchNode(
            llm_name=llm_name,
            breadth=1,
            depth=1,
            save_dir=save_dir,
            scrape_urls=scrape_urls, 
        )
        result = await recursive_paper_search.execute(initial_queries=["In-context learning"])
        print(f"Learnings: {len(result['learnings'])}")
        print(f"Paper candidates: {len(result['paper_candidates'])}")
        
    asyncio.run(main())
