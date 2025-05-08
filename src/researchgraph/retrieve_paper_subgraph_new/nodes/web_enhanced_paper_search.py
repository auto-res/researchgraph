"""
Web Enhanced Paper Search

This module combines traditional academic search with web search capabilities
to provide more comprehensive and up-to-date research results.
"""

import asyncio
from typing import List, Dict, Any, TypedDict, Optional
from pydantic import BaseModel

from researchgraph.retrieve_paper_subgraph_new.nodes.recursive_paper_search import (
    recursive_paper_search,
    CandidatePaperInfo,
)
from researchgraph.retrieve_paper_subgraph_new.nodes.web_based_paper_search import (
    search_papers_with_web_api,
    search_papers_from_multiple_queries,
    WebSearchResultData,
)


class EnhancedSearchResult(TypedDict):
    """Results from combined traditional and web search"""
    papers: List[CandidatePaperInfo]
    insights: List[str]
    visited_urls: List[str]
    github_repositories: List[str]


async def web_enhanced_paper_search(
    queries: List[str],
    llm_name: str,
    save_dir: str,
    breadth: int = 2,
    depth: int = 1,
    web_search_enabled: bool = True,
) -> EnhancedSearchResult:
    """
    Combined search strategy using both traditional academic search and web search.
    
    This approach gives more comprehensive and up-to-date results than either method alone.
    
    Args:
        queries: List of search queries
        llm_name: Name of the LLM to use
        save_dir: Directory to save papers
        breadth: Breadth of recursive search
        depth: Depth of recursive search
        web_search_enabled: Whether to enable web search
        
    Returns:
        EnhancedSearchResult containing papers, insights, and other data
    """
    print(f"Starting web enhanced paper search with queries: {queries}")
    
    # Start both search methods concurrently if web search is enabled
    if web_search_enabled:
        # Run both traditional and web searches concurrently
        traditional_search_task = asyncio.create_task(
            recursive_paper_search(
                llm_name=llm_name,
                queries=queries,
                breadth=breadth,
                depth=depth,
                save_dir=save_dir,
            )
        )
        
        web_search_task = asyncio.create_task(
            search_papers_from_multiple_queries(
                queries=queries,
                llm_name=llm_name,
                save_dir=save_dir,
                max_papers_per_query=3,
            )
        )
        
        # Wait for both searches to complete
        traditional_result, web_result = await asyncio.gather(
            traditional_search_task, 
            web_search_task
        )
    else:
        # Only run traditional search if web search is disabled
        traditional_result = await recursive_paper_search(
            llm_name=llm_name,
            queries=queries,
            breadth=breadth,
            depth=depth,
            save_dir=save_dir,
        )
        
        web_result = {
            "papers": [],
            "insights": [],
            "github_repositories": [],
            "other_resources": [],
        }
    
    # Combine results from both search methods
    all_papers = []
    all_insights = []
    all_visited_urls = traditional_result["visited_urls"]
    all_github_repositories = []
    
    # Add all papers from traditional search
    all_papers.extend(traditional_result["paper_candidates"])
    
    # Add insights from traditional search
    all_insights.extend(traditional_result["learnings"])
    
    # If web search was enabled, combine those results
    if web_search_enabled:
        # Add papers from web search
        web_papers = web_result["papers"]
        
        # Prevent duplicates by checking arxiv_ids
        seen_arxiv_ids = {paper.arxiv_id for paper in all_papers if paper.arxiv_id}
        
        for paper in web_papers:
            if paper.arxiv_id and paper.arxiv_id not in seen_arxiv_ids:
                all_papers.append(paper)
                seen_arxiv_ids.add(paper.arxiv_id)
        
        # Add insights from web search
        all_insights.extend(web_result["insights"])
        
        # Add GitHub repositories
        all_github_repositories.extend(web_result["github_repositories"])
    
    # Remove duplicate insights
    unique_insights = list(dict.fromkeys(all_insights))
    
    # Return combined results
    return {
        "papers": all_papers,
        "insights": unique_insights,
        "visited_urls": all_visited_urls,
        "github_repositories": all_github_repositories,
    }


# Example usage
if __name__ == "__main__":
    async def main():
        import os
        
        save_dir = os.path.join(os.getcwd(), "data")
        
        result = await web_enhanced_paper_search(
            queries=["transformer architecture advancements"],
            llm_name="gpt-4o",
            save_dir=save_dir,
            breadth=1,
            depth=1,
            web_search_enabled=True,
        )
        
        print("\n=== Papers Found ===")
        for i, paper in enumerate(result["papers"], 1):
            print(f"{i}. {paper.title}")
            print(f"   ArXiv ID: {paper.arxiv_id if hasattr(paper, 'arxiv_id') else 'N/A'}")
            print(f"   GitHub URL: {paper.github_url or 'None'}")
            print()
        
        print(f"\nFound {len(result['papers'])} papers")
        print(f"Gathered {len(result['insights'])} insights")
        print(f"Discovered {len(result['github_repositories'])} GitHub repositories")
    
    asyncio.run(main())
