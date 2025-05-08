"""
Web-based Paper Search

This module provides functionality to search for academic papers using
OpenAI's web search capabilities and process the results into a format
compatible with the rest of the research graph system.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel

from researchgraph.retrieve_paper_subgraph_new.nodes.openai_web_search_node import OpenAIWebSearchNode
from researchgraph.retrieve_paper_subgraph_new.nodes.extract_papers_from_web_search import (
    extract_papers_from_web_search,
    ExtractedPaper,
    PaperExtractionResult
)
from researchgraph.retrieve_paper_subgraph_new.nodes.recursive_paper_search import (
    CandidatePaperInfo
)
from researchgraph.retrieve_paper_subgraph.nodes.retrieve_arxiv_text_node import (
    RetrievearXivTextNode,
)
from researchgraph.retrieve_paper_subgraph.nodes.extract_github_url_node import (
    ExtractGithubUrlNode,
)
from researchgraph.retrieve_paper_subgraph.nodes.summarize_paper_node import (
    summarize_paper_node,
    summarize_paper_prompt_base,
)


class WebSearchResultData(TypedDict):
    """Results from web-based paper search"""
    papers: List[CandidatePaperInfo]
    insights: List[str]
    github_repositories: List[str]
    other_resources: List[str]


async def search_papers_with_web_api(
    query: str,
    llm_name: str = "gpt-4o",
    save_dir: str = "./data",
    max_papers: int = 5,
) -> WebSearchResultData:
    """
    Search for academic papers using OpenAI's web search API.
    
    This function combines web search with paper extraction, full text retrieval,
    and summarization to produce a comprehensive result set.
    
    Args:
        query: The search query
        llm_name: Name of the LLM to use for extraction and summarization
        save_dir: Directory to save downloaded papers
        max_papers: Maximum number of papers to process
        
    Returns:
        WebSearchResultData containing processed papers and insights
    """
    print(f"Searching for papers on: '{query}'")
    
    # Initialize the search node and execute search
    search_node = OpenAIWebSearchNode(model=llm_name, max_results=15)
    search_response = search_node.execute(query)
    
    # Extract papers from search results
    print(f"Processing search results...")
    extraction_results = extract_papers_from_web_search(search_response["results"])
    
    # Extract insights from search
    insights = search_response["insights"]
    
    # Initialize result containers
    processed_papers = []
    github_repositories = extraction_results.github_repositories
    other_resources = extraction_results.other_resources
    
    # Process each paper (up to max_papers)
    papers_to_process = extraction_results.papers[:max_papers]
    print(f"Found {len(extraction_results.papers)} potential papers, processing up to {len(papers_to_process)}")
    
    for paper in papers_to_process:
        try:
            print(f"Processing paper: {paper.title}")
            
            # For ArXiv papers, we can retrieve the full text
            if paper.is_arxiv and paper.arxiv_id:
                arxiv_url = f"https://arxiv.org/abs/{paper.arxiv_id}"
                
                # Retrieve full text
                print(f"Retrieving full text from ArXiv: {arxiv_url}")
                retrieve_node = RetrievearXivTextNode(papers_dir=save_dir)
                paper_full_text = retrieve_node.execute(arxiv_url=arxiv_url)
                
                # Extract GitHub URL
                print(f"Extracting GitHub URL...")
                github_node = ExtractGithubUrlNode(llm_name=llm_name)
                github_url = github_node.execute(
                    paper_full_text=paper_full_text,
                    paper_summary=paper.snippet
                )
                
                if github_url and github_url not in github_repositories:
                    github_repositories.append(github_url)
                
                # Summarize paper
                print(f"Summarizing paper...")
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
                
                # Create paper info object
                candidate_paper = CandidatePaperInfo(
                    arxiv_id=paper.arxiv_id,
                    arxiv_url=arxiv_url,
                    title=paper.title,
                    authors=["Unknown"],  # Would need additional API call to get authors
                    published_date="",
                    summary=paper.snippet,
                    github_url=github_url,
                    main_contributions=main_contributions,
                    methodology=methodology,
                    experimental_setup=experimental_setup,
                    limitations=limitations,
                    future_research_directions=future_research_directions,
                )
                
                processed_papers.append(candidate_paper)
                print(f"Successfully processed ArXiv paper: {paper.title}")
                
            else:
                # For non-ArXiv papers, we can't easily get the full text,
                # but we can use the web search data to create a partial record
                
                # For now, we'll skip non-ArXiv papers, but we could add support
                # for other sources in the future
                print(f"Skipping non-ArXiv paper: {paper.title}")
                
        except Exception as e:
            print(f"Error processing paper {paper.title}: {str(e)}")
    
    # Return the structured results
    return {
        "papers": processed_papers,
        "insights": insights,
        "github_repositories": github_repositories,
        "other_resources": other_resources,
    }


async def search_papers_from_multiple_queries(
    queries: List[str],
    llm_name: str = "gpt-4o",
    save_dir: str = "./data",
    max_papers_per_query: int = 3,
) -> WebSearchResultData:
    """
    Search papers from multiple queries, combining the results.
    
    Args:
        queries: List of search queries
        llm_name: Name of the LLM to use
        save_dir: Directory to save papers
        max_papers_per_query: Maximum papers to process per query
        
    Returns:
        Combined WebSearchResultData
    """
    # Initialize result containers
    all_papers = []
    all_insights = []
    all_github_repositories = []
    all_other_resources = []
    
    # Process each query
    for query in queries:
        try:
            # Search papers for this query
            result = await search_papers_with_web_api(
                query=query,
                llm_name=llm_name,
                save_dir=save_dir,
                max_papers=max_papers_per_query,
            )
            
            # Add results to combined lists
            all_papers.extend(result["papers"])
            all_insights.extend(result["insights"])
            all_github_repositories.extend(result["github_repositories"])
            all_other_resources.extend(result["other_resources"])
            
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}")
    
    # Remove duplicates
    unique_papers = []
    seen_arxiv_ids = set()
    
    for paper in all_papers:
        if paper.arxiv_id not in seen_arxiv_ids:
            unique_papers.append(paper)
            seen_arxiv_ids.add(paper.arxiv_id)
    
    unique_insights = list(dict.fromkeys(all_insights))
    unique_github_repositories = list(dict.fromkeys(all_github_repositories))
    unique_other_resources = list(dict.fromkeys(all_other_resources))
    
    # Return the combined results
    return {
        "papers": unique_papers,
        "insights": unique_insights,
        "github_repositories": unique_github_repositories,
        "other_resources": unique_other_resources,
    }


# Example usage
if __name__ == "__main__":
    async def main():
        save_dir = os.path.join(os.getcwd(), "data")
        result = await search_papers_with_web_api(
            query="transformer architecture advancements in NLP",
            llm_name="gpt-4o",
            save_dir=save_dir,
        )
        
        print("\n=== Processed Papers ===")
        for i, paper in enumerate(result["papers"], 1):
            print(f"{i}. {paper.title}")
            print(f"   ArXiv ID: {paper.arxiv_id}")
            print(f"   GitHub: {paper.github_url or 'None'}")
            print(f"   Main contributions: {paper.main_contributions[:100]}...")
            print()
        
        print(f"\nFound {len(result['papers'])} papers, {len(result['github_repositories'])} GitHub repositories")
        print(f"Extracted {len(result['insights'])} insights")
    
    asyncio.run(main())
