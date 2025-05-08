"""
Extract Papers From Web Search Results

This module processes the results from OpenAI web search to extract papers,
identify their arxiv IDs, and prepare them for further processing.
"""

import re
from typing import List, Dict, Any, Optional
import asyncio
from pydantic import BaseModel

from researchgraph.retrieve_paper_subgraph_new.nodes.openai_web_search_node import WebSearchResult


class ExtractedPaper(BaseModel):
    """Structure for extracted paper information"""
    title: str
    url: str
    arxiv_id: Optional[str] = None
    snippet: str
    is_arxiv: bool = False
    is_github: bool = False
    confidence: float = 0.0  # How confident we are this is a research paper


class PaperExtractionResult(BaseModel):
    """Results from paper extraction process"""
    papers: List[ExtractedPaper]
    github_repositories: List[str] = []
    other_resources: List[str] = []
    academic_insights: List[str] = []


def extract_papers_from_web_search(search_results: List[WebSearchResult]) -> PaperExtractionResult:
    """
    Extract paper information from web search results.
    
    Args:
        search_results: List of WebSearchResult objects from OpenAI web search
        
    Returns:
        PaperExtractionResult with structured paper information
    """
    papers = []
    github_repositories = []
    other_resources = []
    academic_insights = []
    
    # Regex pattern for arxiv IDs
    arxiv_pattern = r'arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)'
    
    for result in search_results:
        # Skip empty results
        if not result.url:
            continue
            
        url = result.url.lower()
        title = result.title
        snippet = result.snippet
        
        # Check if it's an arxiv paper
        if 'arxiv.org' in url:
            is_arxiv = True
            arxiv_match = re.search(arxiv_pattern, url)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None
            confidence = 0.9
        else:
            is_arxiv = False
            arxiv_id = None
            
            # Estimate if this is a paper based on URL and title patterns
            confidence = _estimate_paper_confidence(url, title)
        
        # Check if it's a GitHub repository
        is_github = 'github.com' in url
        
        # If it's a GitHub repo, add to repositories list
        if is_github:
            github_repositories.append(url)
            confidence = min(confidence, 0.5)  # Less likely to be a paper if it's a repo
        
        # Create ExtractedPaper object
        paper = ExtractedPaper(
            title=title,
            url=url,
            arxiv_id=arxiv_id,
            snippet=snippet,
            is_arxiv=is_arxiv,
            is_github=is_github,
            confidence=confidence
        )
        
        # Add to appropriate list based on confidence
        if confidence > 0.6:
            papers.append(paper)
        elif is_github:
            # Already added to github_repositories
            pass
        else:
            other_resources.append(url)
    
    return PaperExtractionResult(
        papers=sorted(papers, key=lambda x: x.confidence, reverse=True),
        github_repositories=github_repositories,
        other_resources=other_resources,
        academic_insights=academic_insights
    )


def _estimate_paper_confidence(url: str, title: str) -> float:
    """
    Estimate confidence that a result is an academic paper based on URL and title.
    
    Args:
        url: The URL of the search result
        title: The title of the search result
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 0.0
    
    # Check URL for academic domains
    academic_domains = [
        'arxiv.org',
        'researchgate.net',
        'dl.acm.org',
        'ieeexplore.ieee.org',
        'springer.com',
        'sciencedirect.com',
        'nature.com',
        'semanticscholar.org',
    ]
    
    for domain in academic_domains:
        if domain in url:
            confidence += 0.3
            break
    
    # Check URL for conference patterns
    conference_patterns = [
        'proceedings', 'conf', 'conference', 'symposium', 
        'workshop', 'journal', 'transactions'
    ]
    
    for pattern in conference_patterns:
        if pattern in url.lower():
            confidence += 0.1
            break
    
    # Check title for academic patterns
    title_lower = title.lower()
    paper_title_patterns = [
        'paper', 'research', 'study', 'analysis', 'survey', 
        'review', 'method', 'approach', 'framework', 'algorithm'
    ]
    
    for pattern in paper_title_patterns:
        if pattern in title_lower:
            confidence += 0.1
            break
    
    # Check for PDF link
    if url.endswith('.pdf'):
        confidence += 0.2
    
    # Cap confidence at 0.8 for non-arxiv papers
    return min(0.8, confidence)


# Example usage
if __name__ == "__main__":
    # Create some example results
    results = [
        WebSearchResult(
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            snippet="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms..."
        ),
        WebSearchResult(
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            url="https://github.com/google-research/bert",
            snippet="This repository contains TensorFlow code for the BERT model..."
        ),
        WebSearchResult(
            title="Recent Advances in Transformer Architectures for NLP",
            url="https://www.example.com/blog/transformer-advances",
            snippet="An overview of recent improvements to transformer models..."
        )
    ]
    
    extraction_result = extract_papers_from_web_search(results)
    
    print("=== Extracted Papers ===")
    for i, paper in enumerate(extraction_result.papers, 1):
        print(f"{i}. {paper.title} (confidence: {paper.confidence:.2f})")
        print(f"   URL: {paper.url}")
        print(f"   ArXiv ID: {paper.arxiv_id or 'N/A'}")
        print()
    
    print("\n=== GitHub Repositories ===")
    for i, repo in enumerate(extraction_result.github_repositories, 1):
        print(f"{i}. {repo}")
