"""
OpenAI Web Search Node

This module provides functionality to use OpenAI's web search capabilities
through the OpenAI API to find academic papers and research information.
"""

import os
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel
from openai import OpenAI

import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_api_key


class WebSearchResult(BaseModel):
    """Structured result from OpenAI web search"""
    title: str
    url: str
    snippet: str
    source_type: str = "web"  # Can be "web", "paper", "repository", etc.


class OpenAIWebSearchResponse(TypedDict):
    """Structure for OpenAI API response"""
    results: List[WebSearchResult]
    insights: List[str]
    follow_up_questions: List[str]
    raw_response: str


class OpenAIWebSearchNode:
    """
    Node for searching the web using OpenAI's web search capabilities.
    
    This node leverages OpenAI's web_search_preview tool to obtain comprehensive
    search results for academic queries.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_results: int = 10,
        include_follow_up: bool = True,
        focus_on_academic: bool = True
    ):
        """
        Initialize the OpenAI Web Search Node.
        
        Args:
            model: The OpenAI model to use for search
            max_results: Maximum number of results to return
            include_follow_up: Whether to generate follow-up questions
            focus_on_academic: Whether to focus search on academic sources
        """
        self.model = model
        self.max_results = max_results
        self.include_follow_up = include_follow_up
        self.focus_on_academic = focus_on_academic
        
        # Ensure OpenAI API key is available
        check_api_key()
        self.client = OpenAI()
    
    def execute(self, query: str) -> OpenAIWebSearchResponse:
        """
        Execute a web search using OpenAI's search capabilities.
        
        Args:
            query: The search query to execute
            
        Returns:
            A dictionary containing search results, insights, and follow-up questions
        """
        print(f"Executing OpenAI web search for: '{query}'")
        
        # Enhance query for academic focus if requested
        search_query = query
        if self.focus_on_academic:
            search_query = self._enhance_academic_query(query)
            print(f"Enhanced academic query: '{search_query}'")
        
        # Construct the prompt for the API
        system_message = (
            "You are a research assistant helping to find academic papers and research information. "
            "Use the web search capability to find relevant, recent, and high-quality academic sources. "
            "Focus on finding papers from reputable conferences and journals like NeurIPS, ICML, ICLR, ACL, CVPR, etc. "
            "For each result, extract the title, URL, and a brief snippet of information."
        )
        
        prompt = f"""
        Please provide a comprehensive search for: {search_query}
        
        Return the results with these components:
        1. A list of the most relevant search results (title, URL, and brief snippet)
        2. Key insights extracted from these results
        3. Suggested follow-up questions to deepen the research
        
        Prioritize academic papers, especially those with code implementations on GitHub or similar repositories.
        """
        
        try:
            # Execute the OpenAI API call with web search
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                tools=[{"type": "web_search"}],
                tool_choice="auto"
            )
            
            # Extract the response content
            raw_response = response.choices[0].message.content or ""
            
            # Process and structure the results
            results, insights, follow_up_questions = self._process_response(raw_response)
            
            return {
                "results": results[:self.max_results],
                "insights": insights,
                "follow_up_questions": follow_up_questions if self.include_follow_up else [],
                "raw_response": raw_response
            }
            
        except Exception as e:
            print(f"Error executing OpenAI web search: {str(e)}")
            return {
                "results": [],
                "insights": [f"Search error: {str(e)}"],
                "follow_up_questions": [],
                "raw_response": ""
            }
    
    def _enhance_academic_query(self, query: str) -> str:
        """
        Enhance a query to focus on academic sources.
        
        Args:
            query: The original search query
            
        Returns:
            Enhanced query focused on academic sources
        """
        # Add academic keywords to the query
        academic_terms = [
            "research paper", 
            "academic publication", 
            "NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "AAAI",
            "GitHub implementation", "code repository"
        ]
        
        # Choose a couple of terms to add
        import random
        selected_terms = random.sample(academic_terms, min(3, len(academic_terms)))
        
        # Add the terms to the query
        enhanced_query = f"{query} {' '.join(selected_terms)}"
        return enhanced_query
    
    def _process_response(self, response: str) -> tuple[List[WebSearchResult], List[str], List[str]]:
        """
        Process the raw response from OpenAI API into structured data.
        
        Args:
            response: Raw text response from OpenAI
            
        Returns:
            Tuple of (results, insights, follow_up_questions)
        """
        results = []
        insights = []
        follow_up_questions = []
        
        # Simple parsing logic - this could be enhanced with more robust parsing
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if "search result" in line.lower() or "relevant paper" in line.lower():
                current_section = "results"
                continue
            elif "insight" in line.lower() or "key finding" in line.lower():
                current_section = "insights"
                continue
            elif "follow" in line.lower() and "question" in line.lower():
                current_section = "questions"
                continue
                
            # Process content based on current section
            if current_section == "results":
                # Try to extract a result
                if "http" in line or "https" in line or "arxiv.org" in line:
                    # This line might contain a URL
                    parts = line.split("http")
                    title = parts[0].strip().rstrip(":.,-")
                    url = "http" + parts[1].split()[0].strip()
                    snippet = line.replace(title, "").replace(url, "").strip()
                    
                    results.append(WebSearchResult(
                        title=title or "Untitled",
                        url=url,
                        snippet=snippet or "No snippet available",
                        source_type="paper" if "arxiv" in url else "web"
                    ))
            elif current_section == "insights":
                # Add insight, removing numbering if present
                if line and not line.startswith("#"):
                    # Remove numbering patterns like "1.", "1)", "[1]"
                    cleaned_line = line
                    if line[0].isdigit() and line[1:3] in ['. ', ') ', '- ']:
                        cleaned_line = line[3:]
                    elif line[0:3] in ['1. ', '[1]']:
                        cleaned_line = line[3:]
                    
                    insights.append(cleaned_line)
            elif current_section == "questions":
                # Add follow-up question, removing numbering if present
                if line and not line.startswith("#"):
                    # Remove numbering patterns
                    cleaned_line = line
                    if line[0].isdigit() and line[1:3] in ['. ', ') ', '- ']:
                        cleaned_line = line[3:]
                    elif line[0:3] in ['1. ', '[1]']:
                        cleaned_line = line[3:]
                    
                    follow_up_questions.append(cleaned_line)
        
        return results, insights, follow_up_questions


# Example usage
if __name__ == "__main__":
    node = OpenAIWebSearchNode()
    result = node.execute("transformer architecture advancements NeurIPS")
    
    print("=== Search Results ===")
    for i, res in enumerate(result["results"], 1):
        print(f"{i}. {res.title}")
        print(f"   URL: {res.url}")
        print(f"   Snippet: {res.snippet[:100]}...")
        print()
    
    print("\n=== Insights ===")
    for i, insight in enumerate(result["insights"], 1):
        print(f"{i}. {insight}")
    
    print("\n=== Follow-up Questions ===")
    for i, question in enumerate(result["follow_up_questions"], 1):
        print(f"{i}. {question}")
