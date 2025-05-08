"""
Retrieve Paper Subgraph New

This module implements a simplified paper retrieval subgraph using OpenAI's web search API
and arXiv API to find academic papers and research information.
"""

import os
import json
from typing import List, Dict, Any, TypedDict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Import ArxivNode for direct arXiv API access
from researchgraph.retrieve_paper_subgraph_new.nodes.arxiv_api_node import ArxivNode

# Load environment variables from .env file
load_dotenv()

class CandidatePaperInfo(BaseModel):
    """Information about a candidate paper"""
    arxiv_id: str = ""
    arxiv_url: str = ""
    title: str = ""
    authors: List[str] = []
    published_date: str = ""
    journal: str = ""
    doi: str = ""
    summary: str = ""
    github_url: str = ""
    main_contributions: str = ""
    methodology: str = ""
    experimental_setup: str = ""
    limitations: str = ""
    future_research_directions: str = ""


class RetrievePaperNewState(TypedDict):
    """State for the paper retrieval process"""
    # Input
    queries: List[str]
    save_dir: str
    
    # Output
    search_results: List[Dict[str, Any]]
    papers: List[CandidatePaperInfo]
    insights: List[str]
    research_report: str


class RetrievePaperSubgraphNew:
    """
    Subgraph for retrieving and analyzing research papers using OpenAI's web search.
    
    This implementation uses the OpenAI API to search for papers and extract information.
    """
    
    def __init__(
        self,
        llm_name: str = "gpt-4o",
        save_dir: str = "./data",
        arxiv_query_batch_size: int = 10,
        arxiv_num_retrieve_paper: int = 5,
        arxiv_period_days: Optional[int] = None
    ):
        """
        Initialize the paper retrieval subgraph.
        
        Args:
            llm_name: OpenAI model to use
            save_dir: Directory to save output files
            arxiv_query_batch_size: Number of queries to send to arXiv API at once
            arxiv_num_retrieve_paper: Number of papers to retrieve from arXiv per query
            arxiv_period_days: Number of days to limit arXiv search to
        """
        self.llm_name = llm_name
        self.save_dir = save_dir
        self.client = OpenAI()
        
        # ArXiv search parameters
        self.arxiv_query_batch_size = arxiv_query_batch_size
        self.arxiv_num_retrieve_paper = arxiv_num_retrieve_paper
        self.arxiv_period_days = arxiv_period_days
        
        # Initialize ArxivNode
        self.arxiv_node = ArxivNode(
            num_retrieve_paper=arxiv_num_retrieve_paper,
            period_days=arxiv_period_days
        )
        
        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "json"), exist_ok=True)
    
    def search_papers(self, queries: List[str]) -> Dict[str, Any]:
        """
        Search for papers using OpenAI's web search capability with multiple queries
        to get more comprehensive results.
        
        Args:
            queries: List of search queries
            
        Returns:
            Dictionary with search results
        """
        print(f"Searching for papers with queries: {queries}")
        
        combined_results = {
            "search_results": [],
            "papers": [],
            "insights": []
        }
        
        for query in queries:
            print(f"Processing query: {query}")
            
            # Make multiple API calls with different focuses to get more comprehensive results
            search_queries = [
                # Main query for papers and methodology
                f"{query} research papers from conferences such as NeurIPS, ICML, ICLR with methodology",
                
                # Specific query for GitHub repositories related to papers
                f"{query} github repositories for academic papers",
                
                # Specific query for methodology details
                f"detailed methodology of {query} papers"
            ]
            
            papers_from_query = []
            insights_from_query = []
            
            # Make multiple API calls with different focuses
            for search_query in search_queries:
                print(f"Making API call with focus: {search_query}")
                
                # The prompt for the API
                prompt = f"""
                Please search for academic papers related to: {search_query}
                
                Return your response in the following JSON format:
                {{
                    "base_paper": {{
                        "title": "Title of the most relevant paper",
                        "github_url": "GitHub repository URL associated with the paper (if available, otherwise empty string)",
                        "methodology": "Detailed explanation of the methodology used in the paper"
                    }},
                    "additional_papers": [
                        {{
                            "title": "Title of additional paper 1",
                            "github_url": "GitHub repository URL for paper 1 (if available, otherwise empty string)",
                            "methodology": "Detailed explanation of the methodology used in paper 1"
                        }},
                        {{
                            "title": "Title of additional paper 2",
                            "github_url": "GitHub repository URL for paper 2 (if available, otherwise empty string)",
                            "methodology": "Detailed explanation of the methodology used in paper 2"
                        }}
                        // Include about 3-5 additional papers
                    ],
                    "insights": [
                        "Key insight 1 about the research area",
                        "Key insight 2 about the research area",
                        // Several insights from the papers
                    ]
                }}
                
                Focus on finding papers with associated GitHub repositories or code implementations. Include detailed methodology descriptions for each paper.
                """
                
                # Execute the OpenAI API call with web search using the Responses API
                response = self.client.responses.create(
                    model=self.llm_name,
                    tools=[{"type": "web_search_preview"}],
                    input=prompt
                )
                
                # Extract content from the response
                content = response.output_text
                
                if not content:
                    content = "No content or search results found for the query."
                
                # Store the raw search result
                combined_results["search_results"].append({
                    "query": search_query,
                    "response": content
                })
                
                # Process the response to extract papers
                papers, insights = self._process_search_response(content)
                
                # Add to query results
                papers_from_query.extend(papers)
                insights_from_query.extend(insights)
            
            # Merge papers with the same title or GitHub URL
            merged_papers = self._merge_papers(papers_from_query)
            
            # Add to combined results
            combined_results["papers"].extend(merged_papers)
            combined_results["insights"].extend(insights_from_query)
        
        # Remove duplicates from insights
        combined_results["insights"] = list(dict.fromkeys(combined_results["insights"]))
        
        return combined_results
            
    def _merge_papers(self, papers: List[CandidatePaperInfo]) -> List[CandidatePaperInfo]:
        """
        Merge papers with the same title or GitHub URL to avoid duplicates.
        
        Args:
            papers: List of papers to merge
            
        Returns:
            List of merged papers
        """
        # Dictionary to store merged papers by title or GitHub URL
        merged_dict = {}
        
        for paper in papers:
            # Use title as key if available, otherwise use GitHub URL
            key = paper.title.lower() if paper.title else (paper.github_url if paper.github_url else None)
            
            if not key:
                # If no title or GitHub URL, just add the paper
                merged_dict[f"unknown_{len(merged_dict)}"] = paper
                continue
                
            if key in merged_dict:
                # Merge with existing paper
                existing = merged_dict[key]
                
                # Use non-empty values from either paper
                if not existing.github_url and paper.github_url:
                    existing.github_url = paper.github_url
                    
                if not existing.methodology and paper.methodology:
                    existing.methodology = paper.methodology
                    
                if not existing.authors and paper.authors:
                    existing.authors = paper.authors
                    
                # Merge other fields if needed
                if not existing.arxiv_url and paper.arxiv_url:
                    existing.arxiv_url = paper.arxiv_url
                    
                if not existing.arxiv_id and paper.arxiv_id:
                    existing.arxiv_id = paper.arxiv_id
            else:
                # Add new paper
                merged_dict[key] = paper
        
        # Return list of merged papers
        return list(merged_dict.values())
    
    def _process_search_response(self, response_text: str) -> tuple[List[CandidatePaperInfo], List[str]]:
        """
        Process the search response to extract papers and insights.
        
        Args:
            response_text: Raw text response from OpenAI
            
        Returns:
            Tuple of (papers, insights)
        """
        papers = []
        insights = []
        
        # First, try to extract JSON data from the response
        try:
            # Look for JSON content in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                
                # Try to parse the JSON
                try:
                    data = json.loads(json_content)
                    print("Successfully parsed JSON from response")
                    
                    # Extract papers from the JSON structure
                    if "base_paper" in data:
                        base_paper = data["base_paper"]
                        papers.append(CandidatePaperInfo(
                            title=base_paper.get("title", ""),
                            github_url=base_paper.get("github_url", ""),
                            methodology=base_paper.get("methodology", ""),
                        ))
                    
                    if "additional_papers" in data:
                        for paper in data["additional_papers"]:
                            if isinstance(paper, dict):
                                papers.append(CandidatePaperInfo(
                                    title=paper.get("title", ""),
                                    github_url=paper.get("github_url", ""),
                                    methodology=paper.get("methodology", ""),
                                ))
                    
                    # Extract insights
                    if "insights" in data and isinstance(data["insights"], list):
                        insights.extend(data["insights"])
                    
                    # If we successfully parsed JSON, return the data
                    if papers:
                        return papers, insights
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}")
                    # Continue with fallback text parsing
        except Exception as e:
            print(f"Error extracting JSON: {e}")
        
        # Fallback text parsing if JSON extraction failed
        print("Using fallback text parsing")
        
        # Store all lines as insights for comprehensive coverage
        lines = response_text.split('\n')
        insights = [line.strip() for line in lines if line.strip()]
        
        # Try to identify papers and GitHub repositories from the text
        current_paper = {}
        paper_data = []
        
        # Look for patterns like "Title:" or "GitHub:" in the text
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Title detection
            if "title:" in line.lower() or "**title" in line.lower():
                # Save existing paper if we have one
                if current_paper and "title" in current_paper:
                    paper_data.append(current_paper.copy())
                    
                # Start a new paper
                current_paper = {}
                # Extract title
                if ":" in line:
                    title = line.split(":", 1)[1].strip().strip('"').strip("'")
                    current_paper["title"] = title
            
            # GitHub URL detection
            elif "github" in line.lower() and "http" in line.lower():
                if current_paper:  # Add to current paper if we have one
                    # Extract GitHub URL
                    url_start = line.find("http")
                    if url_start >= 0:
                        url_end = len(line)
                        for delimiter in [" ", ",", ")", "]", '"', "'"]:
                            pos = line.find(delimiter, url_start)
                            if pos > 0:
                                url_end = min(url_end, pos)
                        github_url = line[url_start:url_end]
                        if "github.com" in github_url:
                            current_paper["github_url"] = github_url
                
            # Methodology or description
            elif "methodology:" in line.lower() or "description:" in line.lower():
                if current_paper:
                    if ":" in line:
                        methodology = line.split(":", 1)[1].strip()
                        current_paper["methodology"] = methodology
        
        # Don't forget the last paper
        if current_paper and "title" in current_paper:
            paper_data.append(current_paper.copy())
            
        # If we found papers, convert them to CandidatePaperInfo objects
        for paper in paper_data:
            paper_obj = CandidatePaperInfo(
                title=paper.get("title", ""),
                github_url=paper.get("github_url", ""),
                methodology=paper.get("methodology", ""),
            )
            papers.append(paper_obj)
            
        # If we still haven't found any papers, try direct extraction of GitHub URLs
        if not papers:
            github_urls = []
            for line in lines:
                if "github.com" in line:
                    # Try to extract GitHub URL from text
                    if "[" in line and "]" in line and "(" in line and ")" in line:
                        start_idx = line.find("(") + 1
                        end_idx = line.find(")", start_idx)
                        url = line[start_idx:end_idx]
                        if "github.com" in url:
                            github_urls.append(url)
                    # Direct URL mentions
                    elif "https://github.com" in line:
                        start_idx = line.find("https://github.com")
                        end_idx = min(start_idx + 100, len(line))
                        url_candidate = line[start_idx:end_idx]
                        # Cut at the first delimiter
                        for delimiter in [" ", ")", "]", ",", ";", '"', "'"]:
                            if delimiter in url_candidate:
                                url_candidate = url_candidate.split(delimiter)[0]
                        if url_candidate and len(url_candidate) > 15:
                            github_urls.append(url_candidate)
            
            # Create papers from GitHub URLs
            if github_urls:
                for i, url in enumerate(github_urls):
                    papers.append(CandidatePaperInfo(
                        title=f"Paper from GitHub URL #{i+1}",
                        github_url=url,
                        methodology=f"GitHub repository: {url}"
                    ))
        
        return papers, insights
    
    def _merge_arxiv_with_web_papers(self, web_papers: List[CandidatePaperInfo], arxiv_papers: List[CandidatePaperInfo]) -> List[CandidatePaperInfo]:
        """
        Merge papers from web search with papers from arXiv.
        
        Args:
            web_papers: Papers from web search
            arxiv_papers: Papers from arXiv
            
        Returns:
            List of merged papers
        """
        # Dictionary to store merged papers
        merged_dict = {}
        
        # Add web papers first
        for paper in web_papers:
            # Use title as key, lowercase for case-insensitive comparison
            key = paper.title.lower() if paper.title else f"unknown_{len(merged_dict)}"
            merged_dict[key] = paper
        
        # Add or merge arXiv papers
        for arxiv_paper in arxiv_papers:
            # Use title as key for matching
            key = arxiv_paper.title.lower() if arxiv_paper.title else None
            
            if key and key in merged_dict:
                # Merge with existing paper if title matches
                existing = merged_dict[key]
                
                # Update arXiv-specific fields
                if not existing.arxiv_id and arxiv_paper.arxiv_id:
                    existing.arxiv_id = arxiv_paper.arxiv_id
                    
                if not existing.arxiv_url and arxiv_paper.arxiv_url:
                    existing.arxiv_url = arxiv_paper.arxiv_url
                    
                if not existing.summary and arxiv_paper.summary:
                    existing.summary = arxiv_paper.summary
                    
                if not existing.authors and arxiv_paper.authors:
                    existing.authors = arxiv_paper.authors
                    
                if not existing.published_date and arxiv_paper.published_date:
                    existing.published_date = arxiv_paper.published_date
            elif arxiv_paper.arxiv_id:
                # If no matching title but has arXiv ID, add as new paper
                new_key = f"arxiv_{arxiv_paper.arxiv_id}"
                merged_dict[new_key] = arxiv_paper
        
        # Return list of merged papers
        return list(merged_dict.values())
    
    # Report generation functionality removed as requested
    
    def run(self, queries: List[str], save_dir: str = None) -> Dict[str, Any]:
        """
        Run the paper retrieval and analysis process.
        
        Args:
            queries: List of search queries
            save_dir: Directory to save output files (optional)
            
        Returns:
            Dictionary with:
            - base_github_url: GitHub URL of the base paper
            - base_method_text: Methodology text of the base paper
            - add_github_urls: List of GitHub URLs of additional papers
            - add_method_texts: List of methodology texts of additional papers
        """
        if save_dir:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "reports"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "json"), exist_ok=True)
        
        # Search for papers using web search
        search_results = self.search_papers(queries)
        
        # Search for papers using arXiv API
        print("Searching arXiv for additional paper information...")
        arxiv_results = self.arxiv_node.execute(queries)
        
        # Convert arXiv results to CandidatePaperInfo objects
        arxiv_papers = []
        for paper_info in arxiv_results:
            paper = CandidatePaperInfo(
                arxiv_id=paper_info.get("arxiv_id", ""),
                arxiv_url=paper_info.get("arxiv_url", ""),
                title=paper_info.get("title", ""),
                authors=paper_info.get("authors", []),
                published_date=paper_info.get("published_date", ""),
                summary=paper_info.get("summary", "")
            )
            arxiv_papers.append(paper)
        
        # Merge arXiv papers with web search papers
        all_papers = self._merge_arxiv_with_web_papers(search_results["papers"], arxiv_papers)
        search_results["papers"] = all_papers
        
        # Save the internal results to a JSON file (without report)
        internal_results = {
            "papers": [paper.model_dump() for paper in search_results["papers"]],
            "insights": search_results["insights"]
        }
        
        json_path = os.path.join(self.save_dir, "json", "paper_search_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(internal_results, f, indent=2, ensure_ascii=False)
        
        # Extract base paper and add papers
        papers = search_results["papers"]
        
        # For compatibility with the original subgraph interface
        if papers:
            # Use the first paper as the base paper
            base_paper = papers[0]
            base_github_url = base_paper.github_url
            base_method_text = base_paper.methodology
            
            # Use the rest of the papers as additional papers
            add_papers = papers[1:] if len(papers) > 1 else []
            add_github_urls = [paper.github_url for paper in add_papers]
            add_method_texts = [paper.methodology for paper in add_papers]
        else:
            # No papers found
            base_github_url = ""
            base_method_text = ""
            add_github_urls = []
            add_method_texts = []
        
        # Return in the format matching the original subgraph
        return {
            "base_github_url": base_github_url,
            "base_method_text": base_method_text,
            "add_github_urls": add_github_urls,
            "add_method_texts": add_method_texts,
        }


# Example usage
if __name__ == "__main__":
    # Set up the save directory
    save_dir = os.path.abspath(os.path.join(os.getcwd(), "data"))
    
    # Initialize the subgraph
    retriever = RetrievePaperSubgraphNew(
        llm_name="gpt-4o",
        save_dir=save_dir
    )
    
    # Run the retrieval process
    results = retriever.run(
        queries=["Adam optimizer improvements in deep learning"]
    )
    
    # Read internal results from JSON for display purposes
    json_path = os.path.join(save_dir, "json", "paper_search_result.json")
    with open(json_path, "r", encoding="utf-8") as f:
        internal_results = json.load(f)
    
    # Print a summary of the results
    print("\n=== Summary of Results ===")
    print(f"Found {len(internal_results['papers'])} papers")
    print(f"Extracted {len(internal_results['insights'])} insights")
    
    # Print output format for compatibility with original subgraph
    print("\n=== Output Format (Matching Original Subgraph) ===")
    print(f"Base GitHub URL: {results['base_github_url']}")
    print(f"Base Method Text: {results['base_method_text'][:100]}..." if results['base_method_text'] else "None")
    print(f"Additional GitHub URLs: {len(results['add_github_urls'])}")
    print(f"Additional Method Texts: {len(results['add_method_texts'])}")
    
    # Print the first few papers from internal results
    print("\n=== Papers ===")
    for i, paper in enumerate(internal_results["papers"][:3], 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        if paper["github_url"]:
            print(f"   GitHub: {paper['github_url']}")
        print()
    
    # Print the first few insights
    print("\n=== Insights ===")
    for i, insight in enumerate(internal_results["insights"][:5], 1):
        print(f"{i}. {insight}")
    
    # Report preview removed
    print(f"JSON results saved to: {os.path.join(save_dir, 'json', 'paper_search_result.json')}")
