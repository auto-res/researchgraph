"""
Web Enhanced Paper Retrieval Subgraph

This module implements a paper retrieval subgraph that uses both traditional
academic search methods and OpenAI's web search capabilities to find relevant
papers and research information.
"""

import os
import asyncio
from typing import Annotated, TypedDict, Optional, List
from pydantic import BaseModel
import operator

from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.retrieve_paper_subgraph_new.nodes.recursive_paper_search import (
    CandidatePaperInfo,
)
from researchgraph.retrieve_paper_subgraph_new.nodes.web_enhanced_paper_search import (
    web_enhanced_paper_search,
)
from researchgraph.retrieve_paper_subgraph_new.nodes.select_best_paper_with_context import (
    select_best_paper_with_context,
)
from researchgraph.retrieve_paper_subgraph_new.nodes.generate_enhanced_queries import (
    generate_enhanced_queries,
)
from researchgraph.retrieve_paper_subgraph_new.nodes.generate_report import (
    generate_markdown_report,
)
from researchgraph.github_utils.graph_wrapper import create_wrapped_subgraph


class WebEnhancedRetrievePaperState(TypedDict):
    # Input
    queries: List[str]
    save_dir: str
    web_search_enabled: bool

    # Internal state
    learnings: List[str]
    visited_urls: List[str]
    paper_candidates: List[CandidatePaperInfo]
    github_repositories: List[str]
    
    enhanced_queries: List[str]

    add_learnings: List[str]
    add_visited_urls: List[str]
    add_paper_candidates: List[CandidatePaperInfo]
    add_github_repositories: List[str]

    # Selected papers
    selected_base_paper_info: CandidatePaperInfo
    selected_add_paper_info: CandidatePaperInfo

    # Output
    base_github_url: str
    base_method_text: str
    add_github_url: str
    add_method_text: str
    research_report: str


class WebEnhancedPaperSubgraph:
    """
    Subgraph for retrieving and analyzing research papers using web-enhanced search.
    
    This implementation combines traditional academic search methods with OpenAI's
    web search capabilities to provide more comprehensive and up-to-date results.
    """
    
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
        breadth: int = 2,
        depth: int = 1,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        self.breadth = breadth
        self.depth = depth
    
    def _initialize_state(self, state: WebEnhancedRetrievePaperState) -> dict:
        print("---WebEnhancedPaperSubgraph---")
        return {
            "queries": state["queries"],
            "save_dir": state.get("save_dir", self.save_dir),
            "web_search_enabled": state.get("web_search_enabled", True),
            "learnings": [],
            "visited_urls": [],
            "paper_candidates": [],
            "github_repositories": [],
            "add_learnings": [],
            "add_visited_urls": [],
            "add_paper_candidates": [],
            "add_github_repositories": [],
        }
    
    async def _web_enhanced_search(self, state: WebEnhancedRetrievePaperState) -> dict:
        print("web_enhanced_search")
        
        result = await web_enhanced_paper_search(
            queries=state["queries"],
            llm_name=self.llm_name,
            save_dir=state["save_dir"],
            breadth=self.breadth,
            depth=self.depth,
            web_search_enabled=state["web_search_enabled"],
        )
        
        return {
            "learnings": result["insights"],
            "visited_urls": result["visited_urls"],
            "paper_candidates": result["papers"],
            "github_repositories": result["github_repositories"],
        }
    
    def _select_base_paper_node(self, state: WebEnhancedRetrievePaperState) -> dict:
        print("select_base_paper_node")
        
        selected_paper = select_best_paper_with_context(
            llm_name=self.llm_name,
            candidate_papers=state["paper_candidates"],
            learnings=state["learnings"],
        )
        
        return {
            "selected_base_paper_info": selected_paper,
        }
    
    def _generate_enhanced_queries_node(self, state: WebEnhancedRetrievePaperState) -> dict:
        print("generate_enhanced_queries_node")
        
        generated_queries = generate_enhanced_queries(
            llm_name=self.llm_name,
            base_paper=state["selected_base_paper_info"],
            learnings=state["learnings"],
        )
        
        return {
            "enhanced_queries": generated_queries,
        }
    
    async def _web_enhanced_add_search(self, state: WebEnhancedRetrievePaperState) -> dict:
        print("web_enhanced_add_search")
        
        result = await web_enhanced_paper_search(
            queries=state["enhanced_queries"],
            llm_name=self.llm_name,
            save_dir=state["save_dir"],
            breadth=self.breadth,
            depth=self.depth,
            web_search_enabled=state["web_search_enabled"],
        )
        
        return {
            "add_learnings": result["insights"],
            "add_visited_urls": result["visited_urls"],
            "add_paper_candidates": result["papers"],
            "add_github_repositories": result["github_repositories"],
        }
    
    def _select_add_paper_node(self, state: WebEnhancedRetrievePaperState) -> dict:
        print("select_add_paper_node")
        
        # Check if there are any candidate papers
        if not state["add_paper_candidates"]:
            print("Warning: No additional paper candidates found. Using default paper.")
        
        selected_paper = select_best_paper_with_context(
            llm_name=self.llm_name,
            candidate_papers=state["add_paper_candidates"],
            learnings=state["add_learnings"],
            base_paper=state["selected_base_paper_info"],
        )
        
        return {
            "selected_add_paper_info": selected_paper,
        }
    
    def _prepare_output(self, state: WebEnhancedRetrievePaperState) -> dict:
        print("prepare_output")
        
        base_github_url = state["selected_base_paper_info"].github_url
        base_method_text = state["selected_base_paper_info"].model_dump_json()
        add_github_url = state["selected_add_paper_info"].github_url
        add_method_text = state["selected_add_paper_info"].model_dump_json()
        
        # Merge learnings for the report
        all_learnings = state["learnings"] + state["add_learnings"]
        all_visited_urls = state["visited_urls"] + state["add_visited_urls"]
        
        # Generate Markdown report
        print("Generating research report...")
        report_data = generate_markdown_report(
            base_paper=state["selected_base_paper_info"],
            add_paper=state["selected_add_paper_info"],
            base_learnings=state["learnings"],
            add_learnings=state["add_learnings"],
            base_visited_urls=state["visited_urls"],
            add_visited_urls=state["add_visited_urls"],
            max_learnings=10,  # Limit learnings to 10 items
        )
        
        # Get the Markdown report
        research_report = report_data["markdown"]
        
        # Create directories for saving files
        papers_dir = os.path.join(state["save_dir"], "papers")
        reports_dir = os.path.join(state["save_dir"], "reports")
        json_dir = os.path.join(state["save_dir"], "json")
        
        # Ensure directories exist
        os.makedirs(papers_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        
        # Save the report to a file
        report_path = os.path.join(reports_dir, "research_report.md")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(research_report)
            print(f"Research report saved to: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save research report to file: {e}")
        
        # Save paper information as JSON
        import json
        
        # Base paper
        base_paper_json_path = os.path.join(json_dir, "base_paper_info.json")
        try:
            with open(base_paper_json_path, "w", encoding="utf-8") as f:
                f.write(base_method_text)
            print(f"Base paper info saved to: {base_paper_json_path}")
        except Exception as e:
            print(f"Warning: Could not save base paper info to file: {e}")
        
        # Additional paper
        add_paper_json_path = os.path.join(json_dir, "add_paper_info.json")
        try:
            with open(add_paper_json_path, "w", encoding="utf-8") as f:
                f.write(add_method_text)
            print(f"Additional paper info saved to: {add_paper_json_path}")
        except Exception as e:
            print(f"Warning: Could not save additional paper info to file: {e}")
        
        # Complete output information
        output_json = {
            "base_github_url": base_github_url,
            "add_github_url": add_github_url,
            "base_paper_info": json.loads(base_method_text),
            "add_paper_info": json.loads(add_method_text),
            "learnings": report_data["learnings"],
            "sources": report_data["sources"],
            "report_path": report_path,
            "generated_at": report_data["generated_at"],
        }
        
        output_json_path = os.path.join(json_dir, "paper_search_result.json")
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=2, ensure_ascii=False)
            print(f"Complete search result saved to: {output_json_path}")
        except Exception as e:
            print(f"Warning: Could not save search result to file: {e}")
        
        return {
            "base_github_url": base_github_url,
            "base_method_text": base_method_text,
            "add_github_url": add_github_url,
            "add_method_text": add_method_text,
            "research_report": research_report,
        }
    
    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(WebEnhancedRetrievePaperState)
        
        # Add nodes
        graph_builder.add_node("initialize_state", self._initialize_state)
        graph_builder.add_node("web_enhanced_search", self._web_enhanced_search)
        graph_builder.add_node("select_base_paper", self._select_base_paper_node)
        graph_builder.add_node("generate_enhanced_queries", self._generate_enhanced_queries_node)
        graph_builder.add_node("web_enhanced_add_search", self._web_enhanced_add_search)
        graph_builder.add_node("select_add_paper", self._select_add_paper_node)
        graph_builder.add_node("prepare_output", self._prepare_output)
        
        # Add edges
        graph_builder.add_edge(START, "initialize_state")
        graph_builder.add_edge("initialize_state", "web_enhanced_search")
        graph_builder.add_edge("web_enhanced_search", "select_base_paper")
        graph_builder.add_edge("select_base_paper", "generate_enhanced_queries")
        graph_builder.add_edge("generate_enhanced_queries", "web_enhanced_add_search")
        graph_builder.add_edge("web_enhanced_add_search", "select_add_paper")
        graph_builder.add_edge("select_add_paper", "prepare_output")
        graph_builder.add_edge("prepare_output", END)
        
        return graph_builder.compile()


# Create a wrapped subgraph for export
WebEnhancedPaperRetriever = create_wrapped_subgraph(
    WebEnhancedPaperSubgraph,
    WebEnhancedRetrievePaperState,
    WebEnhancedRetrievePaperState
)


# Example usage
if __name__ == "__main__":
    async def main():
        save_dir = os.path.join(os.getcwd(), "data")
        llm_name = "gpt-4o"
        
        # Setup with minimal settings for testing
        breadth = 1
        depth = 1
        
        print(f"Running with minimal settings: breadth={breadth}, depth={depth}")
        
        subgraph = WebEnhancedPaperSubgraph(
            llm_name=llm_name,
            save_dir=save_dir,
            breadth=breadth,
            depth=depth,
        ).build_graph()
        
        state = {
            "queries": ["Adam optimizer improvements"],
            "web_search_enabled": True,
        }
        
        config = {"recursion_limit": 300}
        
        result = await subgraph.ainvoke(state, config=config)
        
        print("\n=== Search Complete ===")
        print(f"Base paper: {result['selected_base_paper_info'].title}")
        print(f"Base GitHub URL: {result['base_github_url']}")
        print(f"Add paper: {result['selected_add_paper_info'].title}")
        print(f"Add GitHub URL: {result['add_github_url']}")
        
        print("\nResearch Report Preview (first 500 characters):")
        report_preview = result["research_report"][:500] + "..." if len(result["research_report"]) > 500 else result["research_report"]
        print(report_preview)
        
        print(f"\nFull report saved to: {os.path.join(save_dir, 'reports', 'research_report.md')}")
    
    asyncio.run(main())
