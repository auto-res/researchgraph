import os
import json
import asyncio
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.retrieve_paper_subgraph.nodes.select_best_paper_with_context import select_best_paper_with_context
from researchgraph.retrieve_paper_subgraph.nodes.generate_enhanced_queries import generate_enhanced_queries
from researchgraph.retrieve_paper_subgraph.nodes.generate_report import generate_markdown_report
from researchgraph.retrieve_paper_subgraph.nodes.recursive_paper_search import (
    RecursivePaperSearchNode, 
    CandidatePaperInfo,
)

load_dotenv()

class RetrievePaperSubgraphInputState(TypedDict):
    queries: list[str]

class RetrievePaperSubgraphHiddenState(TypedDict):
    base_paper_learnings: list[str]
    base_paper_visited_urls: list[str]
    base_paper_candidates: list[CandidatePaperInfo]
    enhanced_queries: list[str]
    add_paper_learnings: list[str]
    add_paper_visited_urls: list[str]
    add_paper_candidates: list[CandidatePaperInfo]
    selected_base_paper_info: CandidatePaperInfo
    selected_add_paper_info: CandidatePaperInfo

class RetrievePaperSubgraphOutputState(TypedDict):
    base_github_url: str
    base_method_text: str
    add_github_url: str
    add_method_text: str
    research_report: str  # Markdown形式の研究レポート

class RetrievePaperSubgraphState(
    RetrievePaperSubgraphInputState, RetrievePaperSubgraphHiddenState, RetrievePaperSubgraphOutputState
):
    pass


class RetrievePaperSubgraph:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
        breadth: int = 3,
        depth: int = 2,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        self.breadth = breadth
        self.depth = depth
        
    async def _recursive_base_paper_search_node(self, state: RetrievePaperSubgraphState) -> dict:
        queries = state["queries"]
        result = await RecursivePaperSearchNode(
            llm_name=llm_name,
            breadth=2,
            depth=1,
            save_dir=save_dir,
        ).execute(queries)
        return {
            "base_paper_learnings": result["learnings"],
            "base_paper_visited_urls": result["visited_urls"],
            "base_paper_candidates": result["paper_candidates"],
        }
    
    async def _recursive_add_paper_search_node(self, state: RetrievePaperSubgraphState) -> dict:
        enhanced_queries = state["enhanced_queries"]
        result = await RecursivePaperSearchNode(
            llm_name=llm_name,
            breadth=2,
            depth=1,
            save_dir=save_dir,
        ).execute(enhanced_queries)
        return {
            "add_paper_learnings": result["learnings"], 
            "add_paper_visited_urls": result["visited_urls"],
            "add_paper_candidates": result["paper_candidates"],
        }

    def _select_base_paper_node(self, state: RetrievePaperSubgraphState) -> dict:
        print("select_paper_node")
        selected_paper = select_best_paper_with_context(
            llm_name=self.llm_name,
            candidate_papers=state["base_paper_candidates"],
            learnings=state["base_paper_learnings"], 
        )
        return {
            "selected_base_paper_info": selected_paper
        }
    
    def _select_add_paper_node(self, state: RetrievePaperSubgraphState) -> dict:
        print("select_paper_node")
        selected_paper = select_best_paper_with_context(
            llm_name=self.llm_name,
            candidate_papers=state["add_paper_candidates"],
            learnings=state["add_paper_learnings"], 
            base_paper=state["selected_base_paper_info"],
        )
        return {
            "selected_add_paper_info": selected_paper
        }

    def _generate_enhanced_queries_node(self, state: RetrievePaperSubgraphState) -> dict:
        print("generate_queries_node")
        generated_queries = generate_enhanced_queries(
            llm_name=self.llm_name,
            base_paper=state["selected_base_paper_info"],
            learnings=state["base_paper_learnings"],
        )
        return {
            "enhanced_queries": generated_queries,
        }

    def _save_report(self, state: RetrievePaperSubgraphState) -> dict:
        print("Generating research report...")
        base_paper = state["selected_base_paper_info"]
        add_paper = state["selected_add_paper_info"]

        base_github_url, base_method_text = base_paper.github_url, base_paper.model_dump_json()
        add_github_url, add_method_text = add_paper.github_url, add_paper.model_dump_json()

        report_data = generate_markdown_report(
            base_paper=base_paper,
            add_paper=add_paper,
            base_learnings=state["base_paper_learnings"],
            add_learnings=state["add_paper_learnings"],
            base_visited_urls=state["base_paper_visited_urls"],
            add_visited_urls=state["add_paper_visited_urls"],
            max_learnings=10,
        )
        research_report = report_data["markdown"]

        # 保存ディレクトリの設定
        dirs = {key: os.path.join(self.save_dir, key) for key in ["papers", "reports", "json"]}
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)

        report_path = os.path.join(dirs["reports"], "research_report.md")
        self._save_text_file(report_path, research_report, "Research report")

        base_paper_json_path = os.path.join(dirs["json"], "base_paper_info.json")
        self._save_text_file(base_paper_json_path, base_method_text, "Base paper info")

        add_paper_json_path = os.path.join(dirs["json"], "add_paper_info.json")
        self._save_text_file(add_paper_json_path, add_method_text, "Additional paper info")

        output_json = {
            "base_github_url": base_github_url,
            "add_github_url": add_github_url,
            "base_paper_info": json.loads(base_method_text),
            "add_paper_info": json.loads(add_method_text),
            "learnings": report_data["learnings"],
            "sources": report_data["sources"],
            "report_path": report_path,
            "generated_at": report_data["generated_at"]
        }

        output_json_path = os.path.join(dirs["json"], "paper_search_result.json")
        self._save_json(output_json_path, output_json, "Complete search result")

        return {
            "base_github_url": base_github_url,
            "base_method_text": base_method_text,
            "add_github_url": add_github_url,
            "add_method_text": add_method_text,
            "research_report": research_report,
        }
    
    def _save_text_file(self, file_path: str, content: str, description: str) -> None:
        try:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"{description} saved to: {file_path}")
        except Exception as e:
            print(f"Warning: Could not save {description} to file: {e}")
            
    def _save_json(self, file_path: str, data: str, description: str) -> None:
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"{description} saved to: {file_path}")
        except Exception as e:
            print(f"Warning: Could not save {description} to file: {e}")

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(RetrievePaperSubgraphState)

        graph_builder.add_node("recursive_base_paper_search", self._recursive_base_paper_search_node)
        graph_builder.add_node("select_base_paper", self._select_base_paper_node)
        graph_builder.add_node("generate_queries", self._generate_enhanced_queries_node)
        graph_builder.add_node("recursive_add_paper_search", self._recursive_add_paper_search_node)
        graph_builder.add_node("select_add_paper", self._select_add_paper_node)
        graph_builder.add_node("save_report", self._save_report)

        graph_builder.add_edge(START, "recursive_base_paper_search")
        graph_builder.add_edge("recursive_base_paper_search", "select_base_paper")
        graph_builder.add_edge("select_base_paper", "generate_queries")
        graph_builder.add_edge("generate_queries", "recursive_add_paper_search")
        graph_builder.add_edge("recursive_add_paper_search", "select_add_paper")
        graph_builder.add_edge("select_add_paper", "save_report")
        graph_builder.add_edge("save_report", END)

        return graph_builder.compile()


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    llm_name = "gpt-4o-mini-2024-07-18"

    breadth = 1  # 最小の幅
    depth = 1    # 最小の深さ

    print(f"Running with minimal settings: breadth={breadth}, depth={depth}")

    subgraph = RetrievePaperSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
        breadth=breadth,
        depth=depth,
    ).build_graph()

    state = {
        "queries": ["In-context learning"],
    }
    config = {"recursion_limit": 300}

    async def main():
        result = await subgraph.ainvoke(state, config=config)

    asyncio.run(main())
