from typing import Annotated, TypedDict
import operator
import asyncio
import os
from pydantic import BaseModel
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.retrieve_paper_subgraph.nodes.extract_github_url_node import (
    ExtractGithubUrlNode,
)
from researchgraph.retrieve_paper_subgraph.nodes.retrieve_arxiv_text_node import (
    RetrievearXivTextNode,
)
from researchgraph.retrieve_paper_subgraph_new.nodes.recursive_paper_search import (
    recursive_paper_search,
    CandidatePaperInfo,
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


class RetrievePaperNewState(TypedDict):
    # 入力
    queries: list[str]
    save_dir: str

    # 内部状態
    base_paper_learnings: list[str]
    base_paper_visited_urls: list[str]
    base_paper_candidates: list[CandidatePaperInfo]

    enhanced_queries: list[str]

    add_paper_learnings: list[str]
    add_paper_visited_urls: list[str]
    add_paper_candidates: list[CandidatePaperInfo]

    # 選択された論文
    selected_base_paper_info: CandidatePaperInfo
    selected_add_paper_info: CandidatePaperInfo

    # 出力
    base_github_url: str
    base_method_text: str
    add_github_url: str
    add_method_text: str
    research_report: str  # Markdown形式の研究レポート


class RetrievePaperSubgraphNew:
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

    def _initialize_state(self, state: RetrievePaperNewState) -> dict:
        print("---RetrievePaperSubgraphNew---")
        return {
            "queries": state["queries"],
            "save_dir": state.get("save_dir", self.save_dir),
            "base_paper_learnings": [],
            "base_paper_visited_urls": [],
            "base_paper_candidates": [],
            "add_paper_learnings": [],
            "add_paper_visited_urls": [],
            "add_paper_candidates": [],
        }

    async def _recursive_base_paper_search(self, state: RetrievePaperNewState) -> dict:
        print("recursive_base_paper_search")
        research_result = await recursive_paper_search(
            llm_name=self.llm_name,
            queries=state["queries"],
            breadth=self.breadth,
            depth=self.depth,
            save_dir=state["save_dir"],
        )

        return {
            "base_paper_learnings": research_result["learnings"],
            "base_paper_visited_urls": research_result["visited_urls"],
            "base_paper_candidates": research_result["paper_candidates"],
        }

    def _select_base_paper_node(self, state: RetrievePaperNewState) -> dict:
        print("select_base_paper_node")
        selected_paper = select_best_paper_with_context(
            llm_name=self.llm_name,
            candidate_papers=state["base_paper_candidates"],
            learnings=state["base_paper_learnings"],
        )

        return {
            "selected_base_paper_info": selected_paper,
        }

    def _generate_enhanced_queries_node(self, state: RetrievePaperNewState) -> dict:
        print("generate_enhanced_queries_node")
        generated_queries = generate_enhanced_queries(
            llm_name=self.llm_name,
            base_paper=state["selected_base_paper_info"],
            learnings=state["base_paper_learnings"],
        )

        return {
            "enhanced_queries": generated_queries,
        }

    async def _recursive_add_paper_search(self, state: RetrievePaperNewState) -> dict:
        print("recursive_add_paper_search")
        research_result = await recursive_paper_search(
            llm_name=self.llm_name,
            queries=state["enhanced_queries"],
            breadth=self.breadth,
            depth=self.depth,
            previous_learnings=state["base_paper_learnings"],
            save_dir=state["save_dir"],
        )

        return {
            "add_paper_learnings": research_result["learnings"],
            "add_paper_visited_urls": research_result["visited_urls"],
            "add_paper_candidates": research_result["paper_candidates"],
        }

    def _select_add_paper_node(self, state: RetrievePaperNewState) -> dict:
        print("select_add_paper_node")

        # 候補論文がない場合のチェック
        if not state["add_paper_candidates"]:
            print("Warning: No additional paper candidates found. Using default paper.")

        selected_paper = select_best_paper_with_context(
            llm_name=self.llm_name,
            candidate_papers=state["add_paper_candidates"],
            learnings=state["add_paper_learnings"],
            base_paper=state["selected_base_paper_info"],
        )

        return {
            "selected_add_paper_info": selected_paper,
        }

    def _prepare_output(self, state: RetrievePaperNewState) -> dict:
        print("prepare_output")
        base_github_url = state["selected_base_paper_info"].github_url
        base_method_text = state["selected_base_paper_info"].model_dump_json()
        add_github_url = state["selected_add_paper_info"].github_url
        add_method_text = state["selected_add_paper_info"].model_dump_json()

        # Markdownレポートの生成
        print("Generating research report...")
        report_data = generate_markdown_report(
            base_paper=state["selected_base_paper_info"],
            add_paper=state["selected_add_paper_info"],
            base_learnings=state["base_paper_learnings"],
            add_learnings=state["add_paper_learnings"],
            base_visited_urls=state["base_paper_visited_urls"],
            add_visited_urls=state["add_paper_visited_urls"],
            max_learnings=10,  # 学習内容を10項目に制限
        )

        # Markdownレポートを取得
        research_report = report_data["markdown"]

        # 保存用のディレクトリ構造を作成
        papers_dir = os.path.join(state["save_dir"], "papers")
        reports_dir = os.path.join(state["save_dir"], "reports")
        json_dir = os.path.join(state["save_dir"], "json")

        # 各ディレクトリが存在することを確認
        os.makedirs(papers_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        # レポートをファイルに保存
        report_path = os.path.join(reports_dir, "research_report.md")
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(research_report)
            print(f"Research report saved to: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save research report to file: {e}")

        # 論文情報をJSONファイルとして保存
        import json

        # ベース論文の情報
        base_paper_json_path = os.path.join(json_dir, "base_paper_info.json")
        try:
            with open(base_paper_json_path, "w", encoding="utf-8") as f:
                f.write(base_method_text)
            print(f"Base paper info saved to: {base_paper_json_path}")
        except Exception as e:
            print(f"Warning: Could not save base paper info to file: {e}")

        # 追加論文の情報
        add_paper_json_path = os.path.join(json_dir, "add_paper_info.json")
        try:
            with open(add_paper_json_path, "w", encoding="utf-8") as f:
                f.write(add_method_text)
            print(f"Additional paper info saved to: {add_paper_json_path}")
        except Exception as e:
            print(f"Warning: Could not save additional paper info to file: {e}")

        # 出力情報をまとめたJSONファイル
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
        graph_builder = StateGraph(RetrievePaperNewState)

        # ノードの追加
        graph_builder.add_node("initialize_state", self._initialize_state)
        graph_builder.add_node("recursive_base_paper_search", self._recursive_base_paper_search)
        graph_builder.add_node("select_base_paper", self._select_base_paper_node)
        graph_builder.add_node("generate_enhanced_queries", self._generate_enhanced_queries_node)
        graph_builder.add_node("recursive_add_paper_search", self._recursive_add_paper_search)
        graph_builder.add_node("select_add_paper", self._select_add_paper_node)
        graph_builder.add_node("prepare_output", self._prepare_output)

        # エッジの追加
        graph_builder.add_edge(START, "initialize_state")
        graph_builder.add_edge("initialize_state", "recursive_base_paper_search")
        graph_builder.add_edge("recursive_base_paper_search", "select_base_paper")
        graph_builder.add_edge("select_base_paper", "generate_enhanced_queries")
        graph_builder.add_edge("generate_enhanced_queries", "recursive_add_paper_search")
        graph_builder.add_edge("recursive_add_paper_search", "select_add_paper")
        graph_builder.add_edge("select_add_paper", "prepare_output")
        graph_builder.add_edge("prepare_output", END)

        return graph_builder.compile()


if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data")
    llm_name = "gpt-4o-mini-2024-07-18"

    # 最小の幅と深さで実行（テスト用）
    breadth = 1  # 最小の幅
    depth = 1    # 最小の深さ

    print(f"Running with minimal settings: breadth={breadth}, depth={depth}")

    subgraph = RetrievePaperSubgraphNew(
        llm_name=llm_name,
        save_dir=save_dir,
        breadth=breadth,
        depth=depth,
    ).build_graph()

    state = {
        "queries": ["deep learning"],  # シンプルなクエリで実行
    }
    config = {"recursion_limit": 300}

    async def main():
        result = await subgraph.ainvoke(state, config=config)

        print(result.keys())

        print("base_paper_candidates")
        base_paper_candidates = result["base_paper_candidates"]
        print(len(base_paper_candidates))

        print("add_paper_candidates")
        add_paper_candidates = result["add_paper_candidates"]
        print(len(add_paper_candidates))

        print("base_github_url")
        base_github_url = result["base_github_url"]
        print(base_github_url)

        print("add_github_url")
        add_github_url = result["add_github_url"]
        print(add_github_url)

        print("\nResearch Report Preview (first 500 characters):")
        report_preview = result["research_report"][:500] + "..." if len(result["research_report"]) > 500 else result["research_report"]
        print(report_preview)

        print(f"\nFull report saved to: {os.path.join(save_dir, 'reports', 'research_report.md')}")
        print(f"JSON files saved to: {os.path.join(save_dir, 'json')}")

    asyncio.run(main())
