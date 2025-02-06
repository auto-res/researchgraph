from typing import Annotated
import operator
from typing_extensions import TypedDict
from pydantic import BaseModel

from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.retrieve_paper_subgraph.nodes.extract_github_url_node import (
    ExtractGithubUrlNode,
)
from researchgraph.retrieve_paper_subgraph.nodes.generate_queries_node import (
    generate_queries_node,
    generate_queries_prompt_add,
)
from researchgraph.retrieve_paper_subgraph.nodes.search_papers_node import (
    SearchPapersNode,
)
from researchgraph.retrieve_paper_subgraph.nodes.select_best_paper_node import (
    select_best_paper_node,
    select_base_paper_prompt,
    select_add_paper_prompt,
)
from researchgraph.retrieve_paper_subgraph.nodes.summarize_paper_node import (
    summarize_paper_node,
    summarize_paper_prompt_base,
    # summarize_paper_prompt_add,
)
from researchgraph.retrieve_paper_subgraph.nodes.retrieve_arxiv_text_node import (
    RetrievearXivTextNode,
)


class CandidatePaperInfo(BaseModel):
    arxiv_id: str
    arxiv_url: str
    title: str
    authors: list[str]
    published_date: str
    journal: str
    doi: str
    summary: str
    # 途中で取得
    github_url: str
    # 最後のサマリーで取得
    main_contributions: str
    methodology: str
    experimental_setup: str
    limitations: str
    future_research_directions: str


class RetrievePaperState(TypedDict):
    queries: list

    tmp_search_paper_list: list[dict]
    tmp_search_paper_count: int
    tmp_paper_full_text: str
    tmp_github_url: str
    process_index: int

    candidate_base_papers_info_list: Annotated[list[CandidatePaperInfo], operator.add]
    selected_base_paper_arxiv_id: str
    selected_base_paper_info: CandidatePaperInfo

    generate_queries: list[str]

    candidate_add_papers_info_list: Annotated[list[CandidatePaperInfo], operator.add]
    selected_add_paper_arxiv_id: str
    selected_add_paper_info: CandidatePaperInfo

    base_github_url: str
    base_method_text: str
    add_github_url: str
    add_method_text: str


class RetrievePaperSubgraph:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir

    def _initialize_state(self, state: RetrievePaperState) -> dict:
        print("---RetrievePaperSubgraph---")
        return {
            "queries": state["queries"],
            "process_index": 0,
            "candidate_base_papers_info_list": [],
            "candidate_add_papers_info_list": [],
        }

    # Base Paper
    def _search_base_papers_node(self, state: RetrievePaperState) -> dict:
        print("search_papers_node")
        queries = state["queries"]
        search_paper_list = SearchPapersNode().execute(queries=queries)
        print("search_paper_count: ", len(search_paper_list))
        return {
            "tmp_search_paper_list": search_paper_list,
            "tmp_search_paper_count": len(search_paper_list),
        }

    def _retrieve_arxiv_full_text_node(self, state: RetrievePaperState) -> dict:
        print("retrieve_arxiv_full_text_node")
        process_index = state["process_index"]
        print("process_index: ", process_index)
        paper_info = state["tmp_search_paper_list"][process_index]
        arxiv_url = paper_info["arxiv_url"]
        paper_full_text = RetrievearXivTextNode(
            save_dir=self.save_dir,
        ).execute(arxiv_url=arxiv_url)
        return {"tmp_paper_full_text": paper_full_text}

    def _extract_github_url_node(self, state: RetrievePaperState) -> dict:
        print("extract_github_url_node")
        paper_full_text = state["tmp_paper_full_text"]
        process_index = state["process_index"]
        paper_summary = state["tmp_search_paper_list"][process_index]["summary"]
        github_url = ExtractGithubUrlNode(
            llm_name=self.llm_name,
        ).execute(
            paper_full_text=paper_full_text,
            paper_summary=paper_summary,
        )
        # GitHub URLが取得できなかった場合は次の論文を処理するためにProcess Indexを進める
        process_index = state["process_index"]
        if github_url == "":
            process_index += 1
        return {"tmp_github_url": github_url, "process_index": process_index}

    def _check_github_urls(self, state: RetrievePaperState) -> str:
        print("check_github_urls")
        if state["tmp_github_url"] == "":
            if state["process_index"] < state["tmp_search_paper_count"]:
                return "Next paper"
            else:
                return "All complete"
        else:
            return "Generate paper summary"

    def _summarize_base_paper_node(self, state: RetrievePaperState) -> dict:
        print("summarize_paper_node")
        paper_full_text = state["tmp_paper_full_text"]
        (
            main_contributions,
            methodology,
            experimental_setup,
            limitations,
            future_research_directions,
        ) = summarize_paper_node(
            llm_name=self.llm_name,
            prompt_template=summarize_paper_prompt_base,
            paper_text=paper_full_text,
        )

        process_index = state["process_index"]
        paper_info = state["tmp_search_paper_list"][process_index]
        candidate_papers_info = {
            "arxiv_id": paper_info["arxiv_id"],
            "arxiv_url": paper_info["arxiv_url"],
            "title": paper_info.get("title", ""),
            "authors": paper_info.get("authors", ""),
            "published_date": paper_info.get("published_date", ""),
            "journal": paper_info.get("journal", ""),
            "doi": paper_info.get("doi", ""),
            "summary": paper_info.get("summary", ""),
            "github_url": state["tmp_github_url"],
            "main_contributions": main_contributions,
            "methodology": methodology,
            "experimental_setup": experimental_setup,
            "limitations": limitations,
            "future_research_directions": future_research_directions,
        }
        return {
            "process_index": process_index + 1,
            "candidate_base_papers_info_list": [
                CandidatePaperInfo(**candidate_papers_info)
            ],
        }

    def _check_paper_count(self, state: RetrievePaperState) -> str:
        print("check_paper_count")
        if state["process_index"] < state["tmp_search_paper_count"]:
            return "Next paper"
        else:
            return "All complete"

    def _base_select_best_paper_node(self, state: RetrievePaperState) -> dict:
        print("base_select_best_paper_node")
        candidate_papers_info_list = state["candidate_base_papers_info_list"]
        # TODO:論文の検索数の制御がうまくいっていない気がする
        selected_arxiv_id = select_best_paper_node(
            llm_name=self.llm_name,
            prompt_template=select_base_paper_prompt,
            candidate_papers=candidate_papers_info_list,
        )

        # 選択された論文の情報を取得
        for paper_info in candidate_papers_info_list:
            if paper_info.arxiv_id == selected_arxiv_id:
                selected_paper_info = paper_info

        return {
            "selected_base_paper_arxiv_id": selected_arxiv_id,
            "selected_base_paper_info": selected_paper_info,
        }

    # add paper
    def _generate_queries_node(self, state: RetrievePaperState) -> dict:
        print("generate_queries_node")
        selected_base_paper_info = state["selected_base_paper_info"]
        generated_queries_list = generate_queries_node(
            llm_name=self.llm_name,
            prompt_template=generate_queries_prompt_add,
            selected_base_paper_info=selected_base_paper_info,
        )
        return {
            "generate_queries": generated_queries_list,
            "process_index": 0,
        }

    def _search_add_papers_node(self, state: RetrievePaperState) -> dict:
        print("add_search_papers_node")
        queries = state["generate_queries"]
        search_paper_list = SearchPapersNode().execute(queries=queries)
        # NOTE:検索論文の数が多くなりすぎることがあるためTOP_Nで制限
        TOP_N = 15
        search_paper_list = search_paper_list[:TOP_N]
        print("add_search_paper_count: ", len(search_paper_list))
        return {
            "tmp_search_paper_list": search_paper_list,
            "tmp_search_paper_count": len(search_paper_list),
        }

    def _summarize_add_paper_node(self, state: RetrievePaperState) -> dict:
        print("summarize_add_paper_node")
        paper_full_text = state["tmp_paper_full_text"]
        (
            main_contributions,
            methodology,
            experimental_setup,
            limitations,
            future_research_directions,
        ) = summarize_paper_node(
            llm_name=self.llm_name,
            prompt_template=summarize_paper_prompt_base,
            paper_text=paper_full_text,
        )

        process_index = state["process_index"]
        paper_info = state["tmp_search_paper_list"][process_index]
        candidate_papers_info = {
            "arxiv_id": paper_info["arxiv_id"],
            "arxiv_url": paper_info["arxiv_url"],
            "title": paper_info.get("title", ""),
            "authors": paper_info.get("authors", ""),
            "published_date": paper_info.get("published_date", ""),
            "journal": paper_info.get("journal", ""),
            "doi": paper_info.get("doi", ""),
            "summary": paper_info.get("summary", ""),
            "github_url": state["tmp_github_url"],
            "main_contributions": main_contributions,
            "methodology": methodology,
            "experimental_setup": experimental_setup,
            "limitations": limitations,
            "future_research_directions": future_research_directions,
        }
        return {
            "process_index": process_index + 1,
            "candidate_add_papers_info_list": [
                CandidatePaperInfo(**candidate_papers_info)
            ],
        }

    def _add_select_best_paper_node(self, state: RetrievePaperState) -> dict:
        print("add_select_best_paper_node")
        candidate_papers_info_list = state["candidate_add_papers_info_list"]
        selected_arxiv_id = select_best_paper_node(
            llm_name=self.llm_name,
            prompt_template=select_add_paper_prompt,
            candidate_papers=candidate_papers_info_list,
            selected_base_paper_info=state["selected_base_paper_info"],
        )

        # 選択された論文の情報を取得
        for paper_info in candidate_papers_info_list:
            if paper_info.arxiv_id == selected_arxiv_id:
                selected_paper_info = paper_info

        return {
            "selected_add_paper_arxiv_id": selected_arxiv_id,
            "selected_add_paper_info": selected_paper_info,
        }

    def _prepare_state(self, state: RetrievePaperState) -> dict:
        base_github_url = state["selected_base_paper_info"].github_url
        base_method_text = state["selected_base_paper_info"].model_dump_json()
        add_github_url = state["selected_add_paper_info"].github_url
        add_method_text = state["selected_add_paper_info"].model_dump_json()

        return {
            "base_github_url": base_github_url,
            "base_method_text": base_method_text,
            "add_github_url": add_github_url,
            "add_method_text": add_method_text,
        }

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(RetrievePaperState)

        # make nodes
        graph_builder.add_node("initialize_state", self._initialize_state)

        # base paper
        graph_builder.add_node(
            "base_search_papers_node", self._search_base_papers_node
        )  # TODO: 検索結果が空ならEND
        graph_builder.add_node(
            "base_retrieve_arxiv_full_text_node",
            self._retrieve_arxiv_full_text_node,
        )
        graph_builder.add_node(
            "base_extract_github_urls_node", self._extract_github_url_node
        )
        graph_builder.add_node(
            "base_summarize_paper_node", self._summarize_base_paper_node
        )
        graph_builder.add_node(
            "base_select_best_paper_node", self._base_select_best_paper_node
        )

        # add paper
        graph_builder.add_node("generate_queries_node", self._generate_queries_node)
        graph_builder.add_node(
            "add_search_papers_node", self._search_add_papers_node
        )  # TODO: 検索結果が空ならEND
        graph_builder.add_node(
            "add_retrieve_arxiv_full_text_node", self._retrieve_arxiv_full_text_node
        )
        graph_builder.add_node(
            "add_extract_github_urls_node", self._extract_github_url_node
        )
        graph_builder.add_node(
            "add_summarize_paper_node", self._summarize_add_paper_node
        )
        graph_builder.add_node(
            "add_select_best_paper_node", self._add_select_best_paper_node
        )
        graph_builder.add_node("prepare_state", self._prepare_state)

        # make edges
        # base paper
        graph_builder.add_edge(START, "initialize_state")
        graph_builder.add_edge("initialize_state", "base_search_papers_node")
        graph_builder.add_edge(
            "base_search_papers_node", "base_retrieve_arxiv_full_text_node"
        )
        graph_builder.add_edge(
            "base_retrieve_arxiv_full_text_node", "base_extract_github_urls_node"
        )
        graph_builder.add_conditional_edges(
            "base_extract_github_urls_node",
            path=self._check_github_urls,
            path_map={
                "Next paper": "base_retrieve_arxiv_full_text_node",
                "Generate paper summary": "base_summarize_paper_node",
                "All complete": "base_select_best_paper_node",
            },
        )
        graph_builder.add_conditional_edges(
            "base_summarize_paper_node",
            path=self._check_paper_count,
            path_map={
                "Next paper": "base_retrieve_arxiv_full_text_node",
                "All complete": "base_select_best_paper_node",
            },
        )

        # add paper
        graph_builder.add_edge("base_select_best_paper_node", "generate_queries_node")
        graph_builder.add_edge("generate_queries_node", "add_search_papers_node")
        graph_builder.add_edge(
            "add_search_papers_node", "add_retrieve_arxiv_full_text_node"
        )
        graph_builder.add_edge(
            "add_retrieve_arxiv_full_text_node", "add_extract_github_urls_node"
        )
        graph_builder.add_conditional_edges(
            "add_extract_github_urls_node",
            path=self._check_github_urls,
            path_map={
                "Next paper": "add_retrieve_arxiv_full_text_node",
                "Generate paper summary": "add_summarize_paper_node",
                "All complete": "add_select_best_paper_node",
            },
        )
        graph_builder.add_conditional_edges(
            "add_summarize_paper_node",
            path=self._check_paper_count,
            path_map={
                "Next paper": "add_retrieve_arxiv_full_text_node",
                "All complete": "add_select_best_paper_node",
            },
        )
        graph_builder.add_edge("add_select_best_paper_node", "prepare_state")
        graph_builder.add_edge("prepare_state", END)

        return graph_builder.compile()


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    # llm_name = "gpt-4o-2024-11-20"
    llm_name = "gpt-4o-mini-2024-07-18"

    subgraph = RetrievePaperSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
    ).build_graph()

    state = {
        "queries": ["deep learning"],
    }
    config = {"recursion_limit": 300}
    result = subgraph.invoke(state, config=config)

    print(result.keys())

    # 複数になるようにしないといけない
    print("candidate_base_papers_info")
    candidate_base_papers_info = result["candidate_base_papers_info_list"]
    # print(candidate_base_papers_info)
    print(len(candidate_base_papers_info))

    candidate_add_papers_info = result["candidate_add_papers_info_list"]
    # print(candidate_add_papers_info)
    print(len(candidate_add_papers_info))

    base_github_url = result["base_github_url"]
    print(base_github_url)
    base_method_text = result["base_method_text"]
    print(base_method_text)
    add_github_url = result["add_github_url"]
    print(add_github_url)
    add_method_text = result["add_method_text"]
