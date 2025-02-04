from typing import Annotated
import operator
from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.retrieve_paper_subgraph.nodes.extract_github_urls_node import (
    ExtractGithubUrlsNode,
)
from researchgraph.retrieve_paper_subgraph.nodes.generate_queries_node import (
    generate_queries_node,
)
from researchgraph.retrieve_paper_subgraph.nodes.search_papers_node import (
    SearchPapersNode,
)
from researchgraph.retrieve_paper_subgraph.nodes.select_best_paper_node import (
    select_best_paper_node,
)
from researchgraph.retrieve_paper_subgraph.nodes.summarize_paper_node import (
    summarize_paper_node,
)
from researchgraph.retrieve_paper_subgraph.nodes.retrieve_arxiv_text_node import (
    RetrievearXivTextNode,
)

from researchgraph.retrieve_paper_subgraph.prompt import (
    ai_integrator_v3_summarize_paper_prompt_add,
    ai_integrator_v3_generate_queries_prompt_add,
    ai_integrator_v3_select_paper_prompt_add,
    ai_integrator_v3_summarize_paper_prompt_base,
    ai_integrator_v3_select_paper_prompt_base,
)


# class BasePaperState(TypedDict):
#     base_search_paper_list: list[dict]
#     base_search_paper_count: int
#     base_paper_full_text: str
#     base_candidate_papers_info: Annotated[list[dict], operator.add]
#     base_github_urls: list[str]
#     base_paper_full_text: str
#     base_process_index: int
#     base_selected_arxiv_id: str


#     base_base_selected_paper: dict

# arxiv_url: str
# paper_text: str
# github_urls: list
# candidate_papers: list[dict]
# selected_arxiv_id: str
# selected_paper: dict


# class AddPaperState(TypedDict):
#     #queries: list
#     base_selected_paper: dict
#     search_results: list[dict]
#     arxiv_url: str
#     paper_text: str
#     github_urls: list
#     candidate_papers: list[dict]
#     selected_arxiv_id: str
#     selected_paper: dict
#     process_index: int = 0
#     # technical_summary: Optional[dict] = None
#     generated_query_1: str   # NOTE: structuredoutput_llmnodeの出力がlistに対応したら"queries"にまとめます
#     generated_query_2: str
#     generated_query_3: str
#     generated_query_4: str
#     generated_query_5: str


class RetrievePaperState(TypedDict):
    queries: list

    base_search_paper_list: list[dict]
    base_search_paper_count: int
    base_paper_full_text: str
    base_candidate_papers_info: Annotated[list[dict], operator.add]
    # base_candidate_papers_info: list[dict] = []
    base_github_urls: list[str]
    base_paper_full_text: str
    base_process_index: int
    base_selected_arxiv_id: str
    base_selected_paper: dict

    generate_queries: list[str]

    add_search_paper_list: list[dict]
    add_search_paper_count: int
    add_paper_full_text: str
    add_candidate_papers_info: Annotated[list[dict], operator.add]
    add_github_urls: list[str]
    add_paper_full_text: str
    add_process_index: int
    add_selected_arxiv_id: str
    add_selected_paper: dict


class RetrievePaperSubgraph:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir

    def _initialize_state(self, state: RetrievePaperState) -> dict:
        return {
            "queries": state["queries"],
            "base_process_index": 0,
            "add_process_index": 0,
        }

    # Base Paper
    def _base_search_papers_node(self, state: RetrievePaperState) -> dict:
        print("base_search_papers_node")
        queries = state["queries"]
        search_paper_list = SearchPapersNode().execute(queries=queries)
        print("base_search_paper_count: ", len(search_paper_list))
        return {
            "base_search_paper_list": search_paper_list,
            "base_search_paper_count": len(search_paper_list),
        }

    def _base_retrieve_arxiv_full_text_node(self, state: RetrievePaperState) -> dict:
        print("base_retrieve_arxiv_full_text_node")
        process_index = state["base_process_index"]
        print("base_process_index: ", process_index)
        paper_info = state["base_search_paper_list"][process_index]
        arxiv_url = paper_info["arxiv_url"]
        paper_full_text = RetrievearXivTextNode(
            save_dir=self.save_dir,
        ).execute(arxiv_url=arxiv_url)
        return {"base_paper_full_text": paper_full_text}

    def _base_extract_github_urls_node(self, state: RetrievePaperState) -> dict:
        print("base_extract_github_urls_node")
        paper_full_text = state["base_paper_full_text"]
        github_urls = ExtractGithubUrlsNode().execute(paper_text=paper_full_text)
        process_index = state["base_process_index"]
        if not github_urls:
            process_index += 1
        return {"base_github_urls": github_urls, "base_process_index": process_index}

    def _base_check_github_urls(self, state: RetrievePaperState) -> str:
        print("base_check_github_urls")
        if not state["base_github_urls"]:
            if state["base_process_index"] < state["base_search_paper_count"]:
                return "次の論文の処理を開始"
            else:
                return "全ての論文の処理が完了"
        else:
            return "論文のサマリーを生成"

    def _base_summarize_paper_node(self, state: RetrievePaperState) -> dict:
        print("base_summarize_paper_node")
        paper_full_text = state["base_paper_full_text"]
        (
            main_contributions,
            methodology,
            experimental_setup,
            limitations,
            future_research_directions,
        ) = summarize_paper_node(
            llm_name=self.llm_name,
            prompt_template=ai_integrator_v3_summarize_paper_prompt_base,
            paper_text=paper_full_text,
        )

        process_index = state["base_process_index"]
        paper_info = state["base_search_paper_list"][process_index]
        # TODO: ここでgithub_urlsのリスト番号を指定しているが、複数のgithub_urlsがある場合はどうするか
        GITHUB_URLS_LIST_NUMBERS = 0
        candidate_papers_info = {
            "arxiv_id": paper_info["arxiv_id"],
            "arxiv_url": paper_info["arxiv_url"],
            "title": paper_info["title"],
            "authors": paper_info.get("authors", ""),
            "published_date": paper_info.get("published_date", ""),
            "journal": paper_info.get("journal", ""),
            "doi": paper_info.get("doi", ""),
            "github_urls": state["base_github_urls"][GITHUB_URLS_LIST_NUMBERS],
            "main_contributions": main_contributions,
            "methodology": methodology,
            "experimental_setup": experimental_setup,
            "limitations": limitations,
            "future_research_directions": future_research_directions,
        }

        process_index += 1
        return {
            "base_process_index": process_index,
            "base_candidate_papers_info": [
                candidate_papers_info
            ],  # state["base_candidate_papers_info"].append(candidate_papers_info)
        }

    def _base_check_paper_count(self, state: RetrievePaperState) -> str:
        print("base_check_paper_count")
        if state["base_process_index"] < state["base_search_paper_count"]:
            return "次の論文の処理を開始"
        else:
            return "全ての論文の処理が完了"

    def _base_select_best_paper_node(self, state: RetrievePaperState) -> dict:
        print("base_select_best_paper_node")
        candidate_papers_info = state["base_candidate_papers_info"]
        selected_arxiv_id = select_best_paper_node(
            llm_name=self.llm_name,
            prompt_template=ai_integrator_v3_select_paper_prompt_base,
            candidate_papers=candidate_papers_info,
        )

        # 選択された論文の情報を取得
        for paper in candidate_papers_info:
            if paper.get("arxiv_id") == selected_arxiv_id:
                selected_paper = paper

        return {
            "base_selected_arxiv_id": selected_arxiv_id,
            "base_selected_paper": selected_paper,
        }

    # add paper
    def _generate_queries_node(self, state: RetrievePaperState) -> dict:
        print("generate_queries_node")
        base_selected_paper = state["base_selected_paper"]
        (
            generated_query_1,
            generated_query_2,
            generated_query_3,
            generated_query_4,
            generated_query_5,
        ) = generate_queries_node(
            llm_name=self.llm_name,
            prompt_template=ai_integrator_v3_generate_queries_prompt_add,
            base_selected_paper=base_selected_paper,
        )
        return {
            "generate_queries": [
                generated_query_1,
                generated_query_2,
                generated_query_3,
                generated_query_4,
                generated_query_5,
            ]
        }

    def _add_search_papers_node(self, state: RetrievePaperState) -> dict:
        print("add_search_papers_node")
        queries = state["generate_queries"]
        search_paper_list = SearchPapersNode().execute(queries=queries)
        # NOTE:検索論文の数が多くなりすぎることがあるためTOP_Nで制限
        TOP_N = 5
        search_paper_list = search_paper_list[:TOP_N]
        print("add_search_paper_count: ", len(search_paper_list))
        return {
            "add_search_paper_list": search_paper_list,
            "add_search_paper_count": len(search_paper_list),
        }

    def _add_retrieve_arxiv_full_text_node(self, state: RetrievePaperState) -> dict:
        print("add_retrieve_arxiv_full_text_node")
        process_index = state["add_process_index"]
        print("add_process_index: ", process_index)
        paper_info = state["add_search_paper_list"][process_index]
        arxiv_url = paper_info["arxiv_url"]
        paper_full_text = RetrievearXivTextNode(
            save_dir=self.save_dir,
        ).execute(arxiv_url=arxiv_url)
        return {"add_paper_full_text": paper_full_text}

    def _add_extract_github_urls_node(self, state: RetrievePaperState) -> dict:
        print("add_extract_github_urls_node")
        paper_full_text = state["add_paper_full_text"]
        github_urls = ExtractGithubUrlsNode().execute(paper_text=paper_full_text)
        process_index = state["add_process_index"]
        if not github_urls:
            process_index += 1
        return {"add_github_urls": github_urls, "add_process_index": process_index}

    def _add_check_github_urls(self, state: RetrievePaperState) -> str:
        print("add_check_github_urls")
        if not state["add_github_urls"]:
            if state["add_process_index"] < state["add_search_paper_count"]:
                return "次の論文の処理を開始"
            else:
                return "全ての論文の処理が完了"
        else:
            return "論文のサマリーを生成"

    def _add_summarize_paper_node(self, state: RetrievePaperState) -> dict:
        print("add_summarize_paper_node")
        paper_full_text = state["add_paper_full_text"]
        (
            main_contributions,
            methodology,
            experimental_setup,
            limitations,
            future_research_directions,
        ) = summarize_paper_node(
            llm_name=self.llm_name,
            prompt_template=ai_integrator_v3_summarize_paper_prompt_add,
            paper_text=paper_full_text,
        )

        process_index = state["add_process_index"]
        paper_info = state["add_search_paper_list"][process_index]
        # TODO: ここでgithub_urlsのリスト番号を指定しているが、複数のgithub_urlsがある場合はどうするか
        GITHUB_URLS_LIST_NUMBERS = 0
        candidate_papers_info = {
            "arxiv_id": paper_info["arxiv_id"],
            "arxiv_url": paper_info["arxiv_url"],
            "title": paper_info["title"],
            "authors": paper_info.get("authors", ""),
            "published_date": paper_info.get("published_date", ""),
            "journal": paper_info.get("journal", ""),
            "doi": paper_info.get("doi", ""),
            "github_urls": state["add_github_urls"][GITHUB_URLS_LIST_NUMBERS],
            "main_contributions": main_contributions,
            "methodology": methodology,
            "experimental_setup": experimental_setup,
            "limitations": limitations,
            "future_research_directions": future_research_directions,
        }

        process_index += 1
        return {
            "add_process_index": process_index,
            "add_candidate_papers_info": [
                candidate_papers_info
            ],  # state["add_candidate_papers_info"].append(candidate_papers_info)
        }

    def _add_check_paper_count(self, state: RetrievePaperState) -> str:
        print("add_check_paper_count")
        if state["add_process_index"] < state["add_search_paper_count"]:
            return "次の論文の処理を開始"
        else:
            return "全ての論文の処理が完了"

    def _add_select_best_paper_node(self, state: RetrievePaperState) -> dict:
        print("add_select_best_paper_node")
        candidate_papers_info = state["add_candidate_papers_info"]
        base_selected_paper = state["base_selected_paper"]
        selected_arxiv_id = select_best_paper_node(
            llm_name=self.llm_name,
            prompt_template=ai_integrator_v3_select_paper_prompt_add,
            candidate_papers=candidate_papers_info,
            base_selected_paper=base_selected_paper,
        )

        # 選択された論文の情報を取得
        for paper in candidate_papers_info:
            if paper.get("arxiv_id") == selected_arxiv_id:
                selected_paper = paper

        return {"add_selected_arxiv_id": selected_arxiv_id, "add_selected_paper": paper}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(RetrievePaperState)

        # make nodes
        graph_builder.add_node("initialize_state", self._initialize_state)

        # base paper
        graph_builder.add_node(
            "base_search_papers_node", self._base_search_papers_node
        )  # TODO: 検索結果が空ならEND
        graph_builder.add_node(
            "base_retrieve_arxiv_full_text_node",
            self._base_retrieve_arxiv_full_text_node,
        )
        graph_builder.add_node(
            "base_extract_github_urls_node", self._base_extract_github_urls_node
        )
        graph_builder.add_node(
            "base_summarize_paper_node", self._base_summarize_paper_node
        )
        graph_builder.add_node(
            "base_select_best_paper_node", self._base_select_best_paper_node
        )

        # add paper
        graph_builder.add_node("generate_queries_node", self._generate_queries_node)
        graph_builder.add_node(
            "add_search_papers_node", self._add_search_papers_node
        )  # TODO: 検索結果が空ならEND
        graph_builder.add_node(
            "add_retrieve_arxiv_full_text_node", self._add_retrieve_arxiv_full_text_node
        )
        graph_builder.add_node(
            "add_extract_github_urls_node", self._add_extract_github_urls_node
        )
        graph_builder.add_node(
            "add_summarize_paper_node", self._add_summarize_paper_node
        )
        graph_builder.add_node(
            "add_select_best_paper_node", self._add_select_best_paper_node
        )

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
            path=self._base_check_github_urls,
            path_map={
                "次の論文の処理を開始": "base_retrieve_arxiv_full_text_node",
                "論文のサマリーを生成": "base_summarize_paper_node",
                "全ての論文の処理が完了": "base_select_best_paper_node",
            },
        )
        graph_builder.add_conditional_edges(
            "base_summarize_paper_node",
            path=self._base_check_paper_count,
            path_map={
                "次の論文の処理を開始": "base_retrieve_arxiv_full_text_node",
                "全ての論文の処理が完了": "base_select_best_paper_node",
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
            path=self._add_check_github_urls,
            path_map={
                "次の論文の処理を開始": "add_retrieve_arxiv_full_text_node",
                "論文のサマリーを生成": "add_summarize_paper_node",
                "全ての論文の処理が完了": "add_select_best_paper_node",
            },
        )
        graph_builder.add_conditional_edges(
            "add_summarize_paper_node",
            path=self._add_check_paper_count,
            path_map={
                "次の論文の処理を開始": "add_retrieve_arxiv_full_text_node",
                "全ての論文の処理が完了": "add_select_best_paper_node",
            },
        )
        graph_builder.add_edge("add_select_best_paper_node", END)

        return graph_builder.compile()

    # def __call__(self):
    #     return self.build_graph()


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    # llm_name = "gpt-4o-2024-11-20"
    llm_name = "gpt-4o-mini-2024-07-18"

    subgraph = RetrievePaperSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
    )  # .build_graph()

    # draw mermaid
    # print(subgraph.get_graph().draw_mermaid())

    # state = {
    #     "queries": ["deep learning"],
    # }
    # config = {"recursion_limit": 50}
    # result = subgraph.invoke(state, config=config)
    # print(result["base_candidate_papers_info"])
