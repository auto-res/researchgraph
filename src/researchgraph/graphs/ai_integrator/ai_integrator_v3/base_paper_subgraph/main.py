import os
from IPython.display import Image
from pydantic import BaseModel, Field
from langgraph.graph import START,END, StateGraph
from typing import Optional
from researchgraph.graphs.ai_integrator.ai_integrator_v3.base_paper_subgraph.llmnode_prompt import ai_integrator_v3_select_paper_prompt
from researchgraph.graphs.ai_integrator.ai_integrator_v3.base_paper_subgraph.input_data import base_paper_subgraph_input_data
from researchgraph.core.factory import NodeFactory


class BasePaperState(BaseModel):
    base_queries: list = Field(default_factory=list)
    base_search_results: Optional[list[dict]] = None
    base_arxiv_url: Optional[str] = None
    base_paper_text: Optional[str] = None
    base_github_urls: Optional[list] = None
    base_candidate_papers: Optional[list[dict]] = None
    base_selected_arxiv_id: Optional[str] = None
    base_selected_paper: Optional[dict] = None
    process_index: int = 0
    
def retrieve_arxiv_text_node(state: BasePaperState, save_dir: str) -> BasePaperState:
    if not state.base_search_results or state.process_index >= len(state.base_search_results):
        return state

    state.base_arxiv_url = state.base_search_results[state.process_index].get("arxiv_url")

    if not state.base_arxiv_url:
        return state

    arxiv_result = NodeFactory.create_node(
        node_name="retrieve_arxiv_text_node",
        input_key=["base_arxiv_url"],
        output_key=["base_paper_text"],
        save_dir=save_dir,
    ).execute(state)

    state.base_paper_text = arxiv_result.get("base_paper_text")
    return state

def extract_github_urls_node(state: BasePaperState) -> BasePaperState:
    if not state.base_paper_text:
        return state

    github_result = NodeFactory.create_node(
        node_name="extract_github_urls_node",
        input_key=["base_paper_text"],
        output_key=["base_github_urls"]
    ).execute(state)

    state.base_github_urls = github_result.get("base_github_urls")

    if state.base_github_urls:
        search_result = state.base_search_results[state.process_index]
        state.base_candidate_papers = state.base_candidate_papers or []
        state.base_candidate_papers.append(
            {
                "arxiv_id": search_result.get("arxiv_id"),
                "arxiv_url": state.base_arxiv_url,
                "title": search_result.get("title"),
                "authors": search_result.get("authors"),
                "publication_date": search_result.get("published_date"),
                "journal": search_result.get("journal"),
                "doi": search_result.get("doi"),
                "paper_text": state.base_paper_text[:500],  #NOTE: デバッグ用に500文字に制限
                "github_urls": state.base_github_urls
            }
        )
    state.process_index += 1 
    state.base_paper_text = None
    return state

def convert_paper_id_to_dict_node(state: BasePaperState) -> BasePaperState:
    if state.base_selected_arxiv_id is None:
        return state

    if state.base_candidate_papers:
        state.base_selected_paper = next(
            (paper for paper in state.base_candidate_papers if paper["arxiv_id"] == state.base_selected_arxiv_id),
            None
        )
    else:
        state.base_selected_paper = None

    return state

class BasePaperSubgraph:
    def __init__(
        self,
        llm_name: str, 
        num_retrieve_paper: int,
        period_days: int, 
        save_dir: str,
        api_type: str,
        ai_integrator_v3_select_paper_prompt: str,
    ):
        self.llm_name= llm_name
        self.num_retrieve_paper = num_retrieve_paper
        self.period_days = period_days
        self.save_dir = save_dir
        self.api_type = api_type
        self.ai_integrator_v3_select_paper_prompt = ai_integrator_v3_select_paper_prompt

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.graph_builder = StateGraph(BasePaperState)

        #TODO: 取得した全ての論文がgithub_urlを持っていないか、アクセス不可能な場合の処理を追加する
        self.graph_builder.add_node(
            "search_papers_node",
            NodeFactory.create_node(
                node_name="search_papers_node",
                input_key=["base_queries"],
                output_key=["base_search_results"],
                num_retrieve_paper=self.num_retrieve_paper, 
                period_days=self.period_days, 
                api_type="arxiv",
            ),
        )
        self.graph_builder.add_node("retrieve_arxiv_text_node", lambda state: retrieve_arxiv_text_node(state, self.save_dir))
        self.graph_builder.add_node("extract_github_urls_node", extract_github_urls_node)
        self.graph_builder.add_node(
            "select_best_paper_id_node",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["base_candidate_papers"], 
                output_key=["base_selected_arxiv_id"],
                llm_name=self.llm_name,
                prompt_template=self.ai_integrator_v3_select_paper_prompt,
            ),
        )
        self.graph_builder.add_node("convert_paper_id_to_dict_node", convert_paper_id_to_dict_node)
        # make edges
        self.graph_builder.add_edge(START, "search_papers_node")
        self.graph_builder.add_edge("search_papers_node", "retrieve_arxiv_text_node")
        self.graph_builder.add_edge("retrieve_arxiv_text_node", "extract_github_urls_node")
        self.graph_builder.add_conditional_edges(
            "extract_github_urls_node",
            path = self._check_loop_condition,
            path_map={"retrieve_arxiv_text_node": "retrieve_arxiv_text_node", "select_best_paper_id_node": "select_best_paper_id_node"},
        )
        self.graph_builder.add_edge("select_best_paper_id_node", "convert_paper_id_to_dict_node")
        self.graph_builder.add_edge("convert_paper_id_to_dict_node", END)

        self.graph = self.graph_builder.compile()

    def __call__(self, state: BasePaperState) -> dict:
        result = self.graph.invoke(state, debug=True)
        self._cleanup_result(result)
        print(f'result: {result}')
        return result

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v3_base_paper_subgraph.png", "wb") as f:
            f.write(image.data)

    def _check_loop_condition(self, state: BasePaperState) -> str:
        if state.process_index < len(state.base_search_results):
            return "retrieve_arxiv_text_node"
        else:
            return "select_best_paper_id_node"

    def _cleanup_result(self, result: dict) -> None:
        if "process_index" in result:
            del result["process_index"]
        if "base_search_results" in result:
            del result["base_search_results"]
        if "base_paper_text" in result:
            del result["base_paper_text"]
        if "base_arxiv_url" in result:
            del result["base_arxiv_url"]
        if "base_github_urls" in result:
            del result["base_github_urls"]
        if "base_candidate_papers" in result:
            del result["base_candidate_papers"]
        if "base_selected_arxiv_id" in result:
            del result["base_selected_arxiv_id"]

if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    num_retrieve_paper = 3
    period_days = 30
    save_dir = "/workspaces/researchgraph/data"
    api_type = "arxiv"
    base_paper_subgraph = BasePaperSubgraph(
        llm_name=llm_name,
        num_retrieve_paper=num_retrieve_paper,
        period_days=period_days,
        save_dir=save_dir,
        api_type=api_type,
        ai_integrator_v3_select_paper_prompt=ai_integrator_v3_select_paper_prompt,
    )
    
    base_paper_subgraph(
        state = base_paper_subgraph_input_data, 
        )

    image_dir = "/workspaces/researchgraph/images/"
    base_paper_subgraph.make_image(image_dir)