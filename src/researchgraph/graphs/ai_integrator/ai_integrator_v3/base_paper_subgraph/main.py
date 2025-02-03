from IPython.display import Image
from langgraph.graph import START,END, StateGraph
from researchgraph.graphs.ai_integrator.ai_integrator_v3.base_paper_subgraph.llmnode_prompt import (
    ai_integrator_v3_select_paper_prompt, 
    ai_integrator_v3_summarize_paper_prompt, 
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.base_paper_subgraph.input_data import base_paper_subgraph_input_data
from researchgraph.graphs.ai_integrator.ai_integrator_v3.utils.paper_subgraph import PaperState, PaperSubgraph


class BasePaperSubgraph(PaperSubgraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_builder = StateGraph(PaperState)

        self.graph_builder.add_node("search_papers_node", self._search_papers_node) #TODO: 検索結果が空ならEND
        self.graph_builder.add_node("retrieve_arxiv_text_node", self._retrieve_arxiv_text_node)
        self.graph_builder.add_node("extract_github_urls_node", self._extract_github_urls_node)
        self.graph_builder.add_node("summarize_paper_node", self._summarize_paper_node)
        self.graph_builder.add_node("select_best_paper_node", self._select_best_paper_node)

        self.graph_builder.add_edge(START, "search_papers_node")
        self.graph_builder.add_edge("search_papers_node", "retrieve_arxiv_text_node")
        self.graph_builder.add_edge("retrieve_arxiv_text_node", "extract_github_urls_node")
        self.graph_builder.add_conditional_edges(
            "extract_github_urls_node",
            path=self._decide_next_node,
            path_map={
                "retrieve_arxiv_text_node": "retrieve_arxiv_text_node",
                "summarize_paper_node": "summarize_paper_node",
                "search_papers_node": "search_papers_node",
            },
        )
        self.graph_builder.add_edge("summarize_paper_node", "select_best_paper_node")
        self.graph_builder.add_edge("select_best_paper_node", END)

        self.graph = self.graph_builder.compile()

    def __call__(self, state: PaperState) -> dict:
        result = self.graph.invoke(state, debug=True)
        self._cleanup_result(result)
        result = {f"base_{k}": v for k, v in result.items()}
        print(f'result: {result}')
        return result

    def _cleanup_result(self, result: dict) -> None:
        for key in [
            "process_index", 
            "search_results", 
            "paper_text", 
            "arxiv_url", 
            "github_urls", 
            "candidate_papers", 
            "selected_arxiv_id"
        ]:
            if key in result:
                del result[key]

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v3_base_paper_subgraph.png", "wb") as f:
            f.write(image.data)

if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    num_retrieve_paper = 3
    period_days = 90
    save_dir = "/workspaces/researchgraph/data"
    api_type = "arxiv"
    base_paper_subgraph = BasePaperSubgraph(
        llm_name=llm_name,
        num_retrieve_paper=num_retrieve_paper,
        period_days=period_days,
        save_dir=save_dir,
        api_type=api_type,
        ai_integrator_v3_summarize_paper_prompt=ai_integrator_v3_summarize_paper_prompt,
        ai_integrator_v3_select_paper_prompt=ai_integrator_v3_select_paper_prompt,
    )
    
    base_paper_subgraph(
        state = base_paper_subgraph_input_data, 
        )

    image_dir = "/workspaces/researchgraph/images/"
    base_paper_subgraph.make_image(image_dir)