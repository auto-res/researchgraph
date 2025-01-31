import os
from pydantic import BaseModel, Field
from langgraph.graph import START,END, StateGraph
from typing import Optional
from researchgraph.core.factory import NodeFactory


class PaperState(BaseModel):
    base_selected_paper: Optional[dict] = None
    queries: list = Field(default_factory=list)
    search_results: Optional[list[dict]] = None
    arxiv_url: Optional[str] = None
    paper_text: Optional[str] = None
    github_urls: Optional[list] = None
    candidate_papers: Optional[list[dict]] = None
    selected_arxiv_id: Optional[str] = None
    selected_paper: Optional[dict] = None
    process_index: int = 0
    

class PaperSubgraph:
    def __init__(
        self,
        llm_name: str, 
        num_retrieve_paper: int,
        period_days: int, 
        save_dir: str,
        api_type: str,
        ai_integrator_v3_select_paper_prompt: str,
        ai_integrator_v3_summarize_paper_prompt: str,
        ai_integrator_v3_generate_queries_prompt: str = None
    ):
        self.llm_name= llm_name
        self.num_retrieve_paper = num_retrieve_paper
        self.period_days = period_days
        self.save_dir = save_dir
        self.api_type = api_type
        self.ai_integrator_v3_select_paper_prompt = ai_integrator_v3_select_paper_prompt
        self.ai_integrator_v3_summarize_paper_prompt = ai_integrator_v3_summarize_paper_prompt
        self.ai_integrator_v3_generate_queries_prompt = ai_integrator_v3_generate_queries_prompt

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.graph_builder = StateGraph(PaperState)
        

        self.graph_builder.add_node("search_papers_node", self._search_papers_node)
        self.graph_builder.add_node("retrieve_arxiv_text_node", self._retrieve_arxiv_text_node)
        self.graph_builder.add_node("extract_github_urls_node", self._extract_github_urls_node)
        #TODO: base_candidate_papersが空の場合の処理を追加
        self.graph_builder.add_node("summarize_paper_node", self._summarize_paper_node)    
        self.graph_builder.add_node("select_best_paper_id_node", self._select_best_paper_id_node)
        self.graph_builder.add_node("convert_paper_id_to_dict_node", self._convert_paper_id_to_dict_node)
        # make edges
        self.graph_builder.add_edge(START, "search_papers_node")
        self.graph_builder.add_edge("search_papers_node", "retrieve_arxiv_text_node")
        self.graph_builder.add_edge("retrieve_arxiv_text_node", "extract_github_urls_node")
        self.graph_builder.add_conditional_edges(
            "extract_github_urls_node",
            path = self._check_loop_condition,
            path_map={"retrieve_arxiv_text_node": "retrieve_arxiv_text_node", "summarize_paper_node": "summarize_paper_node"},
        )
        self.graph_builder.add_edge("summarize_paper_node", "select_best_paper_id_node")
        self.graph_builder.add_edge("select_best_paper_id_node", "convert_paper_id_to_dict_node")
        self.graph_builder.add_edge("convert_paper_id_to_dict_node", END)

        self.graph = self.graph_builder.compile()


    def _search_papers_node(self, state: PaperState) -> PaperState:
        search_result = NodeFactory.create_node(
            node_name="search_papers_node",
            input_key=["queries"],
            output_key=["search_results"],
            num_retrieve_paper=self.num_retrieve_paper,
            period_days=self.period_days,
            api_type=self.api_type,
        ).execute(state)

        state.search_results = search_result.get("search_results")
        return state
    
    def _retrieve_arxiv_text_node(self, state: PaperState) -> PaperState:
        if not state.search_results or state.process_index >= len(state.search_results):
            return state

        state.arxiv_url = state.search_results[state.process_index].get("arxiv_url")

        if not state.arxiv_url:
            return state

        arxiv_result = NodeFactory.create_node(
            node_name="retrieve_arxiv_text_node",
            input_key=["arxiv_url"],
            output_key=["paper_text"],
            save_dir=self.save_dir,
        ).execute(state)

        state.paper_text = arxiv_result.get("paper_text")
        return state

    def _extract_github_urls_node(self, state: PaperState) -> PaperState:
        if not state.paper_text:
            return state

        github_result = NodeFactory.create_node(
            node_name="extract_github_urls_node",
            input_key=["paper_text"],
            output_key=["github_urls"]
        ).execute(state)

        state.github_urls = github_result.get("github_urls")

        if state.github_urls:
            search_result = state.search_results[state.process_index]
            state.candidate_papers = state.candidate_papers or []
            state.candidate_papers.append(
                {
                    "arxiv_id": search_result.get("arxiv_id"),
                    "arxiv_url": state.arxiv_url,
                    "title": search_result.get("title"),
                    "authors": search_result.get("authors"),
                    "publication_date": search_result.get("published_date"),
                    "journal": search_result.get("journal"),
                    "doi": search_result.get("doi"),
                    "paper_text": state.paper_text,
                    "github_urls": state.github_urls
                }
            )
        state.process_index += 1 
        state.paper_text = None
        return state

    def _summarize_paper_node(self, state: PaperState) -> PaperState:
        if not state.candidate_papers:
            return state

        summaries = []
        for paper in state.candidate_papers:
            state.paper_text = paper["paper_text"]
            summary_result = NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["paper_text"],
                output_key=["technical_summary"],   #TODO: LLMの出力形式が不安定
                llm_name=self.llm_name,
                prompt_template=self.ai_integrator_v3_summarize_paper_prompt,
            ).execute(state)

            paper["technical_summary"] = summary_result.get("technical_summary", "")
            summaries.append(paper)

        state.candidate_papers = summaries
        return state

    def _select_best_paper_id_node(self, state: PaperState) -> PaperState:
        if not state.candidate_papers:
            return state

        selected_paper = NodeFactory.create_node(
            node_name="structuredoutput_llmnode",
            input_key=["candidate_papers", "base_selected_paper"],
            output_key=["selected_arxiv_id"],
            llm_name=self.llm_name,
            prompt_template=self.ai_integrator_v3_select_paper_prompt,
        ).execute(state)

        state.selected_arxiv_id = selected_paper.get("selected_arxiv_id")
        return state

    def _convert_paper_id_to_dict_node(self, state: PaperState) -> PaperState:
        if state.selected_arxiv_id is None:
            return state

        if state.candidate_papers:
            state.selected_paper = next(
                (paper for paper in state.candidate_papers if paper["arxiv_id"] == state.selected_arxiv_id),
                None
            )
        else:
            state.selected_paper = None

        return state

    def _check_loop_condition(self, state: PaperState) -> str:
        if state.process_index < len(state.search_results):
            return "retrieve_arxiv_text_node"
        else:
            return "summarize_paper_node"
