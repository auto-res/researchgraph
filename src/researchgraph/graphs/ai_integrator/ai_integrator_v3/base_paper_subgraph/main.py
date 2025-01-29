import os
from IPython.display import Image
from pydantic import BaseModel, Field
from langgraph.graph import START,END, StateGraph
from typing import Optional
from researchgraph.graphs.ai_integrator.ai_integrator_v3.base_paper_subgraph.llmnode_prompt import ai_integrator_v3_select_paper_prompt
from researchgraph.core.factory import NodeFactory
from langgraph.utils.runnable import Runnable


class BasePaperState(BaseModel):
    base_queries: list = Field(default_factory=list)
    base_search_results: Optional[list[dict]] = None
    base_candidate_papers: Optional[list[dict]] = None
    base_selected_paper: Optional[dict] = None

class ArxivTextRetrieveState(BaseModel):
    base_arxiv_url: Optional[str]
    base_paper_text: Optional[str] = None
class GitHubRetrieveState(BaseModel):
    base_paper_text: Optional[str] 
    base_github_url: Optional[list] = None
    

class ArxivRetrieverLoopNode(Runnable):
    def __init__(self, save_dir, num_retrieve_paper=1, **kwargs):
        self.save_dir = save_dir
        self.num_retrieve_paper = num_retrieve_paper

        self.arxiv_retriever = NodeFactory.create_node(
            node_name="retrieve_arxiv_text_node",
            save_dir=self.save_dir,
            input_key=["base_arxiv_url"],
            output_key=["base_paper_text"],
        )
        self.github_retriever = NodeFactory.create_node(
            node_name="retrieve_github_url_node",
            input_key=["base_paper_text"],
            output_key=["base_github_url"]
        )

    def invoke(self, state: BasePaperState, config=None):
        if not hasattr(state, "base_search_results") or not state.base_search_results:
            return state
        
        base_candidate_papers = []

        for index, search_result in enumerate(state.base_search_results):
            arxiv_url = search_result.get("arxiv_url")
            if not arxiv_url:
                continue

            arxiv_state = ArxivTextRetrieveState(base_arxiv_url=arxiv_url)
            arxiv_result = self.arxiv_retriever.execute(arxiv_state)
            paper_text = arxiv_result.get("base_paper_text")
            
            github_state = GitHubRetrieveState(base_paper_text=paper_text)
            github_result = self.github_retriever.execute(github_state)
            github_url = github_result.get("base_github_url")
            if github_url:
                base_candidate_papers.append(
                    {
                        "index": index,
                        "arxiv_url": arxiv_url,
                        "title": search_result.get("title"),
                        "authors": search_result.get("authors"),
                        "publication_date": search_result.get("published_date"),
                        "journal": search_result.get("journal"),
                        "doi": search_result.get("doi"),
                        "paper_text": paper_text[:500],
                        "github_url": github_url
                    }
                )
        state.base_candidate_papers = base_candidate_papers

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
            "retrieve_paper_node",
            NodeFactory.create_node(
                node_name="retrieve_paper_node",
                input_key=["base_queries"],
                output_key=["base_search_results"],
                num_retrieve_paper=self.num_retrieve_paper, 
                period_days=self.period_days, 
                api_type="arxiv",
            ),
        )
        self.graph_builder.add_node(
            "arxiv_retriever_loop",
            ArxivRetrieverLoopNode(
                save_dir=self.save_dir,
                num_retrieve_paper = self.num_retrieve_paper
            ),
        )
        self.graph_builder.add_node(
            "select_paper_node",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["base_candidate_papers"], 
                output_key=["base_selected_paper"],
                llm_name=self.llm_name,
                prompt_template=self.ai_integrator_v3_select_paper_prompt,
            ),
        )
        # make edges
        self.graph_builder.add_edge(START, "retrieve_paper_node")
        self.graph_builder.add_edge("retrieve_paper_node", "arxiv_retriever_loop")
        self.graph_builder.add_edge("arxiv_retriever_loop", "select_paper_node")
        self.graph_builder.add_edge("select_paper_node", END)

        self.graph = self.graph_builder.compile()

    def __call__(self, state: BasePaperState) -> dict:
        result = self.graph.invoke(state, debug=True)
        self._convert_selected_paper(result)
        self._cleanup_result(result)
        print(f'result: {result}')
        return result

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v3_base_paper_subgraph.png", "wb") as f:
            f.write(image.data)

    def _convert_selected_paper(self, result: dict) -> None:
        if "base_selected_paper" in result and result["base_selected_paper"] is not None:
            selected_index = result["base_selected_paper"]
            if isinstance(selected_index, str):
                try:
                    selected_index = int(selected_index)
                except ValueError:
                    result["base_selected_paper"] = None
            if "base_candidate_papers" in result and isinstance(result["base_candidate_papers"], list):
                if 0 <= selected_index < len(result["base_candidate_papers"]):
                    result["base_selected_paper"] = result["base_candidate_papers"][selected_index]
                else:
                    result["base_selected_paper"] = None
            else:
                result["base_selected_paper"] = None

    def _cleanup_result(self, result: dict) -> None:
        if "base_search_results" in result:
            del result["base_search_results"]
        if "base_candidate_papers" in result:
            del result["base_candidate_papers"]