import json
from IPython.display import Image
from langgraph.graph import START,END, StateGraph
from researchgraph.graphs.ai_integrator.ai_integrator_v3.add_paper_subgraph.llmnode_prompt import (
    ai_integrator_v3_summarize_paper_prompt, 
    ai_integrator_v3_generate_queries_prompt, 
    ai_integrator_v3_select_paper_prompt, 
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.add_paper_subgraph.input_data import add_paper_subgraph_input_data
from researchgraph.graphs.ai_integrator.ai_integrator_v3.utils.paper_subgraph import PaperState, PaperSubgraph
from researchgraph.core.factory import NodeFactory


class AddPaperSubgraph(PaperSubgraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ai_integrator_v3_generate_queries_prompt = ai_integrator_v3_generate_queries_prompt
        self.graph_builder = StateGraph(PaperState)

        # self.graph_builder.add_node("generate_queries_node", self._generate_queries_node) #TDOO: add generate_queries_node
        self.graph_builder.add_node("search_papers_node", self._search_papers_node)
        self.graph_builder.add_node("retrieve_arxiv_text_node", self._retrieve_arxiv_text_node)
        self.graph_builder.add_node("extract_github_urls_node", self._extract_github_urls_node)
        self.graph_builder.add_node("summarize_paper_node", self._summarize_paper_node)
        self.graph_builder.add_node("select_best_paper_id_node", self._select_best_paper_id_node)
        self.graph_builder.add_node("convert_paper_id_to_dict_node", self._convert_paper_id_to_dict_node)

        # self.graph_builder.add_edge(START, "generate_queries_node")
        # self.graph_builder.add_edge("generate_queries_node", "search_papers_node")
        self.graph_builder.add_edge(START, "search_papers_node")
        self.graph_builder.add_edge("search_papers_node", "retrieve_arxiv_text_node")
        self.graph_builder.add_edge("retrieve_arxiv_text_node", "extract_github_urls_node")
        self.graph_builder.add_conditional_edges(
            "extract_github_urls_node", 
            path=self._check_loop_condition,
            path_map={"retrieve_arxiv_text_node": "retrieve_arxiv_text_node", "summarize_paper_node": "summarize_paper_node"},
        )
        self.graph_builder.add_edge("summarize_paper_node", "select_best_paper_id_node")
        self.graph_builder.add_edge("select_best_paper_id_node", "convert_paper_id_to_dict_node")
        self.graph_builder.add_edge("convert_paper_id_to_dict_node", END)

        self.graph = self.graph_builder.compile()

    def __call__(self, state: PaperState) -> dict:
        result = self.graph.invoke(state, debug=True)
        self._cleanup_result(result)
        result = {f"add_{k}": v for k, v in result.items()}
        print(f'result: {result}')
        return result

    def _generate_queries_node(self, state: PaperState) -> PaperState:
        generate_queries_result = NodeFactory.create_node(
            node_name="structuredoutput_llmnode",
            input_key=["base_selected_paper"],
            output_key=["queries"],
            llm_name=self.llm_name,
            prompt_template=self.ai_integrator_v3_generate_queries_prompt,
        ).execute(state)

        raw_queries = generate_queries_result.get("queries", "")

        print(f"Raw LLM output: {raw_queries}")

        if not raw_queries or not isinstance(raw_queries, str):
            print("LLM returned empty or non-string output.")
            state.queries = ["default query"]
            return state

        try:
            queries_dict = json.loads(raw_queries)
            if isinstance(queries_dict, dict) and "queries" in queries_dict:
                queries = queries_dict["queries"]
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    state.queries = queries
                    return state
                else:
                    print("'queries' is not a valid list of strings. Using default query.")
            else:
                print(f"'queries' key is missing in LLM output: {queries_dict}. Using default query.")
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM output as JSON: {e}")
            queries_list = [q.strip() for q in raw_queries.split(",") if q.strip()]
            if queries_list:
                state.queries = queries_list
                return state
            else:
                print("Failed to parse queries, using default query")

        state.queries = ["default query"]
        return state
        
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
        with open(path + "ai_integrator_v3_add_paper_subgraph.png", "wb") as f:
            f.write(image.data)

if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    num_retrieve_paper = 3
    period_days = 90
    save_dir = "/workspaces/researchgraph/data"
    api_type = "arxiv"
    add_paper_subgraph = AddPaperSubgraph(
        llm_name=llm_name,
        num_retrieve_paper=num_retrieve_paper,
        period_days=period_days,
        save_dir=save_dir,
        api_type=api_type,
        ai_integrator_v3_generate_queries_prompt=ai_integrator_v3_generate_queries_prompt,
        ai_integrator_v3_summarize_paper_prompt=ai_integrator_v3_summarize_paper_prompt,
        ai_integrator_v3_select_paper_prompt=ai_integrator_v3_select_paper_prompt,
    )
    
    add_paper_subgraph(
        state = add_paper_subgraph_input_data, 
        )

    image_dir = "/workspaces/researchgraph/images/"
    add_paper_subgraph.make_image(image_dir)