from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from langgraph.graph.graph import CompiledGraph


from researchgraph.retrieve_paper_subgraph.retrieve_paper_subgraph import (
    RetrievePaperSubgraph,
)
from researchgraph.integrate_generator_subgraph.integrate_generator_subgraph import (
    IntegrateGeneratorSubgraph,
    # method_integrate_prompt
)
from researchgraph.executor_subgraph.executor_subgraph import ExecutorSubgraph
from researchgraph.writer_subgraph.writer_subgraph import WriterSubgraph

from researchgraph.input_data import research_graph_input_data


class ResearchGraphState(TypedDict):
    # Input Data
    queries: list[str]
    objective: str
    github_owner: str
    repository_name: str
    save_dir: str
    fix_iteration_count: int

    base_github_url: str
    base_method_code: str
    base_method_text: str
    add_github_url: str
    add_method_code: str
    add_method_text: str

    new_detailed_description_of_methodology: str
    new_novelty: str
    new_experimental_procedure: str
    new_method_code: str

    workflow_run_id: int
    session_id: str
    output_text_data: str
    error_text_data: str
    devin_url: str

    paper_content: dict
    tex_text: str
    completion: bool


class ResearchGraph:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
        # method_integrate_prompt: str,
        max_fix_iteration: int,
        latex_template_file_path: str,
        figures_dir: str,
        pdf_file_path: str, 
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        # Search Subgraph

        # Generator Subgraph
        # self. method_integrate_prompt =  method_integrate_prompt
        # Executor Subgraph
        self.max_fix_iteration = max_fix_iteration
        # Writer Subgraph
        self.latex_template_file_path = latex_template_file_path
        self.figures_dir = figures_dir
        self.pdf_file_path = pdf_file_path

    def build_graph(self) -> CompiledGraph:
        # Search Subgraph
        retrieve_paper_subgraph = RetrievePaperSubgraph(
            llm_name="gpt-4o-mini-2024-07-18",
            save_dir=self.save_dir,
        ).build_graph()
        # Generator Subgraph
        generator_subgraph = IntegrateGeneratorSubgraph(
            llm_name="gpt-4o-2024-11-20",
            # method_integrate_prompt=self. method_integrate_prompt,
        ).build_graph()
        # Executor Subgraph
        executor_subgraph = ExecutorSubgraph(
            max_fix_iteration=self.max_fix_iteration,
        ).build_graph()
        # Witer Subgraph
        writer_subgraph = WriterSubgraph(
            llm_name="gpt-4o-2024-11-20",
            latex_template_file_path=self.latex_template_file_path,
            figures_dir=self.figures_dir,
            pdf_file_path=self.pdf_file_path, 
        ).build_graph()

        graph_builder = StateGraph(ResearchGraphState)
        # make nodes
        graph_builder.add_node("retrieve_paper_subgraph", retrieve_paper_subgraph)
        graph_builder.add_node("generator_subgraph", generator_subgraph)
        graph_builder.add_node("executor_subgraph", executor_subgraph)
        graph_builder.add_node("writer_subgraph", writer_subgraph)
        # make edges
        graph_builder.add_edge(START, "retrieve_paper_subgraph")
        graph_builder.add_edge("retrieve_paper_subgraph", "generator_subgraph")
        graph_builder.add_edge("generator_subgraph", "executor_subgraph")
        graph_builder.add_edge("executor_subgraph", "writer_subgraph")
        graph_builder.add_edge("writer_subgraph", END)

        return graph_builder.compile()


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    save_dir = "/workspaces/researchgraph/data"
    latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"
    pdf_file_path = "/workspaces/researchgraph/data/paper.pdf"

    research_graph = ResearchGraph(
        llm_name=llm_name,
        save_dir=save_dir,
        max_fix_iteration=3,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
        pdf_file_path=pdf_file_path, 
    ).build_graph()

    config = {"recursion_limit": 500}
    result = research_graph.invoke(
        research_graph_input_data,
        config=config,
    )
