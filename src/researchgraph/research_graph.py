from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph


from researchgraph.experimental_plan_subgraph.experimental_plan_subgraph import (
    ExperimentalPlanSubgraph,
    ExperimentalPlanSubgraphState,
)
from researchgraph.executor_subgraph.executor_subgraph import (
    ExecutorSubgraph,
    ExecutorSubgraphState,
)
from researchgraph.writer_subgraph.writer_subgraph import (
    WriterSubgraph,
    WriterSubgraphState,
)
from researchgraph.upload_subgraph.upload_subgraph import (
    UploadSubgraph,
    UploadSubgraphState,
)

from researchgraph.input_data import research_graph_input_data


class ResearchGraphState(
    ExperimentalPlanSubgraphState,
    ExecutorSubgraphState,
    WriterSubgraphState,
    UploadSubgraphState,
):
    pass


class ResearchGraph:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
        # Executor Subgraph
        github_owner: str,
        repository_name: str,
        max_code_fix_iteration: int,
        # Witer Subgraph
        latex_template_file_path: str,
        figures_dir: str,
        pdf_file_path: str,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        # Search Subgraph

        # Generator Subgraph
        # Executor Subgraph
        self.github_owner = github_owner
        self.repository_name = repository_name
        self.max_code_fix_iteration = max_code_fix_iteration
        # Witer Subgraph
        self.latex_template_file_path = latex_template_file_path
        self.figures_dir = figures_dir
        self.pdf_file_path = pdf_file_path

    def build_graph(self) -> CompiledGraph:
        # Search Subgraph
        # Generator Subgraph

        # Experimental Plan Subgraph
        generator_subgraph = ExperimentalPlanSubgraph().build_graph()
        # Executor Subgraph
        executor_subgraph = ExecutorSubgraph(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            save_dir=self.save_dir,
            max_code_fix_iteration=self.max_code_fix_iteration,
        ).build_graph()
        # Witer Subgraph
        writer_subgraph = WriterSubgraph(
            llm_name="gpt-4o-2024-11-20",
            latex_template_file_path=self.latex_template_file_path,
            figures_dir=self.figures_dir,
            pdf_file_path=self.pdf_file_path,
        ).build_graph()
        # Upload Subgraph
        upload_subgraph = UploadSubgraph(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            pdf_file_path=self.pdf_file_path,
        ).build_graph()

        graph_builder = StateGraph(ResearchGraphState)
        # make nodes
        # graph_builder.add_node("retrieve_paper_subgraph", retrieve_paper_subgraph)
        graph_builder.add_node("generator_subgraph", generator_subgraph)
        graph_builder.add_node("executor_subgraph", executor_subgraph)
        graph_builder.add_node("writer_subgraph", writer_subgraph)
        graph_builder.add_node("upload_subgraph", upload_subgraph)
        # make edges
        graph_builder.add_edge(START, "generator_subgraph")
        graph_builder.add_edge("generator_subgraph", "executor_subgraph")
        graph_builder.add_edge("executor_subgraph", "writer_subgraph")
        graph_builder.add_edge("writer_subgraph", "upload_subgraph")
        graph_builder.add_edge("upload_subgraph", END)

        return graph_builder.compile()


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    save_dir = "/workspaces/researchgraph/data"
    github_owner = "auto-res2"
    repository_name = "auto-research"
    latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"
    pdf_file_path = "/workspaces/researchgraph/data/paper.pdf"

    research_graph = ResearchGraph(
        llm_name=llm_name,
        save_dir=save_dir,
        github_owner=github_owner,
        repository_name=repository_name,
        max_code_fix_iteration=3,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
        pdf_file_path=pdf_file_path,
    ).build_graph()

    config = {"recursion_limit": 500}
    result = research_graph.invoke(
        research_graph_input_data,
        config=config,
    )
