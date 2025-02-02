from langgraph.graph import START, END, StateGraph
from typing import TypedDict

# inputデータ
from researchgraph.graphs.ai_integrator.ai_integrator_v3.input_data import (
    ai_integrator_v3_input_data,
)


from researchgraph.graphs.ai_integrator.ai_integrator_v3.integrate_generator_subgraph.integrate_generator_subgraph import (
    IntegrateGeneratorSubgraph,
    # IntegrateGeneratorState,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.executor_subgraph.executor_subgraph import (
    ExecutorSubgraph,
    # ExecutorState,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.writer_subgraph.writer_subgraph import (
    WriterSubgraph,
)

from researchgraph.graphs.ai_integrator.ai_integrator_v3.integrate_generator_subgraph.llmnode_prompt import (
    ai_integrator_v3_creator_prompt,
)


class AIIntegratorv3State(TypedDict):
    objective: str
    base_github_url: str
    base_method_code: str
    base_method_text: str
    add_github_url: str
    add_method_code: str
    add_method_text: str
    new_method_code: str
    new_method_text: str

    branch_name: str
    github_owner: str
    repository_name: str
    workflow_run_id: int
    save_dir: str
    fix_iteration_count: int
    session_id: str
    output_text_data: str
    error_text_data: str
    devin_url: str

    paper_content: dict
    pdf_file_path: str
    completion: bool


class AIIntegratorv3:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
        ai_integrator_v3_creator_prompt: str,
        max_fix_iteration: int,
        latex_template_file_path: str,
        figures_dir: str,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        # Search Subgraph
        # Generator Subgraph
        self.ai_integrator_v3_creator_prompt = ai_integrator_v3_creator_prompt
        # Executor Subgraph
        self.max_fix_iteration = max_fix_iteration
        # Witer Subgraph
        self.latex_template_file_path = latex_template_file_path
        self.figures_dir = figures_dir

    def build_graph(self):
        # Search Subgraph
        # Generator Subgraph
        generate_subgraph = IntegrateGeneratorSubgraph(
            llm_name=self.llm_name,
            ai_integrator_v3_creator_prompt=self.ai_integrator_v3_creator_prompt,
        )
        # Executor Subgraph
        executor_subgraph = ExecutorSubgraph(
            max_fix_iteration=self.max_fix_iteration,
        )
        # Witer Subgraph
        writer_subgraph = WriterSubgraph(
            llm_name="gpt-4o-mini-2024-07-18",
            latex_template_file_path=self.latex_template_file_path,
            figures_dir=self.figures_dir,
        )

        self.graph_builder = StateGraph(AIIntegratorv3State)
        # make nodes
        self.graph_builder.add_node("generator", generate_subgraph())
        self.graph_builder.add_node("executor", executor_subgraph())
        self.graph_builder.add_node("writer", writer_subgraph())
        # make edges
        self.graph_builder.add_edge(START, "generator")
        self.graph_builder.add_edge("generator", "executor")
        self.graph_builder.add_edge("executor", "writer")
        self.graph_builder.add_edge("writer", END)

        return self.graph_builder.compile()

    def __call__(self, state: AIIntegratorv3State) -> dict:
        self.graph = self.build_graph()
        return self.graph.invoke(state, debug=True)


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    save_dir = "/workspaces/researchgraph/data"
    latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"

    ai_integrator_v3 = AIIntegratorv3(
        llm_name=llm_name,
        save_dir=save_dir,
        ai_integrator_v3_creator_prompt=ai_integrator_v3_creator_prompt,
        max_fix_iteration=3,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
    )
    result = ai_integrator_v3(ai_integrator_v3_input_data)
