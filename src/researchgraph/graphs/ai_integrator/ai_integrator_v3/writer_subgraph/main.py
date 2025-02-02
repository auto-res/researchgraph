import os
from IPython.display import Image
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from researchgraph.graphs.ai_integrator.ai_integrator_v3.writer_subgraph.input_data import (
    writer_subgraph_input_data,
)
from researchgraph.core.factory import NodeFactory


class WriterState(BaseModel):
    objective: str = Field(default="")
    base_method_text: str = Field(default="")
    add_method_text: str = Field(default="")
    new_method_text: list = Field(default_factory=list)
    base_method_code: str = Field(default="")
    add_method_code: str = Field(default="")
    new_method_code: list = Field(default_factory=list)
    base_method_results: str = Field(default="")
    add_method_results: str = Field(default="")
    new_method_results: list = Field(default_factory=list)
    arxiv_url: str = Field(default="")
    github_url: str = Field(default="")
    paper_content: dict = Field(default_factory=dict)
    pdf_file_path: str = Field(default="")


class WriterSubgraph:
    def __init__(
        self,
        llm_name: str,
        template_file_path: str,
        figures_dir: str,
        refine_round: int = 2,
    ):
        self.llm_name = llm_name
        self.refine_round = refine_round
        self.template_file_path = template_file_path
        self.figures_dir = figures_dir

        self.graph_builder = StateGraph(WriterState)

        self.graph_builder.add_node(
            "writeup_node",
            NodeFactory.create_node(
                node_name="writeup_node",
                input_key=[],
                output_key=["paper_content"],
                llm_name=self.llm_name,
                refine_round=refine_round,
            ),
        )
        self.graph_builder.add_node(
            "latex_node",
            NodeFactory.create_node(
                node_name="latex_node",
                input_key=["paper_content"],
                output_key=["pdf_file_path"],
                llm_name=self.llm_name,
                template_file_path=template_file_path,
                figures_dir=figures_dir,
            ),
        )
        # make edges
        self.graph_builder.add_edge(START, "writeup_node")
        self.graph_builder.add_edge("writeup_node", "latex_node")
        self.graph_builder.add_edge("latex_node", END)

        self.graph = self.graph_builder.compile()

    def __call__(self, state: WriterState) -> dict:
        result = self.graph.invoke(state, debug=True)
        return result

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v3_refiner_subgraph.png", "wb") as f:
            f.write(image.data)


if __name__ == "__main__":
    GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    # TEST_TEMPLATE_DIR = os.path.join(
    #     GITHUB_WORKSPACE, "src/researchgraph/graphs/ai_scientist/templates/2d_diffusion"
    # )
    template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"
    # TEST_FIGURES_DIR = os.path.join(GITHUB_WORKSPACE, "images")

    llm_name = "gpt-4o-2024-08-06"
    writer_subgraph = WriterSubgraph(
        llm_name=llm_name,
        template_file_path=template_file_path,
        figures_dir=figures_dir,
    )
    writer_subgraph(
        state=writer_subgraph_input_data,
    )
    # image_dir = "/workspaces/researchgraph/images/"
    # writer_subgraph.make_image(image_dir)
