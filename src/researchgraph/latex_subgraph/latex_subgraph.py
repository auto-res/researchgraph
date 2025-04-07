import os
import logging
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.utils.logging_utils import setup_logging

from researchgraph.latex_subgraph.nodes.convert_to_latex import convert_to_latex, convert_to_latex_prompt
from researchgraph.latex_subgraph.nodes.compile_to_pdf import LatexNode
from researchgraph.latex_subgraph.input_data import latex_subgraph_input_data
from researchgraph.utils.execution_timers import time_node, ExecutionTimeState

setup_logging()
logger = logging.getLogger(__name__)


class LatexSubgraphInputState(TypedDict):
    paper_content: dict[str, str]


class LatexSubgraphHiddenState(TypedDict):
    paper_tex_content: dict[str, str]


class LatexSubgraphOutputState(TypedDict):
    tex_text: str
    pdf_file_path: str


class LatexSubgraphState(
    LatexSubgraphInputState,
    LatexSubgraphHiddenState,
    LatexSubgraphOutputState,
    ExecutionTimeState,
):
    pass


class LatexSubgraph:
    def __init__(
        self,
        save_dir: str,
        llm_name: str,
    ):
        self.save_dir = save_dir
        self.llm_name = llm_name
        self.figures_dir = os.path.join(self.save_dir, "images")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.pdf_file_path = os.path.join(self.save_dir, "paper.pdf")
    
    @time_node("latex_subgraph", "_latex_node")
    def _convert_to_latex_node(self, state: LatexSubgraphState) -> dict:
        paper_tex_content = convert_to_latex(
            llm_name=self.llm_name,
            prompt_template=convert_to_latex_prompt,
            paper_content=state["paper_content"],
        )
        return {"paper_tex_content": paper_tex_content}
    
    @time_node("latex_subgraph", "_latex_node")
    def _latex_node(self, state: LatexSubgraphState) -> dict:
        tex_text = LatexNode(
            llm_name=self.llm_name,
            figures_dir=self.figures_dir,
            pdf_file_path=self.pdf_file_path,
            save_dir=self.save_dir,
            timeout=30,
        ).execute(
            paper_tex_content=state["paper_tex_content"],
        )
        return {
            "tex_text": tex_text, 
            "pdf_file_path": self.pdf_file_path
        }

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(LatexSubgraphState)
        # make nodes
        graph_builder.add_node("convert_to_latex_node", self._convert_to_latex_node)
        graph_builder.add_node("latex_node", self._latex_node)
        # make edges
        graph_builder.add_edge(START, "convert_to_latex_node")
        graph_builder.add_edge("convert_to_latex_node", "latex_node")
        graph_builder.add_edge("latex_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    from researchgraph.github_utils.graph_wrapper import GraphWrapper

    llm_name = "o3-mini-2025-01-31"
    save_dir = "/workspaces/researchgraph/data"

    subgraph = LatexSubgraph(
        save_dir=save_dir,
        llm_name=llm_name,
    ).build_graph()

    wrapped_subgraph = GraphWrapper(
        subgraph=subgraph, 
        github_owner="auto-res2", 
        repository_name="experiment_script_matsuzawa",
        input_branch_name="test",  
        input_paths={
            "paper_content": "data/paper_content.json", 
        }, 
        output_branch_name="test", 
        output_paths={
            "pdf_file_path": "paper/paper.pdf"
        }, 
    ).build_graph()
    result = wrapped_subgraph.invoke({})
    print(f"result: {result}")
