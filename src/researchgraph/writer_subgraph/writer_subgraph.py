import os
import logging
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.utils.logging_utils import setup_logging

from researchgraph.writer_subgraph.nodes.generate_note import generate_note
from researchgraph.writer_subgraph.nodes.paper_writing import WritingNode
from researchgraph.writer_subgraph.nodes.convert_to_latex import convert_to_latex, convert_to_latex_prompt
from researchgraph.writer_subgraph.nodes.compile_to_pdf import LatexNode
from researchgraph.writer_subgraph.nodes.convert_to_html import convert_to_html, convert_to_html_prompt
from researchgraph.writer_subgraph.nodes.render_html import render_html
from researchgraph.writer_subgraph.input_data import writer_subgraph_input_data
from researchgraph.utils.execution_timers import time_node, ExecutionTimeState

setup_logging()
logger = logging.getLogger(__name__)


class WriterSubgraphInputState(TypedDict):
    base_method_text: str
    new_method: str
    verification_policy: str
    experiment_details: str
    experiment_code: str
    output_text_data: str


class WriterSubgraphHiddenState(TypedDict):
    note: str
    paper_content: dict[str, str]
    paper_tex_content: dict[str, str]
    paper_html_content: str


class WriterSubgraphOutputState(TypedDict):
    tex_text: str
    full_html: str


class WriterSubgraphState(
    WriterSubgraphInputState,
    WriterSubgraphHiddenState,
    WriterSubgraphOutputState,
    ExecutionTimeState,
):
    pass


class WriterSubgraph:
    def __init__(
        self,
        save_dir: str,
        llm_name: str,
        refine_round: int = 4,
    ):
        self.save_dir = save_dir
        self.llm_name = llm_name
        self.refine_round = refine_round
        self.figures_dir = os.path.join(self.save_dir, "images")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.pdf_file_path = os.path.join(self.save_dir, "paper.pdf")

    @time_node("writer_subgraph", "_generate_note_node")
    def _generate_note_node(self, state: WriterSubgraphState) -> dict:
        logger.info("---WriterSubgraph---")
        note = generate_note(state=dict(state), figures_dir=self.figures_dir)
        return {"note": note}

    @time_node("writer_subgraph", "_writeup_node")
    def _writeup_node(self, state: WriterSubgraphState) -> dict:
        paper_content = WritingNode(
            llm_name=self.llm_name,
            refine_round=self.refine_round,
        ).execute(
            note=state["note"],
        )
        return {"paper_content": paper_content}
    
    @time_node("writer_subgraph", "_latex_node")
    def _convert_to_latex_node(self, state: WriterSubgraphState) -> dict:
        paper_tex_content = convert_to_latex(
            llm_name=self.llm_name,
            prompt_template=convert_to_latex_prompt,
            paper_content=state["paper_content"],
        )
        return {"paper_tex_content": paper_tex_content}
    
    @time_node("writer_subgraph", "_latex_node")
    def _latex_node(self, state: WriterSubgraphState) -> dict:
        tex_text = LatexNode(
            llm_name=self.llm_name,
            figures_dir=self.figures_dir,
            pdf_file_path=self.pdf_file_path,
            save_dir=self.save_dir,
            timeout=30,
        ).execute(
            paper_tex_content=state["paper_tex_content"],
        )
        return {"tex_text": tex_text}
    
    @time_node("writer_subgraph", "_convert_to_html_node")
    def _convert_to_html_node(self, state: WriterSubgraphState) -> dict:
        paper_html_content = convert_to_html(
            llm_name=self.llm_name,
            prompt_template=convert_to_html_prompt,
            paper_content=state["paper_content"],
        )
        return {"paper_html_content": paper_html_content}
    
    @time_node("writer_subgraph", "_render_html_node")
    def _render_html_node(self, state: WriterSubgraphState) -> dict:
        full_html = render_html(
            paper_html_content=state["paper_html_content"],
            save_dir=self.save_dir,
        )
        return {"full_html": full_html}


    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(WriterSubgraphState)
        # make nodes
        graph_builder.add_node("generate_note_node", self._generate_note_node)
        graph_builder.add_node("writeup_node", self._writeup_node)
        graph_builder.add_node("convert_to_latex_node", self._convert_to_latex_node)
        graph_builder.add_node("latex_node", self._latex_node)
        graph_builder.add_node("convert_to_html_node", self._convert_to_html_node)
        graph_builder.add_node("render_html_node", self._render_html_node)
        # make edges
        graph_builder.add_edge(START, "generate_note_node")
        graph_builder.add_edge("generate_note_node", "writeup_node")
        graph_builder.add_edge("writeup_node", "convert_to_latex_node")
        graph_builder.add_edge("writeup_node", "convert_to_html_node")
        graph_builder.add_edge("convert_to_latex_node", "latex_node")
        graph_builder.add_edge("convert_to_html_node", "render_html_node")
        graph_builder.add_edge("render_html_node", END)
        graph_builder.add_edge("latex_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    llm_name = "o3-mini-2025-01-31"
    # llm_name = "gpt-4o-2024-11-20"
    # llm_name = "gpt-4o-mini-2024-07-18"
    save_dir = "/workspaces/researchgraph/data"

    subgraph = WriterSubgraph(
        save_dir=save_dir,
        llm_name=llm_name,
        refine_round=1,
    ).build_graph()
    result = subgraph.invoke(writer_subgraph_input_data)
