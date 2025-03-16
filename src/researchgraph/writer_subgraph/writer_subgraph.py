from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.writer_subgraph.nodes.writeup_node import WriteupNode
from researchgraph.writer_subgraph.nodes.latexnode import LatexNode
from researchgraph.writer_subgraph.input_data import writer_subgraph_input_data


class WriterSubgraphInputState(TypedDict):
    objective: str
    base_method_text: str
    add_method_text: str
    new_method_text: list
    base_method_code: str
    add_method_code: str
    new_method_code: list


class WriterSubgraphHiddenState(TypedDict):
    paper_content: dict


class WriterSubgraphOutputState(TypedDict):
    tex_text: str


class WriterSubgraphState(
    WriterSubgraphInputState, WriterSubgraphHiddenState, WriterSubgraphOutputState
):
    pass


class WriterSubgraph:
    def __init__(
        self,
        llm_name: str,
        latex_template_file_path: str,
        figures_dir: str,
        pdf_file_path: str,
        refine_round: int = 2,
    ):
        self.llm_name = llm_name
        self.refine_round = refine_round
        self.latex_template_file_path = latex_template_file_path
        self.figures_dir = figures_dir
        self.pdf_file_path = pdf_file_path

    def _writeup_node(self, state: WriterSubgraphState) -> dict:
        print("---WriterSubgraph---")
        print("writeup_node")
        paper_content = WriteupNode(
            llm_name=self.llm_name,
            refine_round=self.refine_round,
        ).execute(state)
        return {"paper_content": paper_content}

    def _latex_node(self, state: WriterSubgraphState) -> dict:
        print("latex_node")
        paper_content = state["paper_content"]
        print(paper_content)
        tex_text = LatexNode(
            llm_name=self.llm_name,
            latex_template_file_path=self.latex_template_file_path,
            figures_dir=self.figures_dir,
            pdf_file_path=self.pdf_file_path,
            timeout=30,
        ).execute(
            paper_content,
        )
        return {"tex_text": tex_text}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(WriterSubgraphState)
        # make nodes
        graph_builder.add_node("writeup_node", self._writeup_node)
        graph_builder.add_node("latex_node", self._latex_node)
        # make edges
        graph_builder.add_edge(START, "writeup_node")
        graph_builder.add_edge("writeup_node", "latex_node")
        graph_builder.add_edge("latex_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"
    pdf_file_path = "/workspaces/researchgraph/data/test_output.pdf"
    # llm_name = "gpt-4o-2024-11-20"
    llm_name = "gpt-4o-mini-2024-07-18"

    subgraph = WriterSubgraph(
        llm_name=llm_name,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
        pdf_file_path=pdf_file_path,
    ).build_graph()
    result = subgraph.invoke(writer_subgraph_input_data)
