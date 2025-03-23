import os
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.writer_subgraph.nodes.generate_note import generate_note
from researchgraph.writer_subgraph.nodes.paper_writing import WritingNode
from researchgraph.writer_subgraph.nodes.convert_to_latex import LatexNode
from researchgraph.writer_subgraph.input_data import writer_subgraph_input_data


class WriterSubgraphInputState(TypedDict):
    base_method_text: str
    new_method: str
    verification_policy: str
    experiment_details: str
    experiment_code: str
    output_text_data: str


class WriterSubgraphHiddenState(TypedDict):
    note: str
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

    def _generate_note_node(self, state: WriterSubgraphState) -> dict:
        print("---WriterSubgraph---")
        print("generate_note_node")
        note = generate_note(state=dict(state))
        return {"note": note}

    def _writeup_node(self, state: WriterSubgraphState) -> dict:
        print("writing_node")
        paper_content = WritingNode(
            llm_name=self.llm_name,
            refine_round=self.refine_round,
        ).execute(
            note=state["note"],
        )
        return {"paper_content": paper_content}

    def _latex_node(self, state: WriterSubgraphState) -> dict:
        print("latex_node")
        tex_text = LatexNode(
            llm_name=self.llm_name,
            figures_dir=self.figures_dir,
            pdf_file_path=self.pdf_file_path,
            save_dir=self.save_dir,
            timeout=30,
        ).execute(
            paper_content=state["paper_content"],
        )
        return {"tex_text": tex_text}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(WriterSubgraphState)
        # make nodes
        graph_builder.add_node("generate_note_node", self._generate_note_node)
        graph_builder.add_node("writeup_node", self._writeup_node)
        graph_builder.add_node("latex_node", self._latex_node)
        # make edges
        graph_builder.add_edge(START, "generate_note_node")
        graph_builder.add_edge("generate_note_node", "writeup_node")
        graph_builder.add_edge("writeup_node", "latex_node")
        graph_builder.add_edge("latex_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    # llm_name = "gpt-4o-mini-2024-07-18"
    save_dir = "/workspaces/researchgraph/data"

    subgraph = WriterSubgraph(
        save_dir=save_dir,
        llm_name=llm_name,
    ).build_graph()
    result = subgraph.invoke(writer_subgraph_input_data)
