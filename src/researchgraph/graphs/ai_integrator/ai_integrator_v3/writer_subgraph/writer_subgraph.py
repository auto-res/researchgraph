from IPython.display import Image
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from researchgraph.graphs.ai_integrator.ai_integrator_v3.writer_subgraph.input_data import (
    writer_subgraph_input_data,
)

from researchgraph.nodes.writingnode.writeup_node import WriteupNode
from researchgraph.nodes.writingnode.latexnode import LatexNode
from researchgraph.nodes.writingnode.github_upload_node import GithubUploadNode


class WriterState(TypedDict):
    objective: str
    base_method_text: str
    add_method_text: str
    new_method_text: list
    base_method_code: str
    add_method_code: str
    new_method_code: list
    paper_content: dict
    pdf_file_path: str
    github_owner: str
    repository_name: str
    branch_name: str
    add_github_url: str
    base_github_url: str
    completion: bool


class WriterSubgraph:
    def __init__(
        self,
        llm_name: str,
        latex_template_file_path: str,
        figures_dir: str,
        refine_round: int = 2,
    ):
        self.llm_name = llm_name
        self.refine_round = refine_round
        self.latex_template_file_path = latex_template_file_path
        self.figures_dir = figures_dir

    def _writeup_node(self, state: WriterState) -> dict:
        paper_content = WriteupNode(
            llm_name=self.llm_name,
            refine_round=self.refine_round,
        ).execute(state)
        return {"paper_content": paper_content}

    def _latex_node(self, state: WriterState) -> dict:
        paper_content = state["paper_content"]
        print(paper_content)
        pdf_file_path = state["pdf_file_path"]
        pdf_file_path = LatexNode(
            llm_name=self.llm_name,
            latex_template_file_path=self.latex_template_file_path,
            figures_dir=self.figures_dir,
            timeout=30,
        ).execute(
            paper_content,
            pdf_file_path,
        )
        return {"pdf_file_path": pdf_file_path}

    def _github_upload_node(self, state: WriterState) -> dict:
        pdf_file_path = state["pdf_file_path"]
        github_owner = state["github_owner"]
        repository_name = state["repository_name"]
        branch_name = state["branch_name"]
        title = state["paper_content"]["Title"]
        abstract = state["paper_content"]["Abstract"]
        add_github_url = state["add_github_url"]
        base_github_url = state["base_github_url"]
        devin_url = state["devin_url"]
        all_logs = state
        completion = GithubUploadNode().execute(
            pdf_file_path=pdf_file_path,
            github_owner=github_owner,
            repository_name=repository_name,
            branch_name=branch_name,
            title=title,
            abstract=abstract,
            add_github_url=add_github_url,
            base_github_url=base_github_url,
            devin_url=devin_url,
            all_logs=all_logs,
        )
        return {"completion": completion}

    def build_graph(self):
        graph_builder = StateGraph(WriterState)
        # make nodes
        graph_builder.add_node("writeup_node", self._writeup_node)
        graph_builder.add_node("latex_node", self._latex_node)
        graph_builder.add_node("github_upload_node", self._github_upload_node)
        # make edges
        graph_builder.add_edge(START, "writeup_node")
        graph_builder.add_edge("writeup_node", "latex_node")
        graph_builder.add_edge("latex_node", "github_upload_node")
        graph_builder.add_edge("github_upload_node", END)

        return graph_builder.compile()

    def __call__(self):
        return self.build_graph()

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v3_writer_subgraph.png", "wb") as f:
            f.write(image.data)


if __name__ == "__main__":
    latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"
    # llm_name = "gpt-4o-2024-11-20"
    llm_name = "gpt-4o-mini-2024-07-18"

    writer_subgraph = WriterSubgraph(
        llm_name=llm_name,
        lateX_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
    )
    writer_subgraph().invoke(writer_subgraph_input_data)
    # image_dir = "/workspaces/researchgraph/images/"
    # writer_subgraph.make_image(image_dir)
