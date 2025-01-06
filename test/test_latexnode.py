import os
import pytest
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from researchgraph.nodes.writingnode.latexnode import LatexNode


GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
TEST_TEMPLATE_DIR = os.path.join(GITHUB_WORKSPACE, "src/researchgraph/graphs/ai_scientist/templates/2d_diffusion")
TEST_FIGURES_DIR = os.path.join(GITHUB_WORKSPACE, "images")
TEST_PDF_FILE = os.path.join(GITHUB_WORKSPACE, "data/test_output.pdf")


class State(BaseModel):
    paper_content: dict = Field(default_factory=dict)
    pdf_file_path: str = Field(default="")

def test_latex_node():
    # Define input and output keys
    input_key = ["paper_content"]
    output_key = ["pdf_file_path"]
    llm_name = "gpt-4o"
    template_dir = TEST_TEMPLATE_DIR
    figures_dir = TEST_FIGURES_DIR

    # Initialize LatexNode
    latex_node = LatexNode(
        input_key=input_key,
        output_key=output_key,
        llm_name=llm_name,
        template_dir=template_dir,
        figures_dir=figures_dir,
        timeout=60,
    )

    # Create the StateGraph and add node
    graph_builder = StateGraph(State)
    graph_builder.add_node("latexnode", latex_node)
    graph_builder.set_entry_point("latexnode")
    graph_builder.set_finish_point("latexnode")
    graph = graph_builder.compile()

    # Define initial state
    state = {
        "paper_content": {
            "title": "test title",
            "abstract": "Abstract.",
            "introduction": "This is the introduction.",
            "related work": "This is the related work",
            "background": "This is the background",
            "method": "This is the method section.",
            "experimental setup": "This is the experimental setup",
            "results": "These are the results.",
            "conclusions": "This is the conclusion.",
        },
        "pdf_file_path": TEST_PDF_FILE,  
    }

    # Execute the graph
    assert graph.invoke(state, debug=True)

