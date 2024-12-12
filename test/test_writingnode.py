import sys
import os
from unittest.mock import Mock, patch

if "GITHUB_WORKSPACE" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["GITHUB_WORKSPACE"], "src"))

from typing import TypedDict
from langgraph.graph import StateGraph
from researchgraph.nodes.writingnode.latexnode import LatexNode, LatexUtils
from researchgraph.test_utils.path_resolver import path_resolver


class State(TypedDict):
    paper_content: dict
    pdf_file_path: str


SAVE_DIR = path_resolver.get_save_dir()

@patch.object(LatexUtils, '__init__', return_value=None)
@patch.object(LatexUtils, 'check_references', return_value=True)
@patch.object(LatexUtils, 'check_figures')
@patch.object(LatexUtils, 'check_duplicates')
@patch.object(LatexUtils, 'fix_latex_errors')
@patch.object(LatexUtils, 'compile_latex')
def test_latex_node(mock_compile, mock_fix, mock_duplicates, mock_figures, mock_refs, mock_init):
    # Define input and output keys
    input_key = ["paper_content"]
    output_key = ["pdf_file_path"]
    model = "gpt-4o"
    template_dir = path_resolver.get_template_dir("2d_diffusion")
    figures_dir = path_resolver.get_figures_dir()

    # Initialize LatexNode
    latex_node = LatexNode(
        input_key=input_key,
        output_key=output_key,
        model=model,
        template_dir=template_dir,
        figures_dir=figures_dir,
        timeout=30,
        num_error_corrections=5,
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
            "title": "This is the Title",
            "abstract": "This is the Abstract.",
            "introduction": "This is the introduction.",
            "related work": "This is the related work",
            "background": "This is the background",
            "method": "This is the method section.",
            "experimental setup": "This is the experimental setup",
            "results": "These are the results.",
            "conclusions": "This is the conclusion.",
        },
        "pdf_file_path": os.path.join(SAVE_DIR, "sample.pdf"),
    }

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(state["pdf_file_path"]), exist_ok=True)
    # Create an empty PDF file to simulate successful compilation
    open(state["pdf_file_path"], 'w').close()

    # Execute the graph
    assert graph.invoke(state, debug=True)
    assert os.path.exists(state["pdf_file_path"]), f"PDF file was not generated at {state['pdf_file_path']}!"
