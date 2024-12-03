from typing import TypedDict
from langgraph.graph import StateGraph
from researchgraph.nodes.writingnode.texnode import LatexNode


class State(TypedDict):
    paper_content: dict
    pdf_file_path: str


def test_latex_node():
    # Define input and output keys
    input_key = ["paper_content"]
    output_key = ["pdf_file_path"]
    model = "gpt-4o"
    template_dir = "/workspaces/researchgraph/src/researchgraph/graphs/ai_scientist/templates/2d_diffusion"
    figures_dir = "/workspaces/researchgraph/images"

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
        "pdf_file_path": "/workspaces/researchgraph/data/sample.pdf",
    }

    # Execute the graph
    assert graph.invoke(state, debug=True)
