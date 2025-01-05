import sys
import os

if "GITHUB_WORKSPACE" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["GITHUB_WORKSPACE"], "src"))

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from researchgraph.nodes.writingnode.writeup_node import WriteupNode
# from researchgraph.nodes.writingnode.latexnode import LatexNode
# from researchgraph.core.factory import NodeFactory


class State(BaseModel):
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


def test_writeup_node():
    # Define input and output keys
    input_key = []
    output_key = ["paper_content"]
    llm_name = "gpt-4o"
    refine_round = 2

    # Initialize WriteupNode
    writeup_node = WriteupNode(
        input_key=input_key,
        output_key=output_key,
        llm_name=llm_name,
        refine_round=refine_round, 
        # refine_only=False, 
        # target_sections=Node
    )

    # Create the StateGraph and add node
    graph_builder = StateGraph(State)
    graph_builder.add_node("writeupnode", writeup_node)
    graph_builder.set_entry_point("writeupnode")
    graph_builder.set_finish_point("writeupnode")
    graph = graph_builder.compile()

    state = {
        "objective": "Researching optimizers for fine-tuning LLMs.",
        "base_method_text": "Baseline method description...",
        "add_method_text": "Added method description...",
        "new_method_text": ["New combined method description..."],
        "base_method_code": "def base_method(): pass",
        "add_method_code": "def add_method(): pass",
        "new_method_code": ["def new_method(): pass"],
        "base_method_results": "Accuracy: 0.85",
        "add_method_results": "Accuracy: 0.88",
        "new_method_results": ["Accuracy: 0.92"],
        "arxiv_url": "https://arxiv.org/abs/1234.5678",
        "github_url": "https://github.com/example/repo",
        "paper_content": {}, 
        # "*_analysis": 
    }

    # Execute the graph
    assert graph.invoke(state, debug=True)

# def test_latex_node():
#     # Define input and output keys
#     input_key = ["paper_content"]
#     output_key = ["pdf_file_path"]
#     model = "gpt-4o"
#     SAVE_DIR = os.environ.get("SAVE_DIR", "/workspaces/researchgraph/data")
#     GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", os.path.abspath(os.path.join(os.getcwd(), "..")))
#     template_dir = os.path.join(GITHUB_WORKSPACE, "src/researchgraph/graphs/ai_scientist/templates/2d_diffusion")
#     figures_dir = os.path.join(GITHUB_WORKSPACE, "images")

#     # Initialize LatexNode
#     latex_node = LatexNode(
#         input_key=input_key,
#         output_key=output_key,
#         model=model,
#         template_dir=template_dir,
#         figures_dir=figures_dir,
#         timeout=30,
#         num_error_corrections=5,
#     )

#     # Create the StateGraph and add node
#     graph_builder = StateGraph(State)
#     graph_builder.add_node("latexnode", latex_node)
#     graph_builder.set_entry_point("latexnode")
#     graph_builder.set_finish_point("latexnode")
#     graph = graph_builder.compile()

#     # Define initial state
#     state = {
#         "paper_content": {
#             "title": "This is the Title",
#             "abstract": "This is the Abstract.",
#             "introduction": "This is the introduction.",
#             "related work": "This is the related work",
#             "background": "This is the background",
#             "method": "This is the method section.",
#             "experimental setup": "This is the experimental setup",
#             "results": "These are the results.",
#             "conclusions": "This is the conclusion.",
#         },
#         "pdf_file_path": os.path.join(SAVE_DIR, "sample.pdf"), 
#     }

#     # Execute the graph
#     assert graph.invoke(state, debug=True)
#     assert os.path.exists(state["pdf_file_path"]), f"PDF file was not generated at {state['pdf_file_path']}!"
