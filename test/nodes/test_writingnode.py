import sys
import os

if "GITHUB_WORKSPACE" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["GITHUB_WORKSPACE"], "src"))

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from researchgraph.nodes.writingnode.writeup_node import WriteupNode


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


# NOTE：It is executed by Github actions.
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
