import os
from typing import TypedDict
from langgraph.graph import StateGraph

from researchgraph.nodes.codingnode.aider import AiderNode


class State(TypedDict):
    corrected_code_count: int
    instruction: str


SAVE_DIR = os.environ.get("SAVE_DIR", "/workspaces/researchgraph/data")


def test_aider_node():
    llm_model_name = "gpt-4o"
    file_name = "new_script.py"

    file_path = os.path.join(SAVE_DIR, file_name)
    with open(file_path, "w") as file:
        file.write("# This is a new Python script.\n")

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "aider",
        AiderNode(
            input_key=["corrected_code_count", "instruction"],
            output_key=["corrected_code_count"],
            llm_model_name=llm_model_name,
            save_dir=SAVE_DIR,
            file_name=file_name,
        ),
    )

    graph_builder.set_entry_point("aider")
    graph_builder.set_finish_point("aider")
    graph = graph_builder.compile()

    state = {
        "instruction": "Add a function to new_script.py that prints 'Hello, World!",
        "corrected_code_count": 0,
    }

    assert graph.invoke(state, debug=True)
