from typing import TypedDict
from langgraph.graph import StateGraph


class State(TypedDict):
    code_string: str
    script_save_path: str


class Text2ScriptNode:
    def __init__(self, input_key: str, output_key: str, save_file_path: str):
        self.input_key = input_key
        self.output_key = output_key
        self.save_file_path = save_file_path

    def __call__(self, state: State) -> dict:
        code_string = state[self.input_key]
        with open(self.save_file_path, "w", encoding="utf-8") as file:
            file.write(code_string)
        return {self.output_key: self.save_file_path}


if __name__ == "__main__":
    save_file_path = "/workspaces/researchgraph/data/new_method.py"

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "text2script",
        Text2ScriptNode(
            input_key="code_string",
            output_key="script_save_path",
            save_file_path=save_file_path,
        ),
    )
    graph_builder.set_entry_point("llmtrainer")
    graph_builder.set_finish_point("llmtrainer")
    graph = graph_builder.compile()

    memory = {
        "code_string": "import test",
    }

    graph.invoke(memory, debug=True)
