from typing import TypedDict
from langgraph.graph import StateGraph

from researchgraph.nodes.experimentnode.llm import LLMSFTTrainNode


class State(TypedDict):
    script_save_path: str
    model_save_path: str


def test_llmtain_node():
    model_name = "meta-llama/Llama-3.2-3B"
    dataset_name = "openai/gsm8k"
    model_save_path = "/workspaces/researchgraph/data"
    input_key = ["script_save_path"]
    output_key = ["model_save_path"]

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llmtrainer",
        LLMSFTTrainNode(
            model_name=model_name,
            dataset_name=dataset_name,
            model_save_path=model_save_path,
            input_key=input_key,
            output_key=output_key,
        ),
    )
    graph_builder.set_entry_point("llmtrainer")
    graph_builder.set_finish_point("llmtrainer")
    graph = graph_builder.compile()

    memory = {
        "script_save_path": "/workspaces/researchgraph/test/experimentnode/new_method.py",
    }

    assert graph.invoke(memory)
