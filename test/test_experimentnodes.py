from typing import TypedDict
from langgraph.graph import StateGraph

from researchgraph.nodes.experimentnode.llm import (
    LLMSFTTrainNode,
    LLMInferenceNode,
    LLMEvaluateNode,
)


class State(TypedDict):
    script_save_path: str
    model_save_path: str
    result_save_path: str
    accuracy: str


def test_llmtain_node():
    model_name = "unsloth/Meta-Llama-3.1-8B"
    dataset_name = "openai/gsm8k"
    model_save_path = "model"
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
        "script_save_path": "/content/new_method.py",
    }

    assert graph.invoke(memory)


def test_llminference_node():
    input_key = ["model_save_path"]
    output_key = ["result_save_path"]
    dataset_name = "openai/gsm8k"
    result_save_path = "/content/test.csv"
    num_inference_data = 20

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llminferencer",
        LLMInferenceNode(
            input_key=input_key,
            output_key=output_key,
            dataset_name=dataset_name,
            num_inference_data=num_inference_data,
            result_save_path=result_save_path,
        ),
    )
    graph_builder.set_entry_point("llminferencer")
    graph_builder.set_finish_point("llminferencer")
    graph = graph_builder.compile()

    memory = {
        "model_save_path": "model",
    }

    assert graph.invoke(memory)


def test_llmevaluate_node():
    answer_data_path = "/content/answer_30.csv"
    dataset_name = "openai/gsm8k"
    input_key = ["result_save_path"]
    output_key = ["accuracy"]

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llmevaluater",
        LLMEvaluateNode(
            input_key=input_key,
            output_key=output_key,
            answer_data_path=answer_data_path,
            # dataset_name,
        ),
    )
    graph_builder.set_entry_point("llmevaluater")
    graph_builder.set_finish_point("llmevaluater")
    graph = graph_builder.compile()

    memory = {
        "result_save_path": "/content/test.csv",
    }

    assert graph.invoke(memory, debug=True)
