from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd


class State(TypedDict):
    method_1_model_save_path: str
    method_1_result_path: str


class LLMInferenceNode:
    def __init__(
        self,
        dataset_name,
        result_save_path,
        input_variable,
        output_variable,
        num_inference_data=None,
    ):
        self.dataset_name = dataset_name
        self.result_save_path = result_save_path
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.num_inference_data = num_inference_data
        self.dataset = self._set_up_dataset()

    def _set_up_model(self, state: State):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=state[self.input_variable],
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def _set_up_dataset(self):
        dataset = load_dataset(self.dataset_name, "main")
        dataset = dataset["test"]
        return dataset

    def __call__(self, state: State) -> Any:
        model, tokenizer = self._set_up_model(state)
        result_list = []
        prompt = """### Input:
        {input}
        ### Output:
        {output}"""
        if self.num_inference_data is None:
            for i in range(len(self.dataset)):
                question = self.dataset[i]["question"]
                inputs = tokenizer(
                    [prompt.format(input=question, output="")], return_tensors="pt"
                ).to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
                result_list.append(tokenizer.batch_decode(outputs))
        else:
            for i in range(self.num_inference_data):
                question = self.dataset[i]["question"]
                inputs = tokenizer(
                    [prompt.format(input=question, output="")], return_tensors="pt"
                ).to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
                result_list.append(tokenizer.batch_decode(outputs))

        df = pd.DataFrame({"llm": result_list})
        df.to_csv(self.result_save_path)
        return {self.output_variable: self.result_save_path}


if __name__ == "__main__":
    dataset_name = "openai/gsm8k"
    result_save_path = "/content/test.csv"
    input_variable = "method_1_model_save_path"
    output_variable = "method_1_result_path"
    num_inference_data = 20

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llminferencer",
        LLMInferenceNode(
            dataset_name,
            result_save_path,
            input_variable,
            output_variable,
            num_inference_data,
        ),
    )
    graph_builder.set_entry_point("llminferencer")
    graph_builder.set_finish_point("llminferencer")
    graph = graph_builder.compile()

    memory = {
        "method_1_model_save_path": "lora_model",
    }

    graph.invoke(memory)
