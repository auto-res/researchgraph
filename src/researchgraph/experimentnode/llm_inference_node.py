from typing import Optional, TypedDict
from langgraph.graph import StateGraph

from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd


class State(TypedDict):
    model_save_path: str
    result_save_path: str


class LLMInferenceNode:
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        result_save_path: str,
        dataset_name: str,
        num_inference_data: Optional[int] = None,
    ):
        self.input_variable = input_variable
        self.output_variable = output_variable
        print("LLMInferenceNode")
        print(f"input: {self.input_variable}")
        print(f"output: {self.output_variable}")
        self.result_save_path = result_save_path
        self.dataset_name = dataset_name
        self.num_inference_data = num_inference_data
        self.dataset = self._set_up_dataset()

    def _set_up_model(self, model_save_path):
        train_model, train_tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_save_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(train_model)
        return train_model, train_tokenizer

    def _set_up_dataset(self):
        dataset = load_dataset(self.dataset_name, "main")
        dataset = dataset["test"]
        return dataset

    def __call__(self, state: State) -> dict:
        model_save_path = state[self.input_variable]
        model, tokenizer = self._set_up_model(model_save_path)
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

        df = pd.DataFrame({"llm_output": result_list})
        df.to_csv(self.result_save_path, index=False)
        return {self.output_variable: self.result_save_path}


if __name__ == "__main__":
    dataset_name = "openai/gsm8k"
    result_save_path = "/content/test.csv"
    num_inference_data = 20

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llminferencer",
        LLMInferenceNode(
            input_variable="model_save_path",
            output_variable="result_save_path",
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

    graph.invoke(memory)
