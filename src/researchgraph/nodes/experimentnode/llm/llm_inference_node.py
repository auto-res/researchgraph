from typing import Optional
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from researchgraph.core.node import Node


class LLMInferenceNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        result_save_path: str,
        dataset_name: str,
        num_inference_data: Optional[int] = None,
    ):
        super().__init__(input_key, output_key)
        self.result_save_path = result_save_path
        self.dataset_name = dataset_name
        self.num_inference_data = num_inference_data
        self.dataset = self._set_up_dataset()

    def _set_up_model(self, model_save_path):
        train_model = AutoModelForCausalLM.from_pretrained(model_save_path).to("cuda")
        train_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        return train_model, train_tokenizer

    def _set_up_dataset(self):
        dataset = load_dataset(self.dataset_name, "main")
        dataset = dataset["test"]
        return dataset

    def execute(self, state) -> dict:
        model_save_path = getattr(state, self.input_key[0])
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
        return {self.output_key[0]: self.result_save_path}
