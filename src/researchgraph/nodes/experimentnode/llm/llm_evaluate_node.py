import re
import pandas as pd
from datasets import load_dataset

from researchgraph.core.node import Node


class LLMEvaluateNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        answer_data_path: str = None,
        dataset_name: str = None,
    ):
        super().__init__(input_key, output_key)
        self.answer_data_path = answer_data_path
        self.dataset_name = dataset_name

    def _parse_llm_output(self, path: str) -> list:
        df = pd.read_csv(path)
        llm_output_dataset = df["llm_output"].to_list()
        result_list = [
            int(match.group(1))
            if (match := re.search(r"#### (\d+)", llm_output))
            else None
            for llm_output in llm_output_dataset
        ]
        return result_list

    def _parse_dataset(self) -> list:
        if self.answer_data_path:
            df = pd.read_csv(self.answer_data_path, index_col=0)
            answer_list = df["answer"].to_list()
        elif self.dataset_name == "openai/gsm8k":
            dataset = load_dataset(self.dataset_name, "main")
            answer_dataset = dataset["test"]["answer"]
            answer_list = [
                int(match.group(1))
                if (match := re.search(r"#### (\d+)", answer))
                else None
                for answer in answer_dataset
            ]
        else:
            raise ValueError(
                "Either `answer_data_path` or `dataset_name` must be provided."
            )
        return answer_list

    def _calculate_accuracy(self, result_list: list, answer_list: list) -> int:
        answer_list = answer_list[: len(result_list)]
        valid_pairs = [(pred, act) for pred, act in zip(result_list, answer_list)]
        correct_count = sum(1 for pred, act in valid_pairs if pred == act)
        accuracy = correct_count / len(valid_pairs) if valid_pairs else 0
        return accuracy

    def execute(self, state) -> dict:
        result_save_path = state[self.input_key]
        result_list = self._parse_llm_output(result_save_path)
        answer_list = self._parse_dataset()
        accuracy = self._calculate_accuracy(result_list, answer_list)
        return {self.output_key: accuracy}
