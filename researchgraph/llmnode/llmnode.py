# %%
import logging
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph

from llmlinks.link import LLMLink
from llmlinks.llm_client import LLMClient

from .llm_node_setting_template import (
    translater1_setting,
    translater2_setting,
    translater3_setting,
)

logger = logging.getLogger("researchgraph")


class State(TypedDict):
    source: str
    language: str
    translation1: str
    translation2_1: str
    translation2_2: str
    translation3_1: str
    translation3_2: str
    translation3_3: str


class LLMNode:
    def __init__(self, llm_name: str, setting: dict):
        if isinstance(setting, dict):
            self.setting = setting
        else:
            raise ValueError("setting_data must be a dictionary.")

        self.input_variable = self.setting.get("input")
        self.output_variable = self.setting.get("output")
        self.prompt_template = self.setting.get("prompt")
        self.llm = LLMClient(llm_name)
        print("LLMNode initialized")
        print(f"input: {self.input_variable}")
        print(f"output: {self.output_variable}")

    def __call__(self, state: State) -> Any:
        if isinstance(self.input_variable[0], list):
            num_loop = len(self.input)
            for i in range(num_loop):
                prompt = self.prompt_template[i]
                func = LLMLink(
                    self.llm,
                    prompt,
                    self.input_variable[i],
                    self.output_variable[i],
                )

                kwargs = {key: state[key] for key in self.input_variable[i]}
                response = func(**kwargs)
            return {key: response[key] for key in self.output_variable[i]}

        else:
            func = LLMLink(
                self.llm,
                self.prompt_template,
                self.input_variable,
                self.output_variable,
            )
            kwargs = {key: state[key] for key in self.input_variable}
            response = func(**kwargs)
            logger.info("---LLMNode---")
            logger.info(f"LLMNode response:\n{response}")
            return {**response}


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    # llm_name = 'o1-preview-2024-09-12'
    # llm_name = 'o1-mini-2024-09-12'

    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "translater1", LLMNode(llm_name=llm_name, setting=translater1_setting)
    )
    graph_builder.add_node(
        "translater2", LLMNode(llm_name=llm_name, setting=translater2_setting)
    )
    graph_builder.add_node(
        "translater3", LLMNode(llm_name=llm_name, setting=translater3_setting)
    )

    graph_builder.add_edge("translater1", "translater2")
    graph_builder.add_edge("translater2", "translater3")
    graph_builder.set_entry_point("translater1")
    graph_builder.set_finish_point("translater3")
    graph = graph_builder.compile()

    memory = {
        "source": "Hello World!!",
        "language": "japanese",
    }

    print(memory)
    result = graph.invoke(memory, debug=True)
    print(result["new_method_text"])
    print(result["new_method_code"])
