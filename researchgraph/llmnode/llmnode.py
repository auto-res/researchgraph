# %%
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph

from llmlinks.link import LLMLink
from llmlinks.llm_client import LLMClient
from pydantic import BaseModel, Field


class State(BaseModel):
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

        self.input = self.setting.get("input")
        self.output = self.setting.get("output")
        self.prompt_template = self.setting.get("prompt")
        self.llm = LLMClient(llm_name)
        print("LLMNode initialized")
        print(f"input: {self.input}")
        print(f"output: {self.output}")

    def __call__(self, state: State) -> Any:
        if isinstance(self.input[0], list):
            num_loop = len(self.input)
            for i in range(num_loop):
                prompt = self.prompt_template[i]
                func = LLMLink(
                    self.llm,
                    prompt,
                    self.input[i],
                    self.output[i],
                )

                kwargs = {key: state[key] for key in self.input[i]}
                response = func(**kwargs)
                for key in self.output[i]:
                    if response[key]:
                        state[key] = response[key][0]
                    else:
                        print(f"Warning: No data returned for [{response[key]}]")

        else:
            func = LLMLink(
                self.llm,
                self.prompt_template,
                self.input,
                self.output,
            )

            kwargs = {key: state[key] for key in self.input}
            print(kwargs)
            response = func(**kwargs)
            print(response)
            for key in self.output:
                state[key] = response[key][0]

        return state


if __name__ == "__main__":
    from llm_node_setting_template import (
        translater1_setting,
        translater2_setting,
        translater3_setting,
    )

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
    graph.invoke(memory, debug=True)
