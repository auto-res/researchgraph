from typing import TypedDict
from pydantic import BaseModel, create_model
from jinja2 import Environment
from langgraph.graph import StateGraph

from researchgraph.llmnode.llm_client.openai_model import opnai_structured_output


class State(TypedDict):
    week: str
    name: str
    date: str
    participants: list[str]


class DynamicModel(BaseModel):
    pass


class StructuredLLMNode:
    def __init__(
        self,
        input_variable: list[str],
        output_variable: list[str],
        llm_name: str,
        prompt_template: str,
    ):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.llm_name = llm_name
        self.prompt_template = prompt_template
        print("NewLLMNode")
        print(f"input: {self.input_variable}")
        print(f"output: {self.output_variable}")
        self.dynamicmodel = self._create_dynamic_model(DynamicModel)
        self.llm_client = opnai_structured_output(
            self.llm_name, self.prompt_template, self.dynamicmodel
        )

    def _create_dynamic_model(self, base_model: BaseModel):
        default_type = str
        default_required = ...
        fields = {
            field: (default_type, default_required) for field in self.output_variable
        }
        return create_model(
            base_model.__name__,
            **fields,
            __base__=base_model,
        )

    def __call__(self, state) -> dict:
        data = {key: state[key] for key in self.input_variable}

        env = Environment()
        template = env.from_string(self.prompt_template)
        prompt = template.render(data)

        if self.llm_name == "gpt-4o-2024-08-06":
            result_dict = opnai_structured_output(
                self.llm_name, prompt, self.dynamicmodel
            )
        else:
            print("llm_name not found")
        return {**result_dict}


if __name__ == "__main__":
    input_variable = ["week"]
    output_variable = ["name", "date", "participants"]
    llm_name = "gpt-4o-2024-08-06"
    prompt_template = """
    Extract the event information.
    information：Alice and Bob are going to a science fair on {{week}}.
    """

    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "LLMNode",
        StructuredLLMNode(
            input_variable=input_variable,
            output_variable=output_variable,
            llm_name=llm_name,
            prompt_template=prompt_template,
        ),
    )

    graph_builder.set_entry_point("LLMNode")
    graph_builder.set_finish_point("LLMNode")
    graph = graph_builder.compile()

    state = {
        "week": "Friday",
    }
    graph.invoke(state, debug=True)
