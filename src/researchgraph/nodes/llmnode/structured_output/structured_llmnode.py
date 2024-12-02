from pydantic import BaseModel, create_model
from jinja2 import Environment

from litellm import completion
import ast

from researchgraph.core.node import Node


class DynamicModel(BaseModel):
    pass


class StructuredLLMNode(Node):
    def __init__(
        self,
        input_variable: list[str],
        output_variable: list[str],
        llm_name: str,
        prompt_template: str,
    ):
        super().__init__(input_variable, output_variable)
        self.llm_name = llm_name
        self.prompt_template = prompt_template
        self.dynamicmodel = self._create_dynamic_model(DynamicModel)

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

    def litellm_output(self, llm_name: str, prompt: str, response_field) -> dict:
        response = completion(
            model=llm_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=response_field,
        )
        output = response.choices[0].message.content
        output_dict = ast.literal_eval(output)
        return output_dict

    def execute(self, state) -> dict:
        data = {key: state[key] for key in self.input_variable}

        env = Environment()
        template = env.from_string(self.prompt_template)
        prompt = template.render(data)

        if self.llm_name == "gpt-4o-2024-08-06":
            result_dict = self.litellm_output(self.llm_name, prompt, self.dynamicmodel)
        return {**result_dict}
