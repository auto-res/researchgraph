from llmlinks.link import LLMLink
from llmlinks.llm_client import LLMClient

from researchgraph.core.node import Node


class LLMLinksLLMNode(Node):
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
        self.llm = LLMClient(llm_name)

    def execute(self, state):
        if isinstance(self.input_variable[0], list):
            num_loop = len(self.input_variable)
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
                for key in self.output_variable[i]:
                    state[key] = response[key]
            return {**state}

        else:
            func = LLMLink(
                self.llm,
                self.prompt_template,
                self.input_variable,
                self.output_variable,
            )
            kwargs = {key: state[key] for key in self.input_variable}
            response = func(**kwargs)
            return {**response}
