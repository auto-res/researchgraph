from llmlinks.link import LLMLink
from llmlinks.llm_client import LLMClient

from researchgraph.core.node import Node


class LLMLinksLLMNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        llm_name: str,
        prompt_template: str,
    ):
        super().__init__(input_key, output_key)
        self.llm_name = llm_name
        self.prompt_template = prompt_template
        self.llm = LLMClient(llm_name)

    def execute(self, state):
        if isinstance(self.input_key[0], list):
            num_loop = len(self.input_key)
            for i in range(num_loop):
                prompt = self.prompt_template[i]
                func = LLMLink(
                    self.llm,
                    prompt,
                    self.input_key[i],
                    self.output_key[i],
                )
                kwargs = {key: state[key] for key in self.input_key[i]}
                response = func(**kwargs)
                for key in self.output_key[i]:
                    state[key] = response[key]
            return {**state}

        else:
            func = LLMLink(
                self.llm,
                self.prompt_template,
                self.input_key,
                self.output_key,
            )
            kwargs = {key: state[key] for key in self.input_key}
            response = func(**kwargs)
            return {**response}
