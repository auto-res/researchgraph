import os
import os.path as osp

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model


from researchgraph.core.node import Node


class AiderNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        llm_model_name: str,
        save_dir: str,
        file_name: str,
    ):
        super().__init__(input_key, output_key)
        self.main_model = Model(llm_model_name)
        self.save_dir = save_dir
        self.file_name = file_name

    def _aider_setting(self):
        os.makedirs(self.save_dir, exist_ok=True)
        fnames = [
            osp.join(self.save_dir, self.file_name),
        ]
        io = InputOutput(
            yes=True, chat_history_file=osp.join(self.save_dir, "_aider.txt")
        )
        coder = Coder.create(
            main_model=self.main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )
        return coder

    def execute(self, state):
        corrected_code_count = state[self.input_key[0]]
        instruction = state[self.input_key[1]]
        coder = self._aider_setting()
        output = coder.run(instruction)
        return {
            self.output_key[0]: corrected_code_count + 1,
        }
