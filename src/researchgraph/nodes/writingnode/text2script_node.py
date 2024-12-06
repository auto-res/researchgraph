import re
from researchgraph.core.node import Node


class Text2ScriptNode(Node):
    def __init__(
        self, input_key: list[str], output_key: list[str], save_file_path: str
    ):
        super().__init__(input_key, output_key)
        self.save_file_path = save_file_path

    def _remove_code_block_markers_with_regex(self, code_string: str) -> str:
        return re.sub(r"^```[\w]*\n|```$", "", code_string, flags=re.S).strip()

    def execute(self, state) -> dict:
        code_string = state[self.input_key[0]]
        code_string = self._remove_code_block_markers_with_regex(code_string)
        with open(self.save_file_path, "w", encoding="utf-8") as file:
            file.write(code_string)
        return {self.output_key[0]: self.save_file_path}
