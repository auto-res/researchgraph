from researchgraph.core.node import Node


class RelatedWorkNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
    ):
        super().__init__(input_key, output_key)

    def execute(self, state) -> dict:
        pass
 
        return state
