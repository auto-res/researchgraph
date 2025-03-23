from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph
from typing import TypedDict

from researchgraph.generator_subgraph.nodes.generator_node import generator_node

from researchgraph.generator_subgraph.input_data import (
    generator_subgraph_input_data,
)


class GeneratorSubgraphInputState(TypedDict):
    base_method_text: str
    add_method_texts: list[str]


class GeneratorSubgraphHiddenState(TypedDict):
    pass


class GeneratorSubgraphOutputState(TypedDict):
    new_method: str


class GeneratorSubgraphState(
    GeneratorSubgraphInputState,
    GeneratorSubgraphHiddenState,
    GeneratorSubgraphOutputState,
):
    pass


class GeneratorSubgraph:
    def __init__(
        self,
    ):
        pass

    def _generator_node(self, state: GeneratorSubgraphState) -> dict:
        print("---GeneratorSubgraph---")
        print("generator_node")
        new_method = generator_node(
            base_method_text=state["base_method_text"],
            add_method_text_list=state["add_method_texts"],
        )
        return {"new_method": new_method}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(GeneratorSubgraphState)
        # make nodes
        graph_builder.add_node("generator_node", self._generator_node)
        # make edges
        graph_builder.add_edge(START, "generator_node")
        graph_builder.add_edge("generator_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    subgraph = GeneratorSubgraph().build_graph()

    result = subgraph.invoke(generator_subgraph_input_data)
    print(result)
