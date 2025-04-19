import logging

from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph
from typing import TypedDict

from researchgraph.utils.logging_utils import setup_logging
from researchgraph.generator_subgraph.nodes.generator_node import generator_node

from researchgraph.utils.execution_timers import time_node, ExecutionTimeState
from researchgraph.github_utils.graph_wrapper import create_wrapped_subgraph

setup_logging()
logger = logging.getLogger(__name__)


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
    ExecutionTimeState,
):
    pass


class GeneratorSubgraph:
    def __init__(
        self,
        llm_name: str,
    ):
        self.llm_name = llm_name

    @time_node("generator_subgraph", "_generator_node")
    def _generator_node(self, state: GeneratorSubgraphState) -> dict:
        new_method = generator_node(
            llm_name=self.llm_name,
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


Generator = create_wrapped_subgraph(
    GeneratorSubgraph,
    GeneratorSubgraphInputState,
    GeneratorSubgraphOutputState,
)

if __name__ == "__main__":
    llm_name = "o1-2024-12-17"
    github_repository = "auto-res2/test20"
    branch_name = "test"

    retriever = Generator(
        github_repository=github_repository,
        branch_name=branch_name,
        llm_name=llm_name,
    )

    result = retriever.run()
    print(f"result: {result}")
