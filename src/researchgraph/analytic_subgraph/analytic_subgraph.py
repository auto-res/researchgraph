import logging

from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph
from typing import TypedDict

from researchgraph.analytic_subgraph.nodes.analytic_node import analytic_node

from researchgraph.utils.logging_utils import setup_logging

from researchgraph.generator_subgraph.input_data import (
    generator_subgraph_input_data,
)
from researchgraph.utils.execution_timers import time_node, ExecutionTimeState

setup_logging()
logger = logging.getLogger(__name__)


class AnalyticSubgraphInputState(TypedDict):
    new_method: str
    verification_policy: str
    experiment_code: str
    output_text_data: str


class AnalyticSubgraphHiddenState(TypedDict):
    pass


class AnalyticSubgraphOutputState(TypedDict):
    analysis_results: str
    analysis_report: str


class AnalyticSubgraphState(
    AnalyticSubgraphInputState,
    AnalyticSubgraphHiddenState,
    AnalyticSubgraphOutputState,
    ExecutionTimeState,
):
    pass


class AnalyticSubgraph:
    def __init__(
        self,
        llm_name: str,
    ):
        self.llm_name = llm_name

    @time_node("analytic_subgraph", "_analytic_node")
    def _analytic_node(self, state: AnalyticSubgraphState) -> dict:
        logger.info("---AnalyticSubgraph---")
        analysis_report, analysis_results = analytic_node(
            llm_name=self.llm_name,
            new_method=state["new_method"],
            verification_policy=state["verification_policy"],
            experiment_code=state["experiment_code"],
            output_text_data=state["output_text_data"],
        )
        return {
            "analysis_report": analysis_report,
            "analysis_results": analysis_results,
        }

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(AnalyticSubgraphState)
        # make nodes
        graph_builder.add_node("analytic_node", self._analytic_node)
        # make edges
        graph_builder.add_edge(START, "analytic_node")
        graph_builder.add_edge("analytic_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    llm_name = "o1-2024-12-17"
    subgraph = AnalyticSubgraph(
        llm_name=llm_name,
    ).build_graph()

    result = subgraph.invoke(generator_subgraph_input_data)
    print(result)
