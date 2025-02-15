from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.deep_research_subgraph.nodes.recursive_search import recursive_search
from researchgraph.deep_research_subgraph.nodes.generate_report import (
    generate_analysis_report,
)
from researchgraph.deep_research_subgraph.nodes.input_data import (
    deep_research_subgraph_input_data,
)


class ResearchResult(TypedDict):
    learnings: list[str]
    visited_urls: list[str]


class DeepResearchState(TypedDict):
    research_node: dict
    analysis_report: str


class DeepResearchSubgraph:
    def __init__(self):
        pass

    async def _recursive_search_node(self, state: DeepResearchState) -> dict:
        research_result = await recursive_search(
            query=state["query"],
        )

        return {"research_result": research_result}

    def _generate_analysis_report_node(self, state: DeepResearchState) -> dict:
        analysis_report = generate_analysis_report(
            learnings=state["research_result"]["learnings"],
            visited_urls=state["research_result"]["visited_urls"],
        )
        return {"analysis_report": analysis_report}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(DeepResearchState)
        # make nodes
        graph_builder.add_node("recursive_search_node", self._recursive_search_node)
        graph_builder.add_node(
            "generate_analysis_report_node", self._generate_analysis_report_node
        )

        # make edges
        graph_builder.add_edge(START, "recursive_search_node")
        graph_builder.add_edge("recursive_search_node", "generate_analysis_report_node")
        graph_builder.add_edge("generate_analysis_report_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    graph = DeepResearchSubgraph(
        max_fix_iteration=3,
    ).build_graph()

    # executor_subgraph.output_mermaid
    result = graph.invoke(deep_research_subgraph_input_data)
