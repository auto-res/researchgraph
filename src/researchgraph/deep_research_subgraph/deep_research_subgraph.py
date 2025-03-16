from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph
import asyncio

from researchgraph.deep_research_subgraph.nodes.recurisve_search import recursive_search
from researchgraph.deep_research_subgraph.nodes.generate_report import (
    generate_report,
)
from researchgraph.deep_research_subgraph.input_data import (
    deep_research_subgraph_input_data,
)


class ResearchResult(TypedDict):
    learnings: list[str]
    visited_urls: list[str]


class DeepResearchState(TypedDict):
    query: str
    learnings: list[str]
    visited_urls: list[str]
    analysis_report: str


class DeepResearchSubgraph:
    def __init__(
        self,
        breadth,
        depth,
    ):
        self.breadth = breadth
        self.depth = depth

    async def _recursive_search_node(self, state: DeepResearchState) -> dict:
        research_result = await recursive_search(
            query=state["query"],
            breadth=self.breadth,
            depth=self.depth,
        )
        return {
            "learnings": research_result["learnings"],
            "visited_urls": research_result["visited_urls"],
        }

    def _generate_report_node(self, state: DeepResearchState) -> dict:
        analysis_report = generate_report(
            query=state["query"],
            learnings=state["learnings"],
            visited_urls=state["visited_urls"],
        )
        return {"analysis_report": analysis_report}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(DeepResearchState)
        # make nodes
        graph_builder.add_node("recursive_search_node", self._recursive_search_node)
        graph_builder.add_node("generate_report_node", self._generate_report_node)

        # make edges
        graph_builder.add_edge(START, "recursive_search_node")
        graph_builder.add_edge("recursive_search_node", "generate_report_node")
        graph_builder.add_edge("generate_report_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    graph = DeepResearchSubgraph(
        breadth=2,
        depth=2,
    ).build_graph()

    async def main():
        result = await graph.ainvoke(deep_research_subgraph_input_data)
        print(result["analysis_report"])

    asyncio.run(main())
