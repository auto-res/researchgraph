import os
import logging
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.utils.logging_utils import setup_logging

from researchgraph.html_subgraph.nodes.convert_to_html import convert_to_html, convert_to_html_prompt
from researchgraph.html_subgraph.nodes.render_html import render_html
from researchgraph.html_subgraph.input_data import html_subgraph_input_data
from researchgraph.utils.execution_timers import time_node, ExecutionTimeState

setup_logging()
logger = logging.getLogger(__name__)


class HtmlSubgraphInputState(TypedDict):
    paper_content: dict[str, str]


class HtmlSubgraphHiddenState(TypedDict):
    paper_html_content: str


class HtmlSubgraphOutputState(TypedDict):
    full_html: str


class HtmlSubgraphState(
    HtmlSubgraphInputState,
    HtmlSubgraphHiddenState,
    HtmlSubgraphOutputState,
    ExecutionTimeState,
):
    pass


class HtmlSubgraph:
    def __init__(
        self,
        llm_name: str,
    ):
        self.llm_name = llm_name
    
    @time_node("html_subgraph", "_convert_to_html_node")
    def _convert_to_html_node(self, state: HtmlSubgraphState) -> dict:
        paper_html_content = convert_to_html(
            llm_name=self.llm_name,
            prompt_template=convert_to_html_prompt,
            paper_content=state["paper_content"],
        )
        return {"paper_html_content": paper_html_content}
    
    @time_node("html_subgraph", "_render_html_node")
    def _render_html_node(self, state: HtmlSubgraphState) -> dict:
        full_html = render_html(
            paper_html_content=state["paper_html_content"],
        )
        return {"full_html": full_html}


    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(HtmlSubgraphState)
        # make nodes
        graph_builder.add_node("convert_to_html_node", self._convert_to_html_node)
        graph_builder.add_node("render_html_node", self._render_html_node)
        # make edges
        graph_builder.add_edge(START, "convert_to_html_node")
        graph_builder.add_edge("convert_to_html_node", "render_html_node")
        graph_builder.add_edge("render_html_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    from researchgraph.github_utils.graph_wrapper import GraphWrapper

    llm_name = "o3-mini-2025-01-31"
    input_branch_name = "test"

    subgraph = HtmlSubgraph(
        llm_name=llm_name,
    ).build_graph()
    
    wrapped_subgraph = GraphWrapper(
        subgraph=subgraph, 
        github_owner="auto-res2", 
        repository_name="experiment_script_matsuzawa",
        input_branch_name=input_branch_name, 
        input_paths={
            "paper_content": "data/paper_content.json", 
        }, 
        output_branch_name="gh-pages", 
        output_paths={
            "full_html": f"{input_branch_name}/index.html", 
        }, 
    ).build_graph()
    result = wrapped_subgraph.invoke({})
    print(f"result: {result}")