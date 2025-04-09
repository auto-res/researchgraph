import logging
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.utils.logging_utils import setup_logging

from researchgraph.html_subgraph.nodes.convert_to_html import (
    convert_to_html,
    convert_to_html_prompt,
)
from researchgraph.html_subgraph.nodes.render_html import render_html
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
        save_dir: str, 
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir

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
            save_dir=self.save_dir
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
    import os
    import glob
    from researchgraph.github_utils.graph_wrapper import create_wrapped_subgraph
    from researchgraph.html_subgraph.html_subgraph import HtmlSubgraph

    llm_name = "o3-mini-2025-01-31"
    save_dir = "/workspaces/researchgraph/data"
    figures_dir = "/workspaces/researchgraph/data/images"

    branch_name = "branch-1"
    path="research/research_history.json"

    pdf_files = glob.glob(os.path.join(figures_dir, "*.pdf"))
    extra_files = [
        ("gh-pages", "branches/{{ branch_name }}/", [f"{save_dir}/index.html"]),
        ("gh-pages", "branches/{{ branch_name }}/images/", pdf_files)
    ]

    wrapped_subgraph = create_wrapped_subgraph(
        subgraph=HtmlSubgraph,
        output_state=HtmlSubgraphOutputState, 
        github_owner="auto-res2",
        repository_name="experiment_script_matsuzawa",
        input_branch_name=branch_name,
        input_path=path,
        output_branch_name=branch_name,
        output_path=path,
        extra_files=extra_files, 
        llm_name=llm_name,
        save_dir=save_dir, 
    )

    result = wrapped_subgraph.invoke({})
    print(f"result: {result}")
