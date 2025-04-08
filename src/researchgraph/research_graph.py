import os
import time
import datetime
import logging
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.utils.check_api_key import check_api_key
from researchgraph.utils.logging_utils import setup_logging

from researchgraph.retrieve_paper_subgraph.retrieve_paper_subgraph import (
    RetrievePaperSubgraph,
    RetrievePaperState,
)

from researchgraph.generator_subgraph.generator_subgraph import (
    GeneratorSubgraph,
    GeneratorSubgraphState,
)

from researchgraph.experimental_plan_subgraph.experimental_plan_subgraph import (
    ExperimentalPlanSubgraph,
    ExperimentalPlanSubgraphState,
)
from researchgraph.executor_subgraph.executor_subgraph import (
    ExecutorSubgraph,
    ExecutorSubgraphState,
)
from researchgraph.analytic_subgraph.analytic_subgraph import (
    AnalyticSubgraph,
    AnalyticSubgraphState,
)
from researchgraph.writer_subgraph.writer_subgraph import (
    WriterSubgraph,
    WriterSubgraphState,
)

from researchgraph.latex_subgraph.latex_subgraph import (
    LatexSubgraph,
    LatexSubgraphState,
)

from researchgraph.html_subgraph.html_subgraph import (
    HtmlSubgraph,
    HtmlSubgraphState,
)
from researchgraph.upload_subgraph.upload_subgraph import (
    UploadSubgraph,
    UploadSubgraphState,
)

from researchgraph.retrieve_paper_subgraph.input_data import (
    retrieve_paper_subgraph_input_data,
)
from researchgraph.utils.execution_timers import time_subgraph, ExecutionTimeState
from researchgraph.github_utils.graph_wrapper import GraphWrapper, GraphWrapperState

setup_logging()
logger = logging.getLogger(__name__)


class ResearchGraphState(
    RetrievePaperState,
    GeneratorSubgraphState,
    ExperimentalPlanSubgraphState,
    ExecutorSubgraphState,
    AnalyticSubgraphState,
    WriterSubgraphState,
    LatexSubgraphState,
    HtmlSubgraphState,
    UploadSubgraphState,
    ExecutionTimeState,
    GraphWrapperState,
):
    start_timestamp: float
    execution_logs: dict


class ResearchGraph:
    def __init__(
        self,
        save_dir: str,
        scrape_urls: list[str],
        add_paper_num: int,
        repository: str,
        max_code_fix_iteration: int,
    ):
        check_api_key()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, timestamp)
        os.makedirs(self.save_dir)
        self.scrape_urls = scrape_urls
        self.add_paper_num = add_paper_num
        self.github_owner, self.repository_name = repository.split("/", 1)
        self.max_code_fix_iteration = max_code_fix_iteration

    # NOTE:自作のデータクラスのデータをdictに変換するためのメソッド．githubにアップロードするためにjsonに変換する際に必要．
    @staticmethod
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: ResearchGraph.to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ResearchGraph.to_serializable(v) for v in obj]
        elif hasattr(obj, "__dict__"):
            return ResearchGraph.to_serializable(vars(obj))
        else:
            return obj

    def _make_execution_logs_data(self, state: ResearchGraphState) -> dict:
        return {"execution_logs": ResearchGraph.to_serializable(state)}

    def _init_state(self, state: dict) -> dict:
        state["start_timestamp"] = time.time()
        return state

    def _set_total_execution_time(
        self, state: ResearchGraphState
    ) -> ResearchGraphState:
        start = state.get("start_timestamp", None)
        if start is not None:
            total_duration = round(time.time() - start, 4)
            timings = state.get("execution_time", {})
            timings["__total__"] = [total_duration]
            state["execution_time"] = timings
        return state

    def _check_if_base_paper_found(self, state: ResearchGraphState) -> str:
        if not state.get("selected_base_paper_arxiv_id"):
            logger.warning("No base paper was found. The process will be terminated.")
            return "Stop"
        return "Continue"

    def build_graph(self) -> CompiledGraph:
        # Search Subgraph
        @time_subgraph("retrieve_paper_subgraph")
        def retrieve_paper_subgraph(state: dict):
            subgraph = RetrievePaperSubgraph(
                llm_name="o3-mini-2025-01-31",
                save_dir=self.save_dir,
                scrape_urls=self.scrape_urls,
                add_paper_num=self.add_paper_num,
            ).build_graph()
            return subgraph.invoke(state)

        # Generator Subgraph
        @time_subgraph("generator_subgraph")
        def generator_subgraph(state: dict):
            subgraph = GeneratorSubgraph(
                llm_name="o3-mini-2025-01-31",
            ).build_graph()
            return subgraph.invoke(state)

        # Experimental Plan Subgraph
        @time_subgraph("experimental_plan_subgraph")
        def experimental_plan_subgraph(state: dict):
            subgraph = ExperimentalPlanSubgraph().build_graph()
            return subgraph.invoke(state)

        # Executor Subgraph
        @time_subgraph("executor_subgraph")
        def executor_subgraph(state: dict):
            subgraph = ExecutorSubgraph(
                github_owner=self.github_owner,
                repository_name=self.repository_name,
                save_dir=self.save_dir,
                max_code_fix_iteration=self.max_code_fix_iteration,
            ).build_graph()
            return subgraph.invoke(state)

        @time_subgraph("analytic_subgraph")
        def analytic_subgraph(state: dict):
            subgraph = AnalyticSubgraph(
                llm_name="o3-mini-2025-01-31",
            ).build_graph()
            return subgraph.invoke(state)

        # Writer Subgraph
        @time_subgraph("writer_subgraph")
        def writer_subgraph(state: dict):
            subgraph = WriterSubgraph(
                save_dir=self.save_dir,
                llm_name="o3-mini-2025-01-31",
            ).build_graph()
            return subgraph.invoke(state)

        @time_subgraph("latex_subgraph")
        def latex_subgraph(state: dict):
            subgraph = LatexSubgraph(
                save_dir=self.save_dir,
                llm_name="o3-mini-2025-01-31",
            ).build_graph()
            return subgraph.invoke(state)

        @time_subgraph("html_subgraph")
        def html_subgraph(state: dict):
            subgraph = HtmlSubgraph(
                llm_name="o3-mini-2025-01-31",
            ).build_graph()
            return subgraph.invoke(state)

        # Upload Subgraph
        @time_subgraph("upload_subgraph")
        def upload_subgraph(state: dict):
            subgraph = UploadSubgraph(
                github_owner=self.github_owner,
                repository_name=self.repository_name,
                save_dir=self.save_dir,
            ).build_graph()
            return subgraph.invoke(state)

        graph_builder = StateGraph(ResearchGraphState)
        # make nodes
        graph_builder.add_node("init_state", self._init_state)
        graph_builder.add_node("retrieve_paper_subgraph", retrieve_paper_subgraph)
        graph_builder.add_node("generator_subgraph", generator_subgraph)
        graph_builder.add_node("experimental_plan_subgraph", experimental_plan_subgraph)
        graph_builder.add_node("executor_subgraph", executor_subgraph)
        graph_builder.add_node("analytic_subgraph", analytic_subgraph)
        graph_builder.add_node("writer_subgraph", writer_subgraph)
        graph_builder.add_node("html_subgraph", html_subgraph)
        # graph_builder.add_node("latex_subgraph", latex_subgraph)
        graph_builder.add_node("upload_subgraph", upload_subgraph)
        graph_builder.add_node(
            "make_execution_logs_data", self._make_execution_logs_data
        )
        graph_builder.add_node(
            "set_total_execution_time", self._set_total_execution_time
        )
        # make edges
        graph_builder.add_edge(START, "init_state")
        graph_builder.add_edge("init_state", "retrieve_paper_subgraph")
        graph_builder.add_conditional_edges(
            "retrieve_paper_subgraph",
            path=self._check_if_base_paper_found,
            path_map={"Stop": END, "Continue": "generator_subgraph"},
        )
        graph_builder.add_edge("generator_subgraph", "experimental_plan_subgraph")
        graph_builder.add_edge("experimental_plan_subgraph", "executor_subgraph")
        graph_builder.add_edge("executor_subgraph", "analytic_subgraph")
        graph_builder.add_edge("analytic_subgraph", "writer_subgraph")
        graph_builder.add_edge("writer_subgraph", "html_subgraph")
        # graph_builder.add_edge("html_subgraph", "latex_subgraph")
        graph_builder.add_edge("html_subgraph", "set_total_execution_time")
        graph_builder.add_edge("set_total_execution_time", "make_execution_logs_data")
        graph_builder.add_edge("make_execution_logs_data", "upload_subgraph")
        graph_builder.add_edge("upload_subgraph", END)

        return graph_builder.compile()


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    scrape_urls = [
        "https://icml.cc/virtual/2024/papers.html?filter=title",
        "https://iclr.cc/virtual/2024/papers.html?filter=title",
        # "https://nips.cc/virtual/2024/papers.html?filter=title",
        # "https://cvpr.thecvf.com/virtual/2024/papers.html?filter=title",
        # "https://eccv.ecva.net/virtual/2024/papers.html?filter=title",
    ]
    add_paper_num = 3
    repository = "auto-res2/auto-research"
    max_code_fix_iteration = 5

    research_graph = ResearchGraph(
        save_dir=save_dir,
        scrape_urls=scrape_urls,
        add_paper_num=add_paper_num,
        repository=repository,
        max_code_fix_iteration=max_code_fix_iteration,
    ).build_graph()

    # config = {"recursion_limit": 500}
    # result = research_graph.invoke(
    #     retrieve_paper_subgraph_input_data,
    #     config=config,
    # )
    for event in research_graph.stream(
        retrieve_paper_subgraph_input_data,
        stream_mode="updates",
        config={"recursion_limit": 500},
    ):
        node_name = list(event.keys())[0]
        print("Node Name: ", node_name)
