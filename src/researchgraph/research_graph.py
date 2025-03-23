import os
import datetime

from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.utils.check_api_key import check_api_key

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
from researchgraph.writer_subgraph.writer_subgraph import (
    WriterSubgraph,
    WriterSubgraphState,
)
from researchgraph.upload_subgraph.upload_subgraph import (
    UploadSubgraph,
    UploadSubgraphState,
)

from researchgraph.retrieve_paper_subgraph.input_data import (
    retrieve_paper_subgraph_input_data,
)


class ResearchGraphState(
    RetrievePaperState,
    GeneratorSubgraphState,
    ExperimentalPlanSubgraphState,
    ExecutorSubgraphState,
    WriterSubgraphState,
    UploadSubgraphState,
):
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

    def build_graph(self) -> CompiledGraph:
        # Search Subgraph
        retrieve_paper_subgraph = RetrievePaperSubgraph(
            llm_name="gpt-4o-mini-2024-07-18",
            save_dir=self.save_dir,
            scrape_urls=self.scrape_urls,
            add_paper_num=self.add_paper_num,
        ).build_graph()
        # Generator Subgraph
        generator_subgraph = GeneratorSubgraph().build_graph()
        # Experimental Plan Subgraph
        experimental_plan_subgraph = ExperimentalPlanSubgraph().build_graph()
        # Executor Subgraph
        executor_subgraph = ExecutorSubgraph(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            save_dir=self.save_dir,
            max_code_fix_iteration=self.max_code_fix_iteration,
        ).build_graph()
        # Witer Subgraph
        writer_subgraph = WriterSubgraph(
            save_dir=self.save_dir,
            llm_name="gpt-4o-2024-11-20",
        ).build_graph()
        # Upload Subgraph
        upload_subgraph = UploadSubgraph(
            github_owner=self.github_owner,
            repository_name=self.repository_name,
            save_dir=self.save_dir,
        ).build_graph()

        graph_builder = StateGraph(ResearchGraphState)
        # make nodes
        graph_builder.add_node("retrieve_paper_subgraph", retrieve_paper_subgraph)
        graph_builder.add_node("generator_subgraph", generator_subgraph)
        graph_builder.add_node("experimental_plan_subgraph", experimental_plan_subgraph)
        graph_builder.add_node("executor_subgraph", executor_subgraph)
        graph_builder.add_node("writer_subgraph", writer_subgraph)
        graph_builder.add_node("upload_subgraph", upload_subgraph)
        graph_builder.add_node(
            "make_execution_logs_data", self._make_execution_logs_data
        )
        # make edges
        graph_builder.add_edge(START, "retrieve_paper_subgraph")
        graph_builder.add_edge("retrieve_paper_subgraph", "generator_subgraph")
        graph_builder.add_edge("generator_subgraph", "experimental_plan_subgraph")
        graph_builder.add_edge("experimental_plan_subgraph", "executor_subgraph")
        graph_builder.add_edge("executor_subgraph", "writer_subgraph")
        graph_builder.add_edge("writer_subgraph", "make_execution_logs_data")
        graph_builder.add_edge("make_execution_logs_data", "upload_subgraph")
        graph_builder.add_edge("upload_subgraph", END)

        return graph_builder.compile()


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    scrape_urls = [
        "https://icml.cc/virtual/2024/papers.html?filter=titles",
        "https://iclr.cc/virtual/2024/papers.html?filter=titles",
        # "https://nips.cc/virtual/2024/papers.html?filter=titles",
        # "https://cvpr.thecvf.com/virtual/2024/papers.html?filter=titles",
    ]
    add_paper_num = 5
    repository = "auto-res2/auto-research"
    max_code_fix_iteration = 3

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
