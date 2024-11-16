import pandas as pd
import logging

from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger("researchgraph")


class State(TypedDict):
    index: int
    paper_url: str
    github_url: str


class RetrieveCSVNode:
    def __init__(
        self, input_variable: str, output_variable: list[str], csv_file_path: str
    ):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.csv_file_path = csv_file_path
        print("RetrieveCSVNode initialized")
        print(f"input: {self.input_variable}")
        print(f"output: {self.output_variable}")

    def __call__(self, state: State, config: RunnableConfig) -> State:
        df = pd.read_csv(self.csv_file_path)
        df_row = df.iloc[state[self.input_variable]]

        paper_url = df_row["arxiv_url"]
        github_url = df_row["github_url"]
        logger.info("---RetrieveCSVNode---")
        logger.info(f"Paper URL: {paper_url}")
        logger.info(f"GitHub URL: {github_url}")
        return {
            self.input_variable: state[self.input_variable] + 1,
            self.output_variable[0]: paper_url,
            self.output_variable[1]: github_url,
        }


if __name__ == "__main__":
    csv_file_path = "/workspaces/researchgraph/data/optimization_algorithm.csv"
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "csvretriever",
        RetrieveCSVNode(
            input_variable="index",
            output_variable=["paper_url", "github_url"],
            csv_file_path=csv_file_path,
        ),
    )
    graph_builder.set_entry_point("csvretriever")
    graph_builder.set_finish_point("csvretriever")
    graph = graph_builder.compile()

    memory = {
        "index": 1,
    }

    graph.invoke(memory, debug=True)
