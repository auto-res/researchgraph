import pandas as pd

from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig


class State(TypedDict):
    index: int
    paper_url: str
    github_url: str


class CSVNode:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def __call__(self, state: State, config: RunnableConfig) -> Any:
        print("------------")
        print(state)
        df = pd.read_csv(self.csv_file_path)
        df_row = df.iloc[state["index"]]
        paper_url = df_row["paper_url"]
        github_url = df_row["paper_code"]
        return {
            "paper_url": paper_url,
            "github_url": github_url,
        }


if __name__ == "__main__":
    csv_file_path = "/workspaces/researchgraph/data/optimization_algorithm.csv"
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "csvretriever",
        CSVNode(
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
