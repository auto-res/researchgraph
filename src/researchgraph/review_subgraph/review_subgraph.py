from typing import TypedDict, Optional
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.review_subgraph.nodes.review_node import ReviewNode
from researchgraph.review_subgraph.input_data import review_subgraph_input_data


class ReviewState(TypedDict):
    # 執筆ノード以外の査読対象: Optional[str]
    tex_text: Optional[str]

    review_decision: bool
    review_feedback: str


class ReviewSubgraph:
    def __init__(
        self,
        llm_name: str, 
        save_dir: str,
        review_target: str,
        threshold: float,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        self.review_target = review_target
        self.threshold = threshold

    def _review_node(self, state: ReviewState) -> dict:
        review_content_mapping = {
        "retrieve_paper_subgraph": state.get("", ""),
        "generator_subgraph": state.get("", ""),
        "executor_subgraph": state.get("", ""),
        "writer_subgraph": state.get("tex_text", ""),
        }
        review_content = review_content_mapping[self.review_target]
        if not review_content:
            print("No review content found.")
            return {
                "review_decision": True,
                "review_feedback": "No review content found.",
            }
        print("---ReviewSubgraph---")
        review_decision, review_feedback = ReviewNode(
            llm_name=self.llm_name, 
            save_dir=self.save_dir,
            review_target=self.review_target,
            threshold=self.threshold,
        ).execute(review_content)
        return {
            "review_decision": review_decision, 
            "review_feedback": review_feedback,
        }
    
    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(ReviewState)
        # make nodes
        graph_builder.add_node("review_node", self._review_node)
        # make edges
        graph_builder.add_edge(START, "review_node")
        graph_builder.add_edge("review_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    # llm_name = "gpt-4o-mini-2024-07-18"
    save_dir = "/workspaces/researchgraph/data/review_log"
    review_target = "writer_subgraph"
    threshold = 3.5

    subgraph = ReviewSubgraph(
        llm_name, 
        save_dir,
        review_target,
        threshold = threshold,
    ).build_graph()
    result = subgraph.invoke(review_subgraph_input_data)