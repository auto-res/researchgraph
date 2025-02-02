from IPython.display import Image
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph

from researchgraph.graphs.ai_integrator.ai_integrator_v3.executor_subgraph.input_data import (
    executor_subgraph_input_data,
)
from researchgraph.core.factory import NodeFactory


class ExecutorState(BaseModel):
    new_method_text: str = Field(default="")
    new_method_code: str = Field(default="")
    branch_name: str = Field(default="")
    github_owner: str = Field(default="")
    repository_name: str = Field(default="")
    branch_name: str = Field(default="")
    workflow_run_id: int = Field(default=0)
    save_dir: str = Field(default="")
    fix_iterations: int = Field(default=1)
    output_file_path: str = Field(default="")
    error_file_path: str = Field(default="")
    session_id: str = Field(default="")
    output_file_path: str = Field(default="")
    error_file_path: str = Field(default="")
    devin_url: str = Field(default="")


class ExecutorSubgraph:
    def __init__(
        self,
        max_fix_iteration: int = 3,
    ):
        self.max_fix_iteration = max_fix_iteration
        self.graph_builder = StateGraph(ExecutorState)

        self.graph_builder.add_node(
            "generate_code_with_devin_node",
            NodeFactory.create_node(
                node_name="generate_code_with_devin_node",
                input_key=[
                    "github_owner",
                    "repository_name",
                    "new_method_text",
                    "new_method_code",
                ],
                output_key=["session_id", "branch_name", "devin_url"],
            ),
        )
        self.graph_builder.add_node(
            "execute_github_actions_workflow_node",
            NodeFactory.create_node(
                node_name="execute_github_actions_workflow_node",
                input_key=["github_owner", "repository_name", "branch_name"],
                output_key=["workflow_run_id"],
            ),
        )
        self.graph_builder.add_node(
            "retrieve_github_actions_artifacts_node",
            NodeFactory.create_node(
                node_name="retrieve_github_actions_artifacts_node",
                input_key=[
                    "github_owner",
                    "repository_name",
                    "workflow_run_id",
                    "save_dir",
                    "fix_iterations",
                ],
                output_key=["output_file_path", "error_file_path"],
            ),
        )
        self.graph_builder.add_node(
            "fix_code_with_devin_node",
            NodeFactory.create_node(
                node_name="fix_code_with_devin_node",
                input_key=[
                    "session_id",
                    "output_file_path",
                    "error_file_path",
                    "fix_iterations",
                ],
                output_key=["fix_iterations"],
            ),
        )

        # make edges
        self.graph_builder.add_edge(START, "generate_code_with_devin_node")
        self.graph_builder.add_edge(
            "generate_code_with_devin_node", "execute_github_actions_workflow_node"
        )
        self.graph_builder.add_edge(
            "execute_github_actions_workflow_node",
            "retrieve_github_actions_artifacts_node",
        )
        self.graph_builder.add_conditional_edges(
            "retrieve_github_actions_artifacts_node",
            self.iteration_function,
            {
                "correction": "fix_code_with_devin_node",
                "finish": END,
            },
        )
        self.graph_builder.add_edge(
            "fix_code_with_devin_node", "execute_github_actions_workflow_node"
        )

        self.graph = self.graph_builder.compile()

    def iteration_function(self, state: ExecutorState):
        if state.fix_iterations <= self.max_fix_iteration:
            return "correction"
        else:
            return "finish"

    def __call__(self, state: ExecutorState) -> dict:
        result = self.graph.invoke(state, debug=True)
        return result

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v3_executor_subgraph.png", "wb") as f:
            f.write(image.data)


if __name__ == "__main__":
    executor_subgraph = ExecutorSubgraph()

    executor_subgraph(
        state=executor_subgraph_input_data,
    )

    # image_dir = "/workspaces/researchgraph/images/"
    # executor_subgraph.make_image(image_dir)
