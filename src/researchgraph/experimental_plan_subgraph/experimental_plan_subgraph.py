import os
import time

from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.experimental_plan_subgraph.nodes.retrieve_code_with_devin import (
    retrieve_code_with_devin,
)
from researchgraph.experimental_plan_subgraph.nodes.generate_advantage_criteria import (
    generate_advantage_criteria,
)
from researchgraph.experimental_plan_subgraph.nodes.generate_experiment_details import (
    generate_experiment_details,
)
from researchgraph.experimental_plan_subgraph.nodes.generate_experiment_code import (
    generate_experiment_code,
)
from researchgraph.executor_subgraph.nodes.check_devin_completion import (
    check_devin_completion,
)

from researchgraph.experimental_plan_subgraph.input_data import (
    experimental_subgraph_input_data,
)

API_KEY = os.getenv("DEVIN_API_KEY")


class ExperimentalPlanSubgraphInputState(TypedDict):
    new_method: str
    base_github_url: str
    base_method_text: str


class ExperimentalPlanSubgraphHiddenState(TypedDict):
    retrieve_session_id: str
    retrieve_devin_url: str
    experiment_info_of_source_research: str


class ExperimentalPlanSubgraphOutputState(TypedDict):
    verification_policy: str
    experiment_details: str
    experiment_code: str


class ExperimentalPlanSubgraphState(
    ExperimentalPlanSubgraphInputState,
    ExperimentalPlanSubgraphHiddenState,
    ExperimentalPlanSubgraphOutputState,
):
    pass


class ExperimentalPlanSubgraph:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

    def _retrieve_code_with_devin_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        print("---ExperimentalPlanSubgraph---")
        print("retrieve_code_with_devin_node")
        retrieve_session_id, retrieve_devin_url = retrieve_code_with_devin(
            headers=self.headers,
            github_url=state["base_github_url"],
            base_method_text=state["base_method_text"],
        )
        return {
            "retrieve_session_id": retrieve_session_id,
            "retrieve_devin_url": retrieve_devin_url,
        }

    def _check_devin_completion_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        print("check_devin_completion_node")
        time.sleep(120)
        devin_output_response = check_devin_completion(
            headers=self.headers,
            session_id=state["retrieve_session_id"],
        )
        if devin_output_response is None:
            experiment_info_of_source_research = ""
        else:
            experiment_info_of_source_research = devin_output_response[
                "structured_output"
            ].get("extracted_info", "")
        return {
            "experiment_info_of_source_research": experiment_info_of_source_research,
        }

    def _generate_advantage_criteria_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        print("generate_advantage_criteria_node")
        verification_policy = generate_advantage_criteria(
            model_name="o3-mini-2025-01-31",
            new_method=state["new_method"],
        )
        return {"verification_policy": verification_policy}

    def _generate_experiment_details_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        print("generate_experiment_details_node")
        experimet_details = generate_experiment_details(
            model_name="o3-mini-2025-01-31",
            verification_policy=state["verification_policy"],
            experiment_info_of_source_research=state[
                "experiment_info_of_source_research"
            ],
        )
        return {"experiment_details": experimet_details}

    def _generate_experiment_code_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        print("generate_experiment_code_node")
        experiment_code = generate_experiment_code(
            model_name="o3-mini-2025-01-31",
            experiment_details=state["experiment_details"],
            experiment_info_of_source_research=state[
                "experiment_info_of_source_research"
            ],
        )
        return {"experiment_code": experiment_code}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(ExperimentalPlanSubgraphState)
        # make nodes
        graph_builder.add_node(
            "retrieve_code_with_devin_node", self._retrieve_code_with_devin_node
        )
        graph_builder.add_node(
            "check_devin_completion_node", self._check_devin_completion_node
        )
        graph_builder.add_node(
            "generate_advantage_criteria_node", self._generate_advantage_criteria_node
        )
        graph_builder.add_node(
            "generate_experiment_details_node", self._generate_experiment_details_node
        )
        graph_builder.add_node(
            "generate_experiment_code_node", self._generate_experiment_code_node
        )

        # make edges
        graph_builder.add_edge(START, "generate_advantage_criteria_node")
        graph_builder.add_edge(START, "retrieve_code_with_devin_node")
        graph_builder.add_edge(
            "retrieve_code_with_devin_node", "check_devin_completion_node"
        )
        graph_builder.add_edge(
            ["generate_advantage_criteria_node", "check_devin_completion_node"],
            "generate_experiment_details_node",
        )
        graph_builder.add_edge(
            "generate_experiment_details_node", "generate_experiment_code_node"
        )
        graph_builder.add_edge("generate_experiment_code_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    graph = ExperimentalPlanSubgraph().build_graph()
    # output = graph.invoke(
    #     experimental_subgraph_input_data,
    # )
    # print(output)
    # graph_nodes = list(graph.nodes.keys())[1:]
    for event in graph.stream(experimental_subgraph_input_data, stream_mode="updates"):
        # print(node)
        node_name = list(event.keys())[0]
        print(node_name)
        print(event[node_name])
