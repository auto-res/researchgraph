from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.experimental_plan_subgraph.nodes.generate_advantage_criteria import (
    generate_advantage_criteria,
)
from researchgraph.experimental_plan_subgraph.nodes.generate_experiment_details import (
    generate_experiment_details,
)
from researchgraph.experimental_plan_subgraph.nodes.generate_experiment_code import (
    generate_experiment_code,
)

from researchgraph.experimental_plan_subgraph.input_data import (
    experimental_subgraph_input_data,
)


class ExperimentalPlanSubgraphInputState(TypedDict):
    new_method: str


class ExperimentalPlanSubgraphOutputState(TypedDict):
    verification_policy: str
    experiment_details: str
    experiment_code: str


class ExperimentalPlanSubgraphState(
    ExperimentalPlanSubgraphInputState, ExperimentalPlanSubgraphOutputState
):
    pass


class ExperimentalPlanSubgraph:
    def __init__(self):
        pass

    def _generate_advantage_criteria_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        print("---ExperimentalPlanSubgraph---")
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
        )
        return {"experiment_details": experimet_details}

    def _generate_experiment_code_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        print("generate_experiment_code_node")
        experiment_code = generate_experiment_code(
            model_name="o3-mini-2025-01-31",
            experiment_details=state["experiment_details"],
        )
        return {"experiment_code": experiment_code}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(ExperimentalPlanSubgraphState)
        # make nodes
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
        graph_builder.add_edge(
            "generate_advantage_criteria_node", "generate_experiment_details_node"
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
