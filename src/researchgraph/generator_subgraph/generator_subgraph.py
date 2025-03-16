from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from researchgraph.generator_subgraph.nodes.generate_new_method import (
    generate_new_method,
)
from researchgraph.generator_subgraph.nodes.generate_advantage_criteria import (
    generate_advantage_criteria,
)
from researchgraph.generator_subgraph.nodes.generate_experiment_details import (
    generate_experiment_details,
)
from researchgraph.generator_subgraph.nodes.generate_experiment_code import (
    generate_experiment_code,
)

from researchgraph.generator_subgraph.input_data import generator_subgraph_input_data


class GeneratorSubgraphInputState(TypedDict):
    new_method: str


class GeneratorSubgraphOutputState(TypedDict):
    verification_policy: str
    experiment_details: str
    experiment_code: str


class GeneratorSubgraphState(GeneratorSubgraphInputState, GeneratorSubgraphOutputState):
    pass


class GeneratorSubgraph:
    def __init__(self):
        pass

    def _generate_new_method_node(self, state: GeneratorSubgraphState) -> dict:
        print("---GeneratorSubgraph---")
        print("generate_new_method_node")
        new_method = generate_new_method(
            model_name="o3-mini-2025-01-31",
            new_method=state["new_method"],
        )
        return {"new_method": new_method}

    def _generate_advantage_criteria_node(self, state: GeneratorSubgraphState) -> dict:
        print("generate_advantage_criteria_node")
        verification_policy = generate_advantage_criteria(
            model_name="o3-mini-2025-01-31",
            new_method=state["new_method"],
        )
        return {"verification_policy": verification_policy}

    def _generate_experiment_details_node(self, state: GeneratorSubgraphState) -> dict:
        print("generate_experiment_details_node")
        experimet_details = generate_experiment_details(
            model_name="o3-mini-2025-01-31",
            verification_policy=state["verification_policy"],
        )
        return {"experiment_details": experimet_details}

    def _generate_experiment_code_node(self, state: GeneratorSubgraphState) -> dict:
        print("generate_experiment_code_node")
        experiment_code = generate_experiment_code(
            model_name="o3-mini-2025-01-31",
            experiment_details=state["experiment_details"],
        )
        return {"experiment_code": experiment_code}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(GeneratorSubgraphState)
        # make nodes
        # graph_builder.add_node("generate_new_method_node", self._generate_new_method_node)
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
    graph = GeneratorSubgraph().build_graph()
    # output = graph.invoke(
    #     generator_subgraph_input_data,
    # )
    # print(output)
    # graph_nodes = list(graph.nodes.keys())[1:]
    for event in graph.stream(generator_subgraph_input_data, stream_mode="updates"):
        # print(node)
        node_name = list(event.keys())[0]
        print(node_name)
        print(event[node_name])

    # pprint.pprint(output["verification_policy"])
    # pprint.pprint(output["experiment_details"])
    # pprint.pprint(output["experiment_code"])
