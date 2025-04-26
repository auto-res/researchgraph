import os
import time
import logging

from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph

from airas.utils.logging_utils import setup_logging

from airas.experimental_plan_subgraph.nodes.retrieve_code_with_devin import (
    retrieve_code_with_devin,
)
from airas.experimental_plan_subgraph.nodes.generate_advantage_criteria import (
    generate_advantage_criteria,
)
from airas.experimental_plan_subgraph.nodes.generate_experiment_details import (
    generate_experiment_details,
)
from airas.experimental_plan_subgraph.nodes.generate_experiment_code import (
    generate_experiment_code,
)
from airas.executor_subgraph.nodes.check_devin_completion import (
    check_devin_completion,
)

from airas.utils.execution_timers import time_node, ExecutionTimeState
from airas.utils.github_utils.graph_wrapper import create_wrapped_subgraph

setup_logging()
logger = logging.getLogger(__name__)

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
    ExecutionTimeState,
):
    pass


class ExperimentalPlanSubgraph:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

    @time_node("experimental_plan_subgraph", "_retrieve_code_with_devin_node")
    def _retrieve_code_with_devin_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        retrieve_session_id, retrieve_devin_url = retrieve_code_with_devin(
            headers=self.headers,
            github_url=state["base_github_url"],
            base_method_text=state["base_method_text"],
        )
        return {
            "retrieve_session_id": retrieve_session_id,
            "retrieve_devin_url": retrieve_devin_url,
        }

    @time_node("experimental_plan_subgraph", "_check_devin_completion_node")
    def _check_devin_completion_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        time.sleep(120)
        devin_output_response = check_devin_completion(
            headers=self.headers,
            session_id=state["retrieve_session_id"],
        )
        if devin_output_response is None:
            experiment_info_of_source_research = ""
        else:
            structured_output = devin_output_response.get("structured_output")
            if structured_output is None:
                logger.warning(
                    "Devin output response does not contain `structured_output`. "
                    f"Full response: {devin_output_response}"
                )
                experiment_info_of_source_research = ""
            else:
                experiment_info_of_source_research = structured_output.get(
                    "extracted_info", ""
                )
        return {
            "experiment_info_of_source_research": experiment_info_of_source_research,
        }

    @time_node("experimental_plan_subgraph", "_generate_advantage_criteria_node")
    def _generate_advantage_criteria_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        verification_policy = generate_advantage_criteria(
            llm_name="o3-mini-2025-01-31",
            new_method=state["new_method"],
        )
        return {"verification_policy": verification_policy}

    @time_node("experimental_plan_subgraph", "_generate_experiment_details_node")
    def _generate_experiment_details_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        experimet_details = generate_experiment_details(
            llm_name="o3-mini-2025-01-31",
            verification_policy=state["verification_policy"],
            experiment_info_of_source_research=state[
                "experiment_info_of_source_research"
            ],
        )
        return {"experiment_details": experimet_details}

    @time_node("experimental_plan_subgraph", "_generate_experiment_code_node")
    def _generate_experiment_code_node(
        self, state: ExperimentalPlanSubgraphState
    ) -> dict:
        experiment_code = generate_experiment_code(
            llm_name="o3-mini-2025-01-31",
            experiment_details=state["experiment_details"],
            experiment_info_of_source_research=state[
                "experiment_info_of_source_research"
            ],
        )
        return {"experiment_code": experiment_code}

    def branch_function(self, state: ExperimentalPlanSubgraphState) -> str:
        if state["base_github_url"] == "":
            return "skip"
        else:
            return "github_code_retrieval"

    def __relay_node(self, state: ExperimentalPlanSubgraphState) -> dict:
        if "experiment_info_of_source_research" in state:
            return {
                "experiment_info_of_source_research": state[
                    "experiment_info_of_source_research"
                ],
            }
        else:
            return {
                "experiment_info_of_source_research": "",
            }

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
        graph_builder.add_node("relay_node", self.__relay_node)

        # make edges
        graph_builder.add_edge(START, "generate_advantage_criteria_node")
        graph_builder.add_conditional_edges(
            START,
            path=self.branch_function,
            path_map={
                "skip": "relay_node",
                "github_code_retrieval": "retrieve_code_with_devin_node",
            },
        )
        graph_builder.add_edge(
            "retrieve_code_with_devin_node", "check_devin_completion_node"
        )
        graph_builder.add_edge("check_devin_completion_node", "relay_node")
        graph_builder.add_edge(
            ["generate_advantage_criteria_node", "relay_node"],
            "generate_experiment_details_node",
        )
        graph_builder.add_edge(
            "generate_experiment_details_node", "generate_experiment_code_node"
        )
        graph_builder.add_edge("generate_experiment_code_node", END)

        return graph_builder.compile()


ExperimentalPlaner = create_wrapped_subgraph(
    ExperimentalPlanSubgraph,
    ExperimentalPlanSubgraphInputState,
    ExperimentalPlanSubgraphOutputState,
)

if __name__ == "__main__":
    github_repository = "auto-res2/test20"
    branch_name = "test"

    experimentalplaner = ExperimentalPlaner(
        github_repository=github_repository,
        branch_name=branch_name,
    )

    result = experimentalplaner.run()
    print(f"result: {result}")
