from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph
from typing import TypedDict

from researchgraph.integrate_generator_subgraph.nodes.retrieve_code_with_devin import (
    RetrieveCodeWithDevinNode,
)

from researchgraph.integrate_generator_subgraph.nodes.method_integrate_node import (
    method_integrate,
    method_integrate_prompt,
)

from researchgraph.integrate_generator_subgraph.input_data import (
    integrate_generator_subgraph_input_data,
)


class IntegrateGeneratorState(TypedDict):
    objective: str
    base_github_url: str
    base_method_code: str
    base_method_text: str
    add_github_url: str
    add_method_code: str
    add_method_text: str

    new_detailed_description_of_methodology: str
    new_novelty: str
    new_experimental_procedure: str
    new_method_code: str


class IntegrateGeneratorSubgraph:
    def __init__(
        self,
        llm_name: str,
        # method_integrate_prompt: str,
    ):
        self.llm_name = llm_name
        # self.method_integrate_prompt = method_integrate_prompt

    def _retrieve_base_paper_code_with_devin(
        self, state: IntegrateGeneratorState
    ) -> dict:
        print("---IntegrateGeneratorSubgrap---")

        add_github_url = state["add_github_url"]
        add_method_text = state["add_method_text"]
        extracted_code = RetrieveCodeWithDevinNode().execute(
            add_github_url, add_method_text
        )
        return {"add_method_code": extracted_code}

    def _retrieve_add_paper_code_with_devin(
        self, state: IntegrateGeneratorState
    ) -> dict:
        base_github_url = state["base_github_url"]
        base_method_text = state["base_method_text"]
        extracted_code = RetrieveCodeWithDevinNode().execute(
            base_github_url, base_method_text
        )
        return {"base_method_code": extracted_code}

    def _method_integrate_node(self, state: IntegrateGeneratorState) -> dict:
        # objective = state["objective"]
        base_method_text = state["base_method_text"]
        base_method_code = state["base_method_code"]
        add_method_text = state["add_method_text"]
        add_method_code = state["add_method_code"]
        (
            new_method_detailed_description_of_methodology,
            new_method_novelty,
            new_method_experimental_procedure,
            new_method_code,
        ) = method_integrate(
            llm_name=self.llm_name,
            prompt_template=method_integrate_prompt,
            # objective=objective,
            base_method_code=base_method_code,
            base_method_text=base_method_text,
            add_method_code=add_method_code,
            add_method_text=add_method_text,
        )
        return {
            "new_detailed_description_of_methodology": new_method_detailed_description_of_methodology,
            "new_novelty": new_method_novelty,
            "new_experimental_procedure": new_method_experimental_procedure,
            "new_method_code": new_method_code,
        }

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(IntegrateGeneratorState)
        # make nodes
        graph_builder.add_node(
            "retrieve_base_paper_code_with_devin",
            self._retrieve_base_paper_code_with_devin,
        )
        graph_builder.add_node(
            "retrieve_add_paper_code_with_devin",
            self._retrieve_add_paper_code_with_devin,
        )
        graph_builder.add_node("method_integrate_node", self._method_integrate_node)
        # make edges
        graph_builder.add_edge(START, "retrieve_base_paper_code_with_devin")
        graph_builder.add_edge(START, "retrieve_add_paper_code_with_devin")
        graph_builder.add_edge(
            [
                "retrieve_base_paper_code_with_devin",
                "retrieve_add_paper_code_with_devin",
            ],
            "method_integrate_node",
        )
        graph_builder.add_edge("method_integrate_node", END)

        return graph_builder.compile()


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    subgraph = IntegrateGeneratorSubgraph(
        llm_name=llm_name,
        # method_integrate_prompt=method_integrate_prompt,
    ).build_graph()

    result = subgraph.invoke(integrate_generator_subgraph_input_data)
    print(result)
