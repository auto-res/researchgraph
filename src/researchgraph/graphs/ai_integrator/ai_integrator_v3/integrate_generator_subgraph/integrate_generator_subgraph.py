from IPython.display import Image
from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from researchgraph.graphs.ai_integrator.ai_integrator_v3.integrate_generator_subgraph.llmnode_prompt import (
    # ai_integrator_v3_extractor_prompt,
    ai_integrator_v3_creator_prompt,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.integrate_generator_subgraph.input_data import (
    generator_subgraph_input_data,
)
# from researchgraph.core.factory import NodeFactory

from researchgraph.nodes.retrievenode.github.retrieve_code_with_devin import (
    RetrieveCodeWithDevinNode,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.integrate_generator_subgraph.nodes.llm_integrate_node import (
    execute_llm,
)


class IntegrateGeneratorState(TypedDict):
    objective: str
    base_github_url: str
    base_method_code: str
    base_method_text: str
    add_github_url: str
    add_method_code: str
    add_method_text: str
    new_method_code: str
    new_method_text: str


class IntegrateGeneratorSubgraph:
    def __init__(
        self,
        llm_name: str,
        ai_integrator_v3_creator_prompt: str,
    ):
        self.llm_name = llm_name
        self.ai_integrator_v3_creator_prompt = ai_integrator_v3_creator_prompt

    def _retrieve_code_with_devin_1(self, state: IntegrateGeneratorState) -> dict:
        add_github_url = state["add_github_url"]
        add_method_text = state["add_method_text"]
        extracted_code = RetrieveCodeWithDevinNode().execute(
            add_github_url, add_method_text
        )
        return {"add_method_code": extracted_code}

    def _retrieve_code_with_devin_2(self, state: IntegrateGeneratorState) -> dict:
        base_github_url = state["base_github_url"]
        base_method_text = state["base_method_text"]
        extracted_code = RetrieveCodeWithDevinNode().execute(
            base_github_url, base_method_text
        )
        return {"base_method_code": extracted_code}

    def _creator_node(self, state: IntegrateGeneratorState) -> dict:
        objective = state["objective"]
        base_method_text = state["base_method_text"]
        base_method_code = state["base_method_code"]
        add_method_text = state["add_method_text"]
        add_method_code = state["add_method_code"]
        new_method_text, new_method_code = execute_llm(
            llm_name=self.llm_name,
            prompt_template=self.ai_integrator_v3_creator_prompt,
            objective=objective,
            base_method_code=base_method_code,
            base_method_text=base_method_text,
            add_method_code=add_method_code,
            add_method_text=add_method_text,
        )
        return {"new_method_text": new_method_text, "new_method_code": new_method_code}

    def build_graph(self):
        self.graph_builder = StateGraph(IntegrateGeneratorState)
        # make nodes
        self.graph_builder.add_node(
            "githubretriever_1", self._retrieve_code_with_devin_1
        )
        self.graph_builder.add_node(
            "githubretriever_2", self._retrieve_code_with_devin_2
        )
        self.graph_builder.add_node("creator", self._creator_node)
        # make edges
        self.graph_builder.add_edge(START, "githubretriever_1")
        self.graph_builder.add_edge(START, "githubretriever_2")
        self.graph_builder.add_edge(
            ["githubretriever_1", "githubretriever_2"], "creator"
        )
        self.graph_builder.add_edge("creator", END)

        return self.graph_builder.compile()

    def __call__(self, state: IntegrateGeneratorState) -> dict:
        self.graph = self.build_graph()
        return self.graph.invoke(state, debug=True)

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(
            path + "ai_integrator_v3_integrate_generator_subgraph.png", "wb"
        ) as f:
            f.write(image.data)


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    generator_subgraph = IntegrateGeneratorSubgraph(
        llm_name=llm_name,
        ai_integrator_v3_creator_prompt=ai_integrator_v3_creator_prompt,
    )

    result = generator_subgraph(
        state=generator_subgraph_input_data,
    )

    print(result["new_method_text"])
    print(result["new_method_code"])

    # image_dir = "/workspaces/researchgraph/images/"
    # generator_subgraph.make_image(image_dir)
