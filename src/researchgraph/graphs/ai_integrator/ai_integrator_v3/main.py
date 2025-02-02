from langgraph.graph import START, END, StateGraph

# inputデータ
from researchgraph.graphs.ai_integrator.ai_integrator_v3.integrate_generator_subgraph.input_data import (
    generator_subgraph_input_data,
)


from researchgraph.graphs.ai_integrator.ai_integrator_v3.integrate_generator_subgraph.integrate_generator_subgraph import (
    IntegrateGeneratorSubgraph,
    IntegrateGeneratorState,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.integrate_generator_subgraph.llmnode_prompt import (
    #     ai_integrator_v3_extractor_prompt,
    #     ai_integrator_v3_codeextractor_prompt,
    ai_integrator_v3_creator_prompt,
)
# from researchgraph.graphs.ai_integrator.ai_integrator_v3.executor_subgraph.executor_subgraph import ExecutorSubgraph, ExecutorState
# from researchgraph.graphs.ai_integrator.ai_integrator_v3.executor_subgraph.llmnode_prompt import ai_integrator_v3_modifier_prompt


class AIIntegratorv3State(IntegrateGeneratorState):
    # Generator Subgraph
    # objective: str
    # base_github_url: str
    # base_method_code: str
    # base_method_text: str
    # add_github_url: str
    # add_method_code: str
    # add_method_text: str
    # new_method_code: str
    # new_method_text: str
    pass


class AIIntegratorv3:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
        ai_integrator_v3_creator_prompt: str,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        # Search Subgraph
        # Generator Subgraph
        self.ai_integrator_v3_creator_prompt = ai_integrator_v3_creator_prompt
        # Executor Subgraph
        # Witer Subgraph

    def build_graph(self):
        # Search Subgraph
        # Generator Subgraph
        generate_subgraph = IntegrateGeneratorSubgraph(
            llm_name=self.llm_name,
            ai_integrator_v3_creator_prompt=ai_integrator_v3_creator_prompt,
        )
        # Executor Subgraph
        # Witer Subgraph

        self.graph_builder = StateGraph(AIIntegratorv3State)
        # make nodes
        self.graph_builder.add_node("generator", generate_subgraph)
        # make edges
        self.graph_builder.add_edge(START, "generator")
        self.graph_builder.add_edge("generator", END)

        return self.graph_builder.compile()

    def __call__(self, state: AIIntegratorv3State) -> dict:
        self.graph = self.build_graph()
        return self.graph.invoke(state, debug=True)


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-11-20"
    save_dir = "/workspaces/researchgraph/data"

    ai_integrator_v3 = AIIntegratorv3(
        llm_name=llm_name,
        save_dir=save_dir,
        ai_integrator_v3_creator_prompt=ai_integrator_v3_creator_prompt,
    )
    result = ai_integrator_v3(generator_subgraph_input_data)
