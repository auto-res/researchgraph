from IPython.display import Image
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from researchgraph.graphs.ai_integrator.ai_integrator_v3.refiner_subgraph.llmnode_prompt import (
    ai_integrator_v3_llmcreator_prompt,
    ai_integrator_v3_llmcoder_prompt,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.refiner_subgraph.input_data import refiner_subgraph_input_data
from researchgraph.core.factory import NodeFactory


class RefinerState(BaseModel):
    base_method_text: str = Field(default="")
    base_method_code: str = Field(default="")
    num_ideas: int = Field(default="1")
    generated_ideas: str = Field(default="")
    refined_method_code: str = Field(default="")
    refined_method_text: str = Field(default="")


class RefinerSubgraph:
    def __init__(
        self,
        llm_name: str,
        ai_integrator_v3_llmcreator_prompt: str,
        ai_integrator_v3_llmcoder_prompt: str,
    ):
        self.llm_name = llm_name
        self.ai_integrator_v3_llmcreator_prompt = ai_integrator_v3_llmcreator_prompt
        self.ai_integrator_v3_llmcoder_prompt = ai_integrator_v3_llmcoder_prompt

        self.graph_builder = StateGraph(RefinerState)

        self.graph_builder.add_node(
            "llmcreator",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["base_method_text", "base_method_code", "num_ideas"],
                output_key=["generated_ideas"],
                llm_name=self.llm_name, 
                prompt_template=self.ai_integrator_v3_llmcreator_prompt, 
            ),
        )
        self.graph_builder.add_node(
            "llmcoder",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["base_method_text", "base_method_code", "generated_ideas"],
                output_key=["refined_method_text", "refined_method_code"],
                llm_name=self.llm_name,
                prompt_template=self.ai_integrator_v3_llmcoder_prompt,
            ),
        )
        # make edges
        self.graph_builder.add_edge(START, "llmcreator")
        self.graph_builder.add_edge("llmcreator", "llmcoder")
        self.graph_builder.add_edge("llmcoder", END)

        self.graph = self.graph_builder.compile()

    def __call__(self, state: RefinerState) -> dict:
        result = self.graph.invoke(state, debug=True)
        return result

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v3_refiner_subgraph.png", "wb") as f:
            f.write(image.data)

if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    refiner_subgraph = RefinerSubgraph(
        llm_name=llm_name,
        ai_integrator_v3_llmcreator_prompt=ai_integrator_v3_llmcreator_prompt,
        ai_integrator_v3_llmcoder_prompt=ai_integrator_v3_llmcoder_prompt,
    )
    
    refiner_subgraph(
        state = refiner_subgraph_input_data, 
        )

    image_dir = "/workspaces/researchgraph/images/"
    refiner_subgraph.make_image(image_dir)