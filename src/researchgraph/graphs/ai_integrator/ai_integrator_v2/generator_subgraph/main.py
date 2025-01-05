import os
from IPython.display import Image
from langgraph.graph import START, END, StateGraph
# from typing import TypedDict
from pydantic import BaseModel, Field
from researchgraph.graphs.ai_integrator.ai_integrator_v2.generator_subgraph.llmnode_prompt import (
    ai_integrator_v2_extractor_prompt,
    ai_integrator_v2_codeextractor_prompt,
    ai_integrator_v2_creator_prompt,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v2.generator_subgraph.input_data import generator_subgraph_input_data
from researchgraph.core.factory import NodeFactory


class GeneratorState(BaseModel):
    objective: str = Field(default="")
    method_template: str = Field(default="")
    base_method_text: str = Field(default="")
    base_method_code: str = Field(default="")
    llm_script: str = Field(default="")
    arxiv_url: str = Field(default="")
    github_url: str = Field(default="")
    folder_structure: str = Field(default="")
    github_file: str = Field(default="")
    add_method_code: str = Field(default="")
    paper_text: str = Field(default="")
    add_method_text: str = Field(default="")
    new_method_code: str = Field(default="")
    new_method_text: str = Field(default="")


class GeneratorSubgraph:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
        ai_integrator_v2_extractor_prompt: str,
        ai_integrator_v2_codeextractor_prompt: str,
        ai_integrator_v2_creator_prompt: str,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        self.ai_integrator_v2_extractor_prompt = ai_integrator_v2_extractor_prompt
        self.ai_integrator_v2_codeextractor_prompt = (
            ai_integrator_v2_codeextractor_prompt
        )
        self.ai_integrator_v2_creator_prompt = ai_integrator_v2_creator_prompt

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.graph_builder = StateGraph(GeneratorState)

        self.graph_builder.add_node(
            "githubretriever",
            NodeFactory.create_node(
                node_name="retrieve_github_repository_node",
                save_dir="/content/drive/MyDrive/AutoRes/ai_integrator/exec-test",
                input_key=["github_url"],
                output_key=["folder_structure", "github_file"],
            ),
        )
        self.graph_builder.add_node(
            "arxivretriever",
            NodeFactory.create_node(
                node_name="retrieve_arxiv_text_node",
                save_dir=self.save_dir,
                input_key=["arxiv_url"],
                output_key=["paper_text"],
            ),
        )
        self.graph_builder.add_node(
            "extractor",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["paper_text"],
                output_key=["add_method_text"],
                llm_name=self.llm_name,
                prompt_template=self.ai_integrator_v2_extractor_prompt,
            ),
        )
        self.graph_builder.add_node(
            "codeextractor",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["add_method_text", "folder_structure", "github_file"],
                output_key=["add_method_code"],
                llm_name=self.llm_name,
                prompt_template=self.ai_integrator_v2_codeextractor_prompt,
            ),
        )
        self.graph_builder.add_node(
            "creator",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=[
                    "objective",
                    "add_method_text",
                    "add_method_code",
                    "base_method_text",
                    "base_method_code",
                    "method_template",
                ],
                output_key=["new_method_text", "new_method_code"],
                llm_name=self.llm_name,
                prompt_template=self.ai_integrator_v2_creator_prompt,
            ),
        )
        # make edges
        self.graph_builder.add_edge(START, "arxivretriever")
        self.graph_builder.add_edge("arxivretriever", "githubretriever")
        self.graph_builder.add_edge("arxivretriever", "extractor")
        self.graph_builder.add_edge(["githubretriever", "extractor"], "codeextractor")
        self.graph_builder.add_edge("codeextractor", "creator")
        self.graph_builder.add_edge("creator", END)

        self.graph = self.graph_builder.compile()

    def __call__(self, state: GeneratorState) -> dict:
        result = self.graph.invoke(state)
        return result

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v2_generator_subgraph.png", "wb") as f:
            f.write(image.data)


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    save_dir = "/workspaces/researchgraph/data"
    generator_subgraph = GeneratorSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
        ai_integrator_v2_extractor_prompt=ai_integrator_v2_extractor_prompt,
        ai_integrator_v2_codeextractor_prompt=ai_integrator_v2_codeextractor_prompt,
        ai_integrator_v2_creator_prompt=ai_integrator_v2_creator_prompt,
    )
    
    # generator_subgraph(
    #     state = generator_subgraph_input_data, 
    #     )

    image_dir = "/workspaces/researchgraph/images/"
    generator_subgraph.make_image(image_dir)
