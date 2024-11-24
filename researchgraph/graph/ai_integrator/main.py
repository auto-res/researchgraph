# %%
import os
import logging
from IPython.display import Image
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

from researchgraph.llmnode import LLMNode
from researchgraph.retrievenode import RetrieveCSVNode
from researchgraph.retrievenode import RetrievearXivTextNode
from researchgraph.retrievenode import GithubNode

from researchgraph.graph.ai_integrator.llmnode_setting.extractor import (
    extractor_setting,
)
from researchgraph.graph.ai_integrator.llmnode_setting.codeextractor import (
    codeextractor_setting,
)
from researchgraph.graph.ai_integrator.llmnode_setting.creator import creator_setting

from researchgraph.graph.ai_integrator.config import state

logger = logging.getLogger("researchgraph")


class State(TypedDict):
    environment: str
    objective: str
    base_method_text: str
    base_method_code: str
    llm_script: str
    index: int
    arxiv_url: str
    github_url: str
    folder_structure: str
    github_file: str
    add_method_code: str
    paper_text: str
    add_method_text: str
    new_method_code: list
    new_method_text: list


class AIIntegrator:
    def __init__(self, llm_name: str, save_dir: str):
        self.llm_name = llm_name
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.graph_builder = StateGraph(State)

        # make nodes
        self.graph_builder.add_node(
            "csvretriever",
            RetrieveCSVNode(
                input_variable="index",
                output_variable=["arxiv_url", "github_url"],
                csv_file_path="/workspaces/researchgraph/data/optimization_algorithm.csv",
            ),
        )
        self.graph_builder.add_node(
            "githubretriever",
            GithubNode(
                save_dir=self.save_dir,
                input_variable="github_url",
                output_variable=["folder_structure", "github_file"],
            ),
        )
        self.graph_builder.add_node(
            "arxivretriever",
            RetrievearXivTextNode(
                save_dir=self.save_dir,
                input_variable="arxiv_url",
                output_variable="paper_text",
            ),
        )
        self.graph_builder.add_node(
            "extractor", LLMNode(llm_name=llm_name, setting=extractor_setting)
        )
        self.graph_builder.add_node(
            "codeextractor", LLMNode(llm_name=llm_name, setting=codeextractor_setting)
        )
        self.graph_builder.add_node(
            "creator", LLMNode(llm_name=llm_name, setting=creator_setting)
        )
        # make edges
        self.graph_builder.add_edge("csvretriever", "arxivretriever")
        self.graph_builder.add_edge("csvretriever", "githubretriever")
        self.graph_builder.add_edge("arxivretriever", "extractor")
        self.graph_builder.add_edge(["githubretriever", "extractor"], "codeextractor")
        self.graph_builder.add_edge("codeextractor", "creator")

        # make branches
        # graph_builder.add_conditional_edges("verifier1", branchcontroller1)

        # set entry and finish points
        self.graph_builder.set_entry_point("csvretriever")
        self.graph_builder.set_finish_point("creator")

    def __call__(self, state: State) -> dict:
        self.graph = self.graph_builder.compile()
        result = self.graph.invoke(state)
        return result

    def write_result(self, response: State):
        index = response["index"]
        arxiv_url = response["arxiv_url"]
        add_method_text = response["add_method_text"][0]
        add_method_code = response["add_method_code"][0]
        new_method_text = response["new_method_text"][0]
        new_method_code = response["new_method_code"][0]
        content = (
            f"---Arxiv URL 1---:\n{arxiv_url}\n\n"
            f"---Add Method Text---:\n{add_method_text}\n\n"
            f"---Add Method Code---:\n{add_method_code}\n\n"
            f"---New Method Text---:\n{new_method_text}\n\n"
            f"---New Method Code---:\n{new_method_code}\n\n"
        )
        with open(self.save_dir + f"ai_integrator_{index}.txt", "w") as f:
            f.write(content)
        return

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_graph.png", "wb") as f:
            f.write(image.data)

    def make_mermaid(self):
        print(self.graph.get_graph(xray=1).draw_mermaid())
        return


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    save_dir = "/workspaces/researchgraph/data/exec_ai_integrator"
    research_graph = AIIntegrator(llm_name, save_dir)

    # visualize the graph
    # image_dir = "/workspaces/researchgraph/images/"
    # research_graph.make_image(image_dir)
    # research_graph.make_mermaid()

    result = research_graph(state)
    research_graph.write_result(result)