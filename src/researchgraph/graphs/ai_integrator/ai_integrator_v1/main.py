import os
from IPython.display import Image
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

from researchgraph.core.factory import NodeFactory
from researchgraph.graphs.ai_integrator.ai_integrator_v1 import (
    ai_integrator_v1_extractor_prompt,
    ai_integrator_v1_codeextractor_prompt,
    ai_integrator_v1_creator_prompt,
)


class State(TypedDict):
    objective: str
    method_template: str
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
    script_save_path: str
    model_save_path: str
    result_save_path: str
    accuracy: str


class AIIntegratorv1:
    def __init__(
        self,
        llm_name: str,
        save_dir: str,
        new_method_file_name: str,
        ft_model_name: str,
        dataset_name: str,
        model_save_dir_name: str,
        result_save_file_name: str,
        answer_data_path: str,
        num_train_data: int | None = None,
        num_inference_data: int | None = None,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        self.new_method_file_name = new_method_file_name
        self.ft_model_name = ft_model_name
        self.dataset_name = dataset_name
        self.model_save_dir_name = model_save_dir_name
        self.result_save_file_name = result_save_file_name
        self.answer_data_path = answer_data_path
        self.num_train_data = num_train_data
        self.num_inference_data = num_inference_data

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.graph_builder = StateGraph(State)

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
                llm_name=llm_name,
                prompt_template=ai_integrator_v1_extractor_prompt,
            ),
        )
        self.graph_builder.add_node(
            "codeextractor",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["add_method_text", "folder_structure", "github_file"],
                output_key=["add_method_code"],
                llm_name=llm_name,
                prompt_template=ai_integrator_v1_codeextractor_prompt,
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
                llm_name=llm_name,
                prompt_template=ai_integrator_v1_creator_prompt,
            ),
        )
        self.graph_builder.add_node(
            "text2script",
            NodeFactory.create_node(
                node_name="text2script_node",
                input_key=["new_method_code"],
                output_key=["script_save_path"],
                save_file_path=os.path.join(self.save_dir, self.new_method_file_name),
            ),
        )
        self.graph_builder.add_node(
            "llmsfttrainer",
            NodeFactory.create_node(
                node_name="llmsfttrain_node",
                model_name=self.ft_model_name,
                dataset_name=self.dataset_name,
                num_train_data=self.num_train_data,
                model_save_path=os.path.join(self.save_dir, self.model_save_dir_name),
                lora=True,
                input_key=["script_save_path"],
                output_key=["model_save_path"],
            ),
        )
        self.graph_builder.add_node(
            "llminferencer",
            NodeFactory.create_node(
                node_name="llminference_node",
                input_key=["model_save_path"],
                output_key=["result_save_path"],
                dataset_name=self.dataset_name,
                num_inference_data=self.num_inference_data,
                result_save_path=os.path.join(
                    self.save_dir, self.result_save_file_name
                ),
            ),
        )
        self.graph_builder.add_node(
            "llmevaluater",
            NodeFactory.create_node(
                node_name="llmevaluate_node",
                input_key=["result_save_path"],
                output_key=["accuracy"],
                answer_data_path=self.answer_data_path,
            ),
        )

        # make edges
        self.graph_builder.add_edge("arxivretriever", "githubretriever")
        self.graph_builder.add_edge("arxivretriever", "extractor")
        self.graph_builder.add_edge(["githubretriever", "extractor"], "codeextractor")
        self.graph_builder.add_edge("codeextractor", "creator")
        self.graph_builder.add_edge("creator", "text2script")
        self.graph_builder.add_edge("text2script", "llmsfttrainer")
        self.graph_builder.add_edge("llmsfttrainer", "llminferencer")
        self.graph_builder.add_edge("llminferencer", "llmevaluater")

        # set entry and finish points
        self.graph_builder.set_entry_point("arxivretriever")
        self.graph_builder.set_finish_point("llmevaluater")

        self.graph = self.graph_builder.compile()

    def __call__(self, state: State) -> dict:
        result = self.graph.invoke(state, debug=True)
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
        with open(path + "ai_integrator_v1_graph.png", "wb") as f:
            f.write(image.data)
