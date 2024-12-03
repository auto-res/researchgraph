# %%
import os
import logging
from IPython.display import Image
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

from researchgraph.llmnode import LLMNode
from researchgraph.retrievenode import RetrievearXivTextNode
from researchgraph.retrievenode import GithubNode
from researchgraph.writingnode import Text2ScriptNode
from researchgraph.experimentnode import (
    LLMTrainNode,
    LLMInferenceNode,
    LLMEvaluateNode,
)

from researchgraph.graphs.ai_integrator.ai_integrator_v1.llmnode_setting.extractor import (
    extractor_setting,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v1.llmnode_setting.codeextractor import (
    codeextractor_setting,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v1.llmnode_setting.creator import (
    creator_setting,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v1.config import (
    ai_integratorv1_setting,
)

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
            GithubNode(
                save_dir=self.save_dir,
                input_key="github_url",
                output_key=["folder_structure", "github_file"],
            ),
        )
        self.graph_builder.add_node(
            "arxivretriever",
            RetrievearXivTextNode(
                save_dir=self.save_dir,
                input_key="arxiv_url",
                output_key="paper_text",
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
        self.graph_builder.add_node(
            "text2script",
            Text2ScriptNode(
                input_key="new_method_code",
                output_key="script_save_path",
                save_file_path=os.path.join(self.save_dir, self.new_method_file_name),
            ),
        )
        self.graph_builder.add_node(
            "llmtrainer",
            LLMTrainNode(
                model_name=self.ft_model_name,
                dataset_name=self.dataset_name,
                num_train_data=self.num_train_data,
                model_save_path=os.path.join(self.save_dir, self.model_save_dir_name),
                input_key="script_save_path",
                output_key="model_save_path",
            ),
        )
        self.graph_builder.add_node(
            "llminferencer",
            LLMInferenceNode(
                input_key="model_save_path",
                output_key="result_save_path",
                dataset_name=self.dataset_name,
                num_inference_data=self.num_inference_data,
                result_save_path=os.path.join(
                    self.save_dir, self.result_save_file_name
                ),
            ),
        )
        self.graph_builder.add_node(
            "llmevaluater",
            LLMEvaluateNode(
                input_key="result_save_path",
                output_key="accuracy",
                answer_data_path=self.answer_data_path,
            ),
        )

        # make edges
        self.graph_builder.add_edge("arxivretriever", "githubretriever")
        self.graph_builder.add_edge("arxivretriever", "extractor")
        self.graph_builder.add_edge(["githubretriever", "extractor"], "codeextractor")
        self.graph_builder.add_edge("codeextractor", "creator")
        self.graph_builder.add_edge("creator", "text2script")
        self.graph_builder.add_edge("text2script", "llmtrainer")
        self.graph_builder.add_edge("llmtrainer", "llminferencer")
        self.graph_builder.add_edge("llminferencer", "llmevaluater")

        # set entry and finish points
        self.graph_builder.set_entry_point("arxivretriever")
        # self.graph_builder.set_finish_point("creator")
        self.graph_builder.set_finish_point("llmevaluater")

        self.graph = self.graph_builder.compile()

    # def __call__(self, state: State, debug: bool = True) -> dict:
    def __call__(self, state: State) -> dict:
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
    save_dir = "/content/drive/MyDrive/AutoRes/ai_integrator/exec-test"
    ft_model_name = "unsloth/Meta-Llama-3.1-8B"
    dataset_name = "openai/gsm8k"
    new_method_file_name = "new_method.py"
    model_save_dir_name = "train_model"
    result_save_file_name = "pred_file"
    answer_data_path = "/content/drive/MyDrive/AutoRes/ai_integrator/gsm8k_answer.csv"
    # num_train_data = 100
    # num_inference_data = 100

    research_graph = AIIntegratorv1(
        llm_name=llm_name,
        save_dir=save_dir,
        new_method_file_name=new_method_file_name,
        ft_model_name=ft_model_name,
        dataset_name=dataset_name,
        model_save_dir_name=model_save_dir_name,
        result_save_file_name=result_save_file_name,
        answer_data_path=answer_data_path,
        # num_train_data = num_train_data,
        # num_inference_data = num_inference_data,
    )

    # visualize the graph
    # image_dir = "/workspaces/researchgraph/images/"
    # research_graph.make_image(image_dir)
    # research_graph.make_mermaid()

    result = research_graph(
        state=ai_integratorv1_setting,
        # debug=True
    )
    # research_graph.write_result(result)
