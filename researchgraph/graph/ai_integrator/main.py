# %%
import os
import logging
from datetime import datetime
from IPython.display import Image
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

from researchgraph.llmnode import LLMNode
from researchgraph.retrievenode import RetrieveCSVNode
from researchgraph.retrievenode import RetrievearXivTextNode
from researchgraph.retrievenode import GithubNode
# from researchgraph.evaluatenode import LLMEvaluateNode

from researchgraph.graph.ai_integrator.llmnode_setting.extractor import (
    extractor1_setting,
    extractor2_setting,
)
from researchgraph.graph.ai_integrator.llmnode_setting.codeextractor import (
    codeextractor1_setting,
    codeextractor2_setting,
)
from researchgraph.graph.ai_integrator.llmnode_setting.creator import creator_setting
from researchgraph.graph.ai_integrator.llmnode_setting.verifier import (
    verifier1_setting,
    verifier2_setting,
)
from researchgraph.graph.ai_integrator.llmnode_setting.coder import (
    coder1_setting,
    coder2_setting,
)
# from researchgraph.graph.ai_integrator.llmnode_setting.debugger import (
#     debugger1_setting,
#     debugger2_setting,
# )
# from researchgraph.graph.ai_integrator.llmnode_setting.comparator import (
#     comparator_setting,
# )


from researchgraph.graph.ai_integrator.branch import (
    branchcontroller1,
    branchcontroller3,
)

# ログディレクトリの定義と作成
log_dir = "/workspaces/researchgraph/logs/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# ログファイルの設定
log_filename = datetime.now().strftime("app_%Y-%m-%d_%H-%M-%S.log")
file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
file_handler.setFormatter(formatter)

# ロガーの設定
logger = logging.getLogger("researchgraph")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(file_handler)


class State(TypedDict):
    environment: str
    objective: str
    llm_script: str
    index_1: int
    index_2: int
    arxiv_url_1: str
    arxiv_url_2: str
    github_url_1: str
    github_url_2: str
    folder_structure_1: str
    folder_structure_2: str
    github_file_1: str
    github_file_2: str
    method_1_code: str
    method_2_code: str
    paper_text_1: str
    paper_text_2: str
    method_1_text: str
    method_2_text: str
    new_method_code: list
    new_method_text: list
    method_1_executable: str
    new_method_executable: str
    method_1_experimental_code: list
    new_method_experimental_code: list
    # method_1_code_error: str
    # new_method_code_error: str
    # method_1_score: str
    # new_method_score: str
    # method_1_completion: str
    # new_method_completion: str
    # comparison_result: str
    # comparison_result_content: str


class AIIntegrator:
    def __init__(self, llm_name, save_dir):
        self.llm_name = llm_name
        self.save_dir = save_dir
        graph_builder = StateGraph(State)

        # make nodes
        graph_builder.add_node(
            "csvretriever1",
            RetrieveCSVNode(
                input_variable="index_1",
                output_variable=["arxiv_url_1", "github_url_1"],
                csv_file_path="/workspaces/researchgraph/data/optimization_algorithm.csv",
            ),
        )
        graph_builder.add_node(
            "csvretriever2",
            RetrieveCSVNode(
                input_variable="index_2",
                output_variable=["arxiv_url_2", "github_url_2"],
                csv_file_path="/workspaces/researchgraph/data/optimization_algorithm.csv",
            ),
        )
        graph_builder.add_node(
            "githubretriever1",
            GithubNode(
                save_dir=self.save_dir,
                input_variable="github_url_1",
                output_variable=["folder_structure_1", "github_file_1"],
            ),
        )
        graph_builder.add_node(
            "githubretriever2",
            GithubNode(
                save_dir=self.save_dir,
                input_variable="github_url_2",
                output_variable=["folder_structure_2", "github_file_2"],
            ),
        )
        graph_builder.add_node(
            "arxivretriever1",
            RetrievearXivTextNode(
                save_dir=self.save_dir,
                input_variable="arxiv_url_1",
                output_variable="paper_text_1",
            ),
        )
        graph_builder.add_node(
            "arxivretriever2",
            RetrievearXivTextNode(
                save_dir=self.save_dir,
                input_variable="arxiv_url_2",
                output_variable="paper_text_2",
            ),
        )
        graph_builder.add_node(
            "extractor1", LLMNode(llm_name=llm_name, setting=extractor1_setting)
        )
        graph_builder.add_node(
            "extractor2", LLMNode(llm_name=llm_name, setting=extractor2_setting)
        )
        graph_builder.add_node(
            "codeextractor1", LLMNode(llm_name=llm_name, setting=codeextractor1_setting)
        )
        graph_builder.add_node(
            "codeextractor2", LLMNode(llm_name=llm_name, setting=codeextractor2_setting)
        )
        graph_builder.add_node(
            "creator", LLMNode(llm_name=llm_name, setting=creator_setting)
        )
        graph_builder.add_node(
            "verifier1", LLMNode(llm_name=llm_name, setting=verifier1_setting)
        )
        graph_builder.add_node(
            "verifier2", LLMNode(llm_name=llm_name, setting=verifier2_setting)
        )
        graph_builder.add_node(
            "coder1", LLMNode(llm_name=llm_name, setting=coder1_setting)
        )
        graph_builder.add_node(
            "coder2", LLMNode(llm_name=llm_name, setting=coder2_setting)
        )
        # graph_builder.add_node("evaluator1", LLMEvaluateNode)
        # graph_builder.add_node("evaluator2", LLMEvaluateNode)
        # graph_builder.add_node(
        #     "debugger1", LLMNode(llm_name=llm_name, setting=debugger1_setting)
        # )
        # graph_builder.add_node(
        #     "debugger2", LLMNode(llm_name=llm_name, setting=debugger2_setting)
        # )
        # graph_builder.add_node(
        #     "comparator", LLMNode(llm_name=llm_name, setting=comparator_setting)
        # )

        # make edges
        graph_builder.add_edge("csvretriever1", "csvretriever2")
        graph_builder.add_edge("csvretriever1", "arxivretriever1")
        graph_builder.add_edge("csvretriever1", "githubretriever1")
        graph_builder.add_edge("csvretriever2", "arxivretriever2")
        graph_builder.add_edge("csvretriever2", "githubretriever2")
        graph_builder.add_edge("arxivretriever1", "extractor1")
        graph_builder.add_edge("arxivretriever2", "extractor2")
        graph_builder.add_edge(["githubretriever1", "extractor1"], "codeextractor1")
        graph_builder.add_edge(["githubretriever2", "extractor2"], "codeextractor2")
        graph_builder.add_edge(["codeextractor1", "codeextractor2"], "creator")

        graph_builder.add_edge("codeextractor1", "verifier1")
        # graph_builder.add_edge("coder1", "evaluator1")
        # graph_builder.add_edge("debugger1", "evaluator1")
        graph_builder.add_edge("creator", "verifier2")
        graph_builder.add_edge("coder1", "coder2")
        # graph_builder.add_edge("coder2", "evaluator2")
        # graph_builder.add_edge("debugger2", "evaluator2")
        # graph_builder.add_edge("verifier1", "comparator")

        # make branches
        graph_builder.add_conditional_edges("verifier1", branchcontroller1)
        # graph_builder.add_conditional_edges("evaluator1", branchcontroller2)
        graph_builder.add_conditional_edges("verifier2", branchcontroller3)
        # graph_builder.add_conditional_edges("evaluator2", branchcontroller4)

        # set entry and finish points
        graph_builder.set_entry_point("csvretriever1")
        # graph_builder.set_finish_point("comparator")
        graph_builder.set_finish_point("coder2")

        self.graph = graph_builder.compile()

    def __call__(self, memory):
        result = self.graph.invoke(memory)
        return result

    def write_result(self, response):
        arxiv_url_1 = response["arxiv_url_1"]
        arxiv_url_2 = response["arxiv_url_2"]
        method_1_text = response["method_1_text"][0]
        method_1_code = response["method_1_code"][0]
        method_2_text = response["method_2_text"][0]
        method_2_code = response["method_2_code"][0]
        new_method_text = response["new_method_text"][0]
        new_method_code = response["new_method_code"][0]
        method_1_executable = response["method_1_executable"]
        new_method_executable = response["new_method_executable"]
        if method_1_executable == "True":
            method_1_experimental_code = response["method_1_experimental_code"]
        else:
            method_1_experimental_code = ""
        if new_method_executable == "True":
            new_method_experimental_code = response["new_method_experimental_code"]
        else:
            new_method_experimental_code = ""

        content = (
            f"---Arxiv URL 1---:\n{arxiv_url_1}\n\n"
            f"---Arxiv URL 2---:\n{arxiv_url_2}\n\n"
            f"---Method 1 Text---:\n{method_1_text}\n\n"
            f"---Method 1 Code---:\n{method_1_code}\n\n"
            f"---Method 2 Text---:\n{method_2_text}\n\n"
            f"---Method 2 Code---:\n{method_2_code}\n\n"
            f"---New Method Text---:\n{new_method_text}\n\n"
            f"---New Method Code---:\n{new_method_code}\n\n"
            f"---Method 1 Executable---:\n{method_1_executable}\n\n"
            f"---New Method Executable---:\n{new_method_executable}\n\n"
            f"---Method 1 Experimental Code---:\n{method_1_experimental_code}\n\n"
            f"---New Method Experimental Code---:\n{new_method_experimental_code}\n\n"
        )
        with open(self.save_dir + "ai_integrator_response.txt", "w") as f:
            f.write(content)
        return

    def visualize(self, path):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_graph.png", "wb") as f:
            f.write(image.data)


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    save_dir = "/workspaces/researchgraph/data/exec_ai_integrator"
    research_graph = AIIntegrator(llm_name, save_dir)

    # visualize the graph
    # image_dir = "/workspaces/researchgraph/images/"
    # research_graph.visualize(image_dir)

    # execute the graph
    # state = {
    #     "environment": """
    #     The following two experimental environments are available
    #     ・Fine tuning of the LLM and experiments with rewriting the Optimizer or loss function.
    #     ・Verification of the accuracy of prompt engineering.
    #     """,
    #     "objective": """
    #     Batch Size Grokking: Assessing the impact of the training batchsize on the grokking phenomenon. Modify the experiments to dynamically adjust the batch size during training, starting with a small batch size and gradually increasing it. This could potentially lead to faster generalization on the validation set.
    #     """,
    # }
    with open("run_clm.py", "r", encoding="utf-8") as f:
        script_content = f.read()

    state = {
        "objective": "I am researching Optimizers for fine-tuning LLM. The aim is to find a better Optimizer.",
        "index_1": 8,
        "index_2": 10,
        "llm_script": script_content,
    }
    result = research_graph(state)
    research_graph.write_result(result)
