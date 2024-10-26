# %%
from IPython.display import Image
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

from researchgraph.llmnode import LLMNode
from researchgraph.retrievenode import OpenAlexNode
from researchgraph.retrievenode import GithubNode
from researchgraph.evaluatenode import LLMEvaluateNode


from researchgraph.graph.ai_integrator.llmnode_setting.keyworder import (
    keyworder1_setting,
    keyworder2_setting,
)
from researchgraph.graph.ai_integrator.llmnode_setting.selector import (
    selector1_setting,
    selector2_setting,
)
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
from researchgraph.graph.ai_integrator.llmnode_setting.debugger import (
    debugger1_setting,
    debugger2_setting,
)
from researchgraph.graph.ai_integrator.llmnode_setting.comparator import (
    comparator_setting,
)


from researchgraph.graph.ai_integrator.branch import (
    branchcontroller1,
    branchcontroller2,
    branchcontroller3,
    branchcontroller4,
)


class State(TypedDict):
    environment: str
    objective: str
    keywords_mid_thought_1: str
    keywords_mid_thought_2: str
    keywords_1: str
    keywords_2: str
    collection_of_papers_1: list
    collection_of_papers_2: list
    selected_paper_1: str
    selected_paper_2: str
    github_url_1: str
    github_url_2: str
    folder_structure_1: str
    folder_structure_2: str
    github_file_1: str
    github_file_2: str
    method_1_code: str
    method_2_code: str
    method_1_text: str
    method_2_text: str
    new_method_code: list
    new_method_text: list
    method_1_executable: str
    new_method_executable: str
    method_1_code_experiment: list
    new_method_code_experiment: list
    method_1_code_error: str
    new_method_code_error: str
    method_1_score: str
    new_method_score: str
    method_1_completion: str
    new_method_completion: str
    comparison_result: str
    comparison_result_content: str
    # あとで削除する
    evaluator_test: str


class AIIntegrator:
    def __init__(self, llm_name, save_dir, num_keywords, num_retrieve_paper):
        self.llm_name = llm_name
        self.save_dir = save_dir
        self.num_keywords = num_keywords
        self.num_retrieve_paper = num_retrieve_paper

        graph_builder = StateGraph(State)

        # make nodes
        graph_builder.add_node(
            "keyworder1", LLMNode(llm_name=llm_name, setting=keyworder1_setting)
        )
        graph_builder.add_node(
            "keyworder2", LLMNode(llm_name=llm_name, setting=keyworder2_setting)
        )
        graph_builder.add_node(
            "openalexretriever1",
            OpenAlexNode(
                save_dir=save_dir,
                search_variable="keywords_1",
                output_variable="collection_of_papers_1",
                num_keywords=1,
                num_retrieve_paper=1,
            ),
        )
        graph_builder.add_node(
            "openalexretriever2",
            OpenAlexNode(
                save_dir=save_dir,
                search_variable="keywords_2",
                output_variable="collection_of_papers_2",
                num_keywords=1,
                num_retrieve_paper=1,
            ),
        )
        graph_builder.add_node(
            "extractor1", LLMNode(llm_name=llm_name, setting=extractor1_setting)
        )
        graph_builder.add_node(
            "extractor2", LLMNode(llm_name=llm_name, setting=extractor2_setting)
        )
        graph_builder.add_node(
            "selector1", LLMNode(llm_name=llm_name, setting=selector1_setting)
        )
        graph_builder.add_node(
            "selector2", LLMNode(llm_name=llm_name, setting=selector2_setting)
        )
        graph_builder.add_node(
            "githubretriever1",
            GithubNode(
                save_dir=self.save_dir,
                search_variable="github_url_1",
                output_variable=["folder_structure_1", "github_file_1"],
            ),
        )
        graph_builder.add_node(
            "githubretriever2",
            GithubNode(
                save_dir=self.save_dir,
                search_variable="github_url_1",
                output_variable=["folder_structure_1", "github_file_1"],
            ),
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
        graph_builder.add_node("evaluator1", LLMEvaluateNode)
        graph_builder.add_node("evaluator2", LLMEvaluateNode)
        graph_builder.add_node(
            "debugger1", LLMNode(llm_name=llm_name, setting=debugger1_setting)
        )
        graph_builder.add_node(
            "debugger2", LLMNode(llm_name=llm_name, setting=debugger2_setting)
        )
        graph_builder.add_node(
            "comparator", LLMNode(llm_name=llm_name, setting=comparator_setting)
        )

        # make edges
        graph_builder.add_edge("keyworder1", "openalexretriever1")
        graph_builder.add_edge("openalexretriever1", "extractor1")
        graph_builder.add_edge("extractor1", "selector1")
        graph_builder.add_edge("selector1", "githubretriever1")
        graph_builder.add_edge("githubretriever1", "codeextractor1")
        graph_builder.add_edge("selector1", "keyworder2")
        graph_builder.add_edge("keyworder2", "openalexretriever2")
        graph_builder.add_edge("openalexretriever2", "extractor2")
        graph_builder.add_edge("extractor2", "selector2")
        graph_builder.add_edge("selector2", "githubretriever2")
        graph_builder.add_edge("githubretriever2", "codeextractor2")
        graph_builder.add_edge("codeextractor1", "creator")
        graph_builder.add_edge("codeextractor2", "creator")
        graph_builder.add_edge("codeextractor1", "verifier1")
        graph_builder.add_edge("coder1", "evaluator1")
        graph_builder.add_edge("debugger1", "evaluator1")
        graph_builder.add_edge("creator", "verifier2")
        graph_builder.add_edge("coder2", "evaluator2")
        graph_builder.add_edge("debugger2", "evaluator2")

        # make branches
        graph_builder.add_conditional_edges("verifier1", branchcontroller1)
        graph_builder.add_conditional_edges("evaluator1", branchcontroller2)
        graph_builder.add_conditional_edges("verifier2", branchcontroller3)
        graph_builder.add_conditional_edges("evaluator2", branchcontroller4)

        # set entry and finish points
        graph_builder.set_entry_point("keyworder1")
        graph_builder.set_finish_point("comparator")

        self.graph = graph_builder.compile()

    def __call__(self, memory):
        self.graph.invoke(memory, debug=True)
        return

    def visualize(self, path):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_graph.png", "wb") as f:
            f.write(image.data)


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    save_dir = "/workspaces/researchgraph/data/"
    num_keywords = 1
    num_retrieve_paper = 1
    research_graph = AIIntegrator(llm_name, save_dir, num_keywords, num_retrieve_paper)

    # visualize the graph
    image_dir = "/workspaces/researchgraph/images/"
    research_graph.visualize(image_dir)

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
    # research_graph(state)
