import os
from IPython.display import Image
from pydantic import BaseModel, Field
from langgraph.graph import START,END, StateGraph

from researchgraph.graphs.ai_integrator.ai_integrator_v2.executor_subgraph.input_data import executor_subgraph_input_data
from researchgraph.graphs.ai_integrator.ai_integrator_v2.executor_subgraph.llmnode_prompt import ai_integrator_v2_modifier_prompt
from researchgraph.core.factory import NodeFactory


class ExecutorState(BaseModel):
    new_method_code: str = Field(default="")
    result_save_path: str = Field(default="")
    accuracy: float = Field(default=0.0)
    script_save_path: str = Field(default="")
    model_save_path: str = Field(default="")
    logs: str = Field(default="")
    error_logs: str = Field(default="")
    execution_flag_list: list = Field(default_factory=list)


class ExecutorSubgraph:
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
        ai_integrator_v2_modifier_prompt: str,
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
        self.ai_integrator_v2_modifier_prompt = ai_integrator_v2_modifier_prompt
        self.num_train_data = num_train_data
        self.num_inference_data = num_inference_data

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.graph_builder = StateGraph(ExecutorState)

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
            "modifier",
            NodeFactory.create_node(
                node_name="structuredoutput_llmnode",
                input_key=["new_method_code", "error_logs"],
                output_key=["new_method_code"],
                llm_name=self.llm_name,
                prompt_template=self.ai_integrator_v2_modifier_prompt,
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
                input_key=["script_save_path"],
                output_key = ["logs","model_save_path", "error_logs", "execution_flag_list"]
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
        self.graph_builder.add_edge(START, "text2script")
        self.graph_builder.add_edge("text2script", "llmsfttrainer")
        self.graph_builder.add_conditional_edges(
            "llmsfttrainer", 
            self.decision_function,
            {
                "continue": "llminferencer",
                "correction": "modifier",
                "finish": END,
            }
            )
        self.graph_builder.add_edge("modifier", "text2script")
        self.graph_builder.add_edge("llminferencer", "llmevaluater")
        self.graph_builder.add_edge("llmevaluater", END)

        self.graph = self.graph_builder.compile()
    
    def decision_function(self, state: ExecutorState):
        if len(state.execution_flag_list) <= 3:
            if state.execution_flag_list[-1]:
                return "continue"
            else:
                return "correction"
        else:
            return "finish"

    def __call__(self, state: ExecutorState) -> dict:
        result = self.graph.invoke(state, debug=True)
        return result

    def make_image(self, path: str):
        image = Image(self.graph.get_graph().draw_mermaid_png())
        with open(path + "ai_integrator_v2_executor_subgraph.png", "wb") as f:
            f.write(image.data)

if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    save_dir = "/workspaces/researchgraph/data"
    new_method_file_name = "/workspaces/researchgraph/test/subgraph/ai_integrator_v2/new_method.py"
    ft_model_name = "meta-llama/Llama-3.2-3B"
    dataset_name = "openai/gsm8k"
    model_save_path = "/workspaces/researchgraph/data/model"
    result_save_file_name = "result.csv"
    answer_data_path = "/workspaces/researchgraph/data/gsm8k_answer.csv"
    num_train_data = 5
    num_inference_data = 5
    executor_subgraph = ExecutorSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
        new_method_file_name=new_method_file_name,
        ft_model_name=ft_model_name,
        dataset_name=dataset_name,
        model_save_dir_name=model_save_path,
        result_save_file_name=result_save_file_name,
        answer_data_path=answer_data_path,
        ai_integrator_v2_modifier_prompt=ai_integrator_v2_modifier_prompt,
        num_train_data=num_train_data,
        num_inference_data=num_inference_data,
    )
    
    # executor_subgraph(
    #     state = executor_subgraph_input_data, 
    #     )
    
    image_dir = "/workspaces/researchgraph/images/"
    executor_subgraph.make_image(image_dir)
