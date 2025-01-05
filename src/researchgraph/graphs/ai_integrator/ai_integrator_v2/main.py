import os
from langgraph.graph import START, END, StateGraph

from researchgraph.graphs.ai_integrator.ai_integrator_v2.generator_subgraph.main import GeneratorSubgraph, GeneratorState
from researchgraph.graphs.ai_integrator.ai_integrator_v2.generator_subgraph.input_data import generator_subgraph_input_data
from researchgraph.graphs.ai_integrator.ai_integrator_v2.generator_subgraph.llmnode_prompt import (
    ai_integrator_v2_extractor_prompt,
    ai_integrator_v2_codeextractor_prompt,
    ai_integrator_v2_creator_prompt,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v2.executor_subgraph.main import ExecutorSubgraph, ExecutorState

class AIIntegratorv2State(GeneratorState, ExecutorState):
    pass


class AIIntegratorv2:
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
        ai_integrator_v2_extractor_prompt: str,
        ai_integrator_v2_codeextractor_prompt: str,
        ai_integrator_v2_creator_prompt: str,
        num_train_data: int | None = None,
        num_inference_data: int | None = None,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        # Generator Subgraph
        self.ai_integrator_v2_extractor_prompt = ai_integrator_v2_extractor_prompt
        self.ai_integrator_v2_codeextractor_prompt = (
            ai_integrator_v2_codeextractor_prompt
        )
        self.ai_integrator_v2_creator_prompt = ai_integrator_v2_creator_prompt
        # Executor Subgraph
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
        self.graph_builder = StateGraph(AIIntegratorv2State)

        self.graph_builder.add_node(
            "generator",
            GeneratorSubgraph(
                llm_name=llm_name,
                save_dir=save_dir,
                ai_integrator_v2_extractor_prompt=self.ai_integrator_v2_extractor_prompt,
                ai_integrator_v2_codeextractor_prompt=self.ai_integrator_v2_codeextractor_prompt,
                ai_integrator_v2_creator_prompt=self.ai_integrator_v2_creator_prompt,
            )
        )
        self.graph_builder.add_node(
            "executor",
            ExecutorSubgraph(
                llm_name=self.llm_name,
                save_dir=self.save_dir,
                new_method_file_name=self.new_method_file_name,
                ft_model_name=self.ft_model_name,
                dataset_name=self.dataset_name,
                model_save_dir_name=self.model_save_dir_name,
                result_save_file_name=self.result_save_file_name,
                answer_data_path=self.answer_data_path,
                num_train_data=self.num_train_data,
                num_inference_data=self.num_inference_data,
            )
        )
   
        # make edges
        self.graph_builder.add_edge(START, "executor")
        self.graph_builder.add_edge("generator", "executor")
        self.graph_builder.add_edge("executor", END)

        self.graph = self.graph_builder.compile()

    def __call__(self, state: AIIntegratorv2State) -> dict:
        result = self.graph.invoke(state, debug=True)
        return result


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    save_dir = "/workspaces/researchgraph/data"
    new_method_file_name = "new_method.py"
    ft_model_name = "meta-llama/Llama-3.2-3B"
    dataset_name = "openai/gsm8k"
    model_save_dir_name = "/workspaces/researchgraph/data/model"
    result_save_file_name = "result.csv"
    answer_data_path = "/workspaces/researchgraph/data/gsm8k_answer.csv"
    num_train_data = 5
    num_inference_data = 5
    ai_integrator_v2 = AIIntegratorv2(
        llm_name=llm_name,
        save_dir=save_dir,
        new_method_file_name=new_method_file_name,
        ft_model_name=ft_model_name,
        dataset_name=dataset_name,
        model_save_dir_name=model_save_dir_name,
        result_save_file_name=result_save_file_name,
        answer_data_path=answer_data_path,
        ai_integrator_v2_extractor_prompt=ai_integrator_v2_extractor_prompt,
        ai_integrator_v2_codeextractor_prompt=ai_integrator_v2_codeextractor_prompt,
        ai_integrator_v2_creator_prompt=ai_integrator_v2_creator_prompt,
        num_train_data=num_train_data,
        num_inference_data=num_inference_data,
    )
    ai_integrator_v2(
        state = AIIntegratorv2State(
            state = generator_subgraph_input_data,
        )
    )
