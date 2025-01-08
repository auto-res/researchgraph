from researchgraph.graphs.ai_integrator.ai_integrator_v3.executor_subgraph.main import ExecutorSubgraph
from researchgraph.graphs.ai_integrator.ai_integrator_v3.executor_subgraph.input_data import executor_subgraph_input_data


def test_executor_subgraph():
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
        num_train_data=num_train_data,
        num_inference_data=num_inference_data,
    )
    
    assert executor_subgraph(
        state = executor_subgraph_input_data, 
        )
