# %%
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

import subprocess

class State(TypedDict):
    method_1_code_experiment: str
    

class LLMEvaluateNode:
    def __init__(self, save_dir, evaluate_code):
        self.save_dir = save_dir
        self.evaluate_code = evaluate_code
        
    def train(self, state: State):
        exec_code = state[evaluate_code]
        
        with open(self.save_dir + f'{evaluate_code}.py', 'w') as file:
            file.write(exec_code)
            
        command = [
            "python", 
            f"self.save_dir{evaluate_code}.py",
            "--model_name_or_path=meta-llama/Meta-Llama-3-8B",
            "--tokenizer_name=meta-llama/Meta-Llama-3-8B",
            "--dataset_name=izumi-lab/cc100-ja-filter-ja-normal",
            "--auto_find_batch_size",
            "--do_train",
            "--do_eval",
            "--num_train_epochs=3",
            "--save_steps=1000",
            "--save_total_limit=3",
            f"--output_dir={self.save_dir}/{evaluate_code}",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        return
        
        
        with open(self.save

    def __call__(self, state: State) -> Any:
        self.train(state)
        
        return state

if __name__ == "__main__": 
    memory = {"evaluator_test": "hoge"}
    evaluate_code = "method_1_code_experiment"
    
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("evaluator",LLMEvaluateNode(evaluate_code))
    graph_builder.set_entry_point("evaluator")
    graph_builder.set_finish_point("evaluator")
    graph = graph_builder.compile()
    graph.invoke(memory, debug=True)
