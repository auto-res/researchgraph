import subprocess
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node

class State(BaseModel):
    repository_url: str = Field(default="")
    
class ExecuteCodeNode(Node):
    def __init__(
        self, 
        input_key: list[str], 
        output_key: list[str],
        output_logs_path: str,
    ):
        super().__init__(input_key, output_key)
        self.output_logs_path = output_logs_path

    def _git_clone(self, repository_url: str, branch_name: str, save_dir: str):
        command = f"git clone -b {branch_name} {repository_url} {save_dir}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"エラーが発生しました: {e}")

    def _execute_script(self, save_dir: str):
        command = ["python", f"{save_dir}/src/main.py"]
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"エラーが発生しました: {e}")

    def execute(self, state: State) -> dict:
        repository_url = getattr(state, self.input_key[0])
        branch_name = getattr(state, self.input_key[1])
        save_dir = getattr(state, self.input_key[2])
        self._git_clone(repository_url, branch_name, save_dir)
        self._execute_script(save_dir)
        return {
            self.output_key[0]: self.output_logs_path
        }

if __name__ == "__main__":
    output_logs_path = "/workspaces/researchgraph/data/experimental-script/logs/logs.txt"
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "executor",
        ExecuteCodeNode(
            input_key = ["repository_url", "branch_name", "save_dir"],
            output_key = ["output_logs_path"],
            output_logs_path = output_logs_path,
            
            )
    )
    graph_builder.add_edge(START, "executor")
    graph_builder.add_edge("executor", END)
    graph = graph_builder.compile()
    state = {
        "repository_url": "https://github.com/auto-res/experimental-script",
        'branch_name': 'devin/1736784724-learnable-gated-pooling',
        "save_dir" : "/workspaces/researchgraph/data",
    }
    graph.invoke(state, debug=True)
