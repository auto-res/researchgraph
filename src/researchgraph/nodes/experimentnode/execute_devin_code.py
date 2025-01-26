import os
import subprocess
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node
from datetime import datetime


class State(BaseModel):
    repository_url: str = Field(default="")
    branch_name: str = Field(default="")
    save_dir: str = Field(default="")
    output_file_name: str = Field(default="")
    error_file_name: str = Field(default="")


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

    def _execute_script(
        self, save_dir: str, output_file_path: str, error_file_path: str
    ):
        script_path = f"{save_dir}/src/main.py"
        if not os.path.exists(script_path):
            print(f"スクリプトが見つかりません: {script_path}")
            return
        command = ["python", script_path]
        with (
            open(output_file_path, "w") as output_file,
            open(error_file_path, "w") as error_file,
        ):
            try:
                subprocess.run(
                    command,
                    stdout=output_file,
                    stderr=error_file,
                    shell=False,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                error_file.write(f"command: {command}\n")
                error_file.write(f"Error occurred: {e}\n")

    def execute(self, state: State) -> dict:
        repository_url = getattr(state, self.input_key[0])
        branch_name = getattr(state, self.input_key[1])
        save_dir = getattr(state, self.input_key[2])
        output_file_name = getattr(state, self.input_key[3])
        error_file_name = getattr(state, self.input_key[4])
        # Repository Clone
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_repository_dir = os.path.join(save_dir, f"experimental-script_{timestamp}")
        self._git_clone(repository_url, branch_name, save_repository_dir)
        # Script Execution
        output_file_path = os.path.join(save_dir, output_file_name)
        error_file_path = os.path.join(save_dir, error_file_name)
        self._execute_script(save_repository_dir, output_file_path, error_file_path)
        return {self.output_key[0]: self.output_logs_path}


if __name__ == "__main__":
    output_logs_path = (
        "/workspaces/researchgraph/data/experimental-script/logs/logs.txt"
    )
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "executor",
        ExecuteCodeNode(
            input_key=[
                "repository_url",
                "branch_name",
                "save_dir",
                "output_file_name",
                "error_file_name",
            ],
            output_key=["output_logs_path"],
            output_logs_path=output_logs_path,
        ),
    )
    graph_builder.add_edge(START, "executor")
    graph_builder.add_edge("executor", END)
    graph = graph_builder.compile()
    state = {
        "repository_url": "https://github.com/auto-res/experimental-script",
        "branch_name": "devin/1736784724-learnable-gated-pooling",
        "save_dir": "/workspaces/researchgraph/data",
        "output_file_name": "output.txt",
        "error_file_name": "error.txt",
    }
    graph.invoke(state, debug=True)
