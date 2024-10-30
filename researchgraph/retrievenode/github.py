import os
import subprocess
import glob

from typing import Any
from pydantic import BaseModel, DirectoryPath, HttpUrl
from langgraph.graph import StateGraph


class State(BaseModel):
    github_url: HttpUrl
    folder_structure: str
    github_file: str


class GithubNode:
    def __init__(self, save_dir: DirectoryPath, search_variable: str, output_variable: list[str]):
        self.save_dir = save_dir
        self.search_variable = search_variable
        self.output_variable = output_variable
        print("GithubRetriever initialized")
        print(f"input: {self.search_variable}")
        print(f"output: {self.output_variable}")

    def get_folder_structure(self, path=".") -> str | None:
        try:
            # subprocess.runを使用してフォルダ構造を取得
            result = subprocess.run(
                ["ls", "-R", path], capture_output=True, text=True, check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            return None

    def git_clone(self, url: str) -> subprocess.CompletedProcess:
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        os.chdir(self.save_dir)
        command = f"git clone {url}"
        ls = subprocess.run(command, shell=True, text=True)
        return ls

    def get_py_files(self, url: str) -> tuple[str, str]:
        folder_structure = self.get_folder_structure(url.split("/")[-1])
        py_filelist = glob.glob(url.split("/")[-1] + "/**/*.py", recursive=True)
        py_text = []
        for i in py_filelist:
            with open(i) as f:
                file_read = f.read()
            py_text.append(
                "<FILE="
                + i
                + "> \n"
                + file_read.replace("\n\n", "\n").replace("\n\n", "\n")
                + "</FILE> \n"
            )
            result = "".join(py_text)
        return folder_structure, result

    def __call__(self, state: State) -> Any:
        github_url = state[self.search_variable]
        self.git_clone(github_url)
        folder_structure, get_file = self.get_py_files(github_url)
        state[self.output_variable[0]] = folder_structure
        state[self.output_variable[1]] = get_file
        return state


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    search_variable = "github_url"
    output_variable = ["folder_structure", "github_file"]

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "githubretriever",
        GithubNode(
            save_dir=save_dir,
            search_variable=search_variable,
            output_variable=output_variable,
        ),
    )
    graph_builder.set_entry_point("githubretriever")
    graph_builder.set_finish_point("githubretriever")
    graph = graph_builder.compile()

    memory = {"github_url": "https://github.com/fuyu-quant/IBLM"}

    graph.invoke(memory, debug=True)
