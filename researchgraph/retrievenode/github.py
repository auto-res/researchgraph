import os
import re
import subprocess
import glob
import logging

from typing_extensions import TypedDict
from langgraph.graph import StateGraph

logger = logging.getLogger("researchgraph")


class State(TypedDict):
    github_url: str
    folder_structure: str
    github_file: str


class GithubNode:
    def __init__(self, save_dir: str, input_variable: str, output_variable: list[str]):
        self.save_dir = save_dir
        self.input_variable = input_variable
        self.output_variable = output_variable
        print("GithubNode initialized")
        print(f"input: {self.input_variable}")
        print(f"output: {self.output_variable}")

    def _format_url(self, state: State) -> str:
        pattern = r"(https://github\.com/[^/]+/[^/?#]+)"
        url = re.search(pattern, state[self.input_variable]).group(1)
        return url

    def _get_repository(self, url: str, repo_name: str):
        repo_path = os.path.join(self.save_dir, repo_name)
        if os.path.exists(repo_path):
            print(f"Repository '{repo_name}' already exists")
            return None
        else:
            command = ["git", "clone", url, repo_path]
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print(f"Successfully cloned '{repo_name}'")
                return result
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone '{repo_name}': {e.stderr}")
                return None

    def _get_all_file_path(self, search_dir: str) -> str:
        try:
            result = subprocess.run(
                ["ls", "-R", search_dir], capture_output=True, text=True, check=True
            )
            all_file_paths = result.stdout
            all_file_paths = "\n".join(
                [line for line in all_file_paths.splitlines() if line.strip()]
            )
            return all_file_paths
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            return None

    def _get_python_script_text(self, search_dir: str) -> str:
        py_filelist = glob.glob(search_dir + "/**/*.py", recursive=True)
        python_script_text = ""
        for i in py_filelist:
            with open(i) as f:
                file_read = f.read()
            cleaned_code = re.sub(r"\n+", "\n", file_read)
            python_script_text += f"<FILE={i}>\n{cleaned_code}\n"
        return python_script_text[
            :10000
        ]  # TODO: The problem of Python code becoming too long. GPT-4o context window is 128,000.

    def __call__(self, state: State) -> dict:
        github_url = self._format_url(state)
        repository_name = github_url.split("/")[-1]
        target_dir = os.path.join(self.save_dir, repository_name)

        result = self._get_repository(github_url, repository_name)
        all_file_path = self._get_all_file_path(target_dir)
        python_script_text = self._get_python_script_text(target_dir)
        logger.info("---GithubNode---")
        logger.info(f"All file path: {all_file_path}")
        logger.info(f"Python script text: {python_script_text[:500]}")
        return {
            self.input_variable: github_url,
            self.output_variable[0]: all_file_path,
            self.output_variable[1]: python_script_text,
        }


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    input_variable = "github_url"
    output_variable = ["folder_structure", "github_file"]

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "githubretriever",
        GithubNode(
            save_dir=save_dir,
            input_variable=input_variable,
            output_variable=output_variable,
        ),
    )
    graph_builder.set_entry_point("githubretriever")
    graph_builder.set_finish_point("githubretriever")
    graph = graph_builder.compile()

    # memory = {"github_url": "https://github.com/abhi2610/ohem/tree/1f07dd09b50c8c21716ae36aede92125fe437579"}
    memory = {
        "github_url": "https://github.com/adelnabli/acid?tab=readme-ov-file/info/refs"
    }

    # graph.invoke(memory, debug=True)
    graph.invoke(memory)
