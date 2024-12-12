import os
import re
import subprocess
import glob

from researchgraph.core.node import Node


class RetrieveGithubRepositoryNode(Node):
    def __init__(
        self, input_key: list[str], output_key: list[str], save_dir: str
    ):
        super().__init__(input_key, output_key)
        self.save_dir = save_dir

    def _format_url(self, state) -> str:
        pattern = r"(https://github\.com/[^/]+/[^/?#]+)"
        url = re.search(pattern, state[self.input_key[0]]).group(1)
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

    def execute(self, state) -> dict:
        github_url = self._format_url(state)
        repository_name = github_url.split("/")[-1]
        target_dir = os.path.join(self.save_dir, repository_name)

        result = self._get_repository(github_url, repository_name)
        all_file_path = self._get_all_file_path(target_dir)
        python_script_text = self._get_python_script_text(target_dir)
        return {
            self.input_key[0]: github_url,
            self.output_key[0]: all_file_path,
            self.output_key[1]: python_script_text,
        }
