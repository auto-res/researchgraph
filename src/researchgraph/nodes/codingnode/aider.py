import os
import os.path as osp
import logging

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from typing_extensions import TypedDict
from langgraph.graph import StateGraph

logger = logging.getLogger("researchgraph")


class State(TypedDict):
    instruction: str


class AiderNode:
    def __init__(
        self,
        input_key: str,
        # output_key: str,
        llm_model_name: str,
        folder_name: str,
    ):
        """
        Initializes the MyCoder instance.

        Args:
            llm_model_name (str): The name of the language model to use.
            folder_name (str): The directory where work files are stored.
        """
        self.input_key = input_key
        # self.output_variable = output_variable
        self.main_model = Model(llm_model_name)
        self.folder_name = folder_name

        # Ensure the working directory exists
        os.makedirs(self.folder_name, exist_ok=True)

        # Define default files
        self.fnames = [
            osp.join(self.folder_name, "experiment.py"),
            osp.join(self.folder_name, "plot.py"),
            osp.join(self.folder_name, "notes.txt"),
        ]

        # Initialize InputOutput
        self.io = InputOutput(
            yes=True, chat_history_file=osp.join(self.folder_name, "_aider.txt")
        )

        # Create the Coder instance
        self.coder = Coder.create(
            main_model=self.main_model,
            fnames=self.fnames,
            io=self.io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

    def _run_prompt(self, prompt):
        """
        Runs a prompt using the Coder instance.

        Args:
            prompt (str): The prompt to execute.

        Returns:
            The output from the Coder.
        """
        if not hasattr(self, "coder"):
            raise AttributeError("Coder instance is not initialized.")

        print("Running prompt...")
        output = self.coder.run(prompt)
        print("Prompt executed.")
        return output

    def list_files(self):
        """
        Lists all files managed by the Coder.

        Returns:
            A list of file paths.
        """
        return self.fnames

    def add_file(self, filename, content=""):
        """
        Adds a new file to the working directory and updates the Coder instance.

        Args:
            filename (str): The name of the file to add.
            content (str): Optional initial content for the file.
        """
        filepath = osp.join(self.folder_name, filename)

        # Add the new file to the list of filenames
        self.fnames.append(filepath)

        # Create the file with optional content
        with open(filepath, "w") as f:
            f.write(content)

        # Recreate the Coder instance with the updated file list
        self.coder = Coder.create(
            main_model=self.main_model,
            fnames=self.fnames,
            io=self.io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )
        print(f"Added new file: {filepath}")

    def remove_file(self, filename: str):
        """
        Removes a file from the working directory and updates the Coder instance.

        Args:
            filename (str): The name of the file to remove.
        """
        filepath = osp.join(self.folder_name, filename)
        if filepath in self.fnames:
            self.fnames.remove(filepath)
            if osp.exists(filepath):
                os.remove(filepath)
            # Recreate the Coder instance without the removed file
            self.coder = Coder.create(
                main_model=self.main_model,
                fnames=self.fnames,
                io=self.io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            print(f"Removed file: {filepath}")
        else:
            print(f"File not found in managed list: {filepath}")

    def __call__(self, state: State):
        prompt = state[self.input_key]
        output = self._run_prompt(prompt)
        print("Coder Output:", output)
        return


# Example Usage
if __name__ == "__main__":
    # Initialize MyCoder
    # my_coder = MyCoder()

    # Add a new file
    # my_coder.add_file("new_script.py", "# Initial content\n")

    # List current files
    # print("Current files:", my_coder.list_files())

    # Run a prompt
    # prompt = "Add a function to new_script.py that prints 'Hello, World!'"
    # output = my_coder.run_prompt(prompt)
    # print("Coder Output:", output)

    # Remove a file
    # my_coder.remove_file("new_script.py")

    llm_model_name = "gpt-4o"
    folder_name = "/workspaces/researchgraph/researchgraph/codingnode/aidernode_test/aider_work_dir"

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "aider",
        AiderNode(
            input_key="instruction",
            llm_model_name=llm_model_name,
            folder_name=folder_name,
        ),
    )

    graph_builder.set_entry_point("aider")
    graph_builder.set_finish_point("aider")
    graph = graph_builder.compile()

    memory = {
        "instruction": "Add a function to new_script.py that prints 'Hello, World!"
    }

    graph.invoke(memory, debug=True)
    # graph.invoke(memory)
