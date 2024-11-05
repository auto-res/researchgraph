import os
import os.path as osp
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

class MyCoder:
    def __init__(self, llm_model_name="gpt-4o", folder_name="./aider_work_dir"):
        """
        Initializes the MyCoder instance.

        Args:
            llm_model_name (str): The name of the language model to use.
            folder_name (str): The directory where work files are stored.
        """
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
        with open(filepath, 'w') as f:
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

    def run_prompt(self, prompt):
        """
        Runs a prompt using the Coder instance.

        Args:
            prompt (str): The prompt to execute.

        Returns:
            The output from the Coder.
        """
        if not hasattr(self, 'coder'):
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

    def remove_file(self, filename):
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

# Example Usage
if __name__ == "__main__":
    # Initialize MyCoder
    my_coder = MyCoder()

    # Add a new file
    my_coder.add_file("new_script.py", "# Initial content\n")

    # List current files
    print("Current files:", my_coder.list_files())

    # Run a prompt
    prompt = "Add a function to new_script.py that prints 'Hello, World!'"
    output = my_coder.run_prompt(prompt)
    print("Coder Output:", output)

    # Remove a file
    my_coder.remove_file("new_script.py")

