import os
import os.path as osp
import json
from typing import TypedDict
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from langgraph.graph import StateGraph


class State(TypedDict):
    notes_path: str
    writeup_file_path: str
    review_path: str | None


class DraftImprovementComponent:
    def __init__(
        self,
        input_variable: list,  # writeup_file, notes, review_path
        output_variable: str,  # review_path
        model: str,
        io: InputOutput,
    ):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.model = model
        self.io = io
        self.coder = None

    def __call__(self, state: State) -> dict:
        # Extract review content from state
        review_path = state.get("review_path")
        if not review_path:
            raise ValueError("Review content is required in the state.")

        # Initialize the Coder instance
        self.coder = Coder.create(
            main_model=Model(self.model), 
            fnames=self.input_variable,
            io=self.io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        # Read review content
        with open(review_path, "r") as f:
            review_content = f.read()

        # Perform improvement using the review
        improved_content = self._perform_improvement(review_content)

        # Save the improved content to the output file
        output_file = state[self.output_variable]
        with open(output_file, "w") as f:
            f.write(improved_content + "\n")

        return {
            self.output_variable: output_file
        }

    def _perform_improvement(self, review: str) -> str:
        improvement_prompt = '''The following review has been created for your research paper:
        """
        {review}
        """

        Improve the text using the review.'''.format(review=json.dumps(review))
        coder_out = self.coder.run(improvement_prompt)
        return coder_out


if __name__ == "__main__":

    import openai

    # Define input and output variables
    input_variable = ["writeup_file_path", "notes_path"]
    output_variable = "review_path"
    model = "gpt-3.5-turbo"
    io = InputOutput()
    template_dir = "/workspaces/researchgraph/src/researchgraph/graph/ai_scientist/templates/2d_diffusion"
    cite_client = openai

    # Initialize DraftImprovementComponent as a LangGraph node
    draft_improvement_component = DraftImprovementComponent(
        input_variable=input_variable,
        output_variable=output_variable,
        model=model,
        io=io,
    )

    # Create the StateGraph and add node
    graph_builder = StateGraph(State)
    graph_builder.add_node("draft_improvement_component", draft_improvement_component)
    graph_builder.set_entry_point("draft_improvement_component")
    graph_builder.set_finish_point("draft_improvement_component")
    graph = graph_builder.compile()

    # Define initial state
    memory = {
        "notes_path": "/workspaces/researchgraph/data/notes.txt",
        "writeup_file_path": "/workspaces/researchgraph/data/writeup_file.txt",
        "review_path": "/workspaces/researchgraph/data/review.txt",
    }

    # Execute the graph
    graph.invoke(memory)
