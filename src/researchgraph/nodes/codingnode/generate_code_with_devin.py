import os
import time
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from researchgraph.core.node import Node

from researchgraph.nodes.utils.api_request_handler import fetch_api_data, retry_request


DEVIN_API_KEY = os.getenv("DEVIN_API_KEY")


class State(BaseModel):
    github_owner: str = Field(default="")
    repository_name: str = Field(default="")
    new_method_text: str = Field(default="")
    new_method_code: str = Field(default="")
    session_id: str = Field(default="")
    branch_name: str = Field(default="")
    devin_url: str = Field(default="")


class GenerateCodeWithDevinNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
    ):
        super().__init__(input_key, output_key)
        self.headers = {
            "Authorization": f"Bearer {DEVIN_API_KEY}",
            "Content-Type": "application/json",
        }

    def _request_create_session(self, repository_url, new_method_text, new_method_code):
        url = "https://api.devin.ai/v1/sessions"
        data = {
            "prompt": f"""
The “New Method Text” and “New Method Code” sections contain ideas for new machine learning research and the code associated with those ideas. 
Please follow the “Rules” section to create an experimental script to conduct this research.
Also, please make sure that you can output the file according to the “Output Format”.
# Rules
- Create and implement a new branch in the repository given in “Repository URL”. The name of the branch should be associated with the new methodology.
- The experimental scripts should be run through a simple test run to verify that they work.
- Install and use the necessary python packages as needed.
- Please also list the python packages required for the experiment in the requirements.txt file.
- The roles of directories and scripts are listed below. Follow the roles to complete your implementation.
    - .github/workflows...Do not change the contents of this directory.
    - config...If you want to set parameters for running the experiment, place the file that completes the parameters under this directory.
    - data...This directory is used to store data used for model training and evaluation.
    - models...This directory is used to store pre-trained and trained models.
    - paper...Do not change anything in this directory.
    - src
        - train.py...Scripts for training models. Implement the code to train the models.
        - evaluate.py...Script to evaluate the model. Implement the code to evaluate the model.
        - preprocess.py...Script for preprocessing data. Implement the code necessary for data preprocessing.
        - main.py...Scripts for running the experiment, using train.py, evaluate.py, and preprocess.py to implement the entire process from model training to evaluation.
    - requirements.txt...Please list the python packages required to run the model.        
# New Method Text
{new_method_text}
# New Method Code
{new_method_code}
# Repository URL
{repository_url}
# Output Format
the name of the new branch created when creating the experimental script should be able to be output as "branch_name".
""",
            "idempotent": True,
        }
        return retry_request(
            fetch_api_data, url, headers=self.headers, data=data, method="POST"
        )

    def _request_devin_output(self, session_id):
        url = f"https://api.devin.ai/v1/session/{session_id}"

        def should_retry(response):
            # Describe the process so that it is True if you want to retry
            return response.get("status_enum") not in ["blocked", "stopped"]

        return retry_request(
            fetch_api_data,
            url,
            headers=self.headers,
            method="GET",
            check_condition=should_retry,
        )

    def execute(self, state: State) -> dict:
        github_owner = getattr(state, self.input_key[0])
        repository_name = getattr(state, self.input_key[1])
        repository_url = f"https://github.com/{github_owner}/{repository_name}"
        new_method_text = getattr(state, self.input_key[2])
        new_method_code = getattr(state, self.input_key[3])
        response = self._request_create_session(
            repository_url, new_method_text, new_method_code
        )
        if response:
            print("Successfully created Devin session.")
            session_id = response["session_id"]
            devin_url = response["url"]
        else:
            print("Failed to create Devin session.")

        # NOTE: Devin takes a while to complete its execution, so it does not send unnecessary requests.
        time.sleep(120)
        if session_id is not None:
            response = self._request_devin_output(session_id)
            print(response)
        return {
            self.output_key[0]: session_id,
            self.output_key[1]: response["structured_output"].get("branch_name", ""),
            self.output_key[2]: devin_url,
        }


if __name__ == "__main__":
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "codegenerator",
        GenerateCodeWithDevinNode(
            input_key=[
                "github_owner",
                "repository_name",
                "new_method_text",
                "new_method_code",
            ],
            output_key=["session_id", "branch_name", "devin_url"],
        ),
    )
    graph_builder.add_edge(START, "codegenerator")
    graph_builder.add_edge("codegenerator", END)
    graph = graph_builder.compile()
    state = {
        "github_owner": "auto-res",
        "repository_name": "experimental-script",
        "new_method_text": """
Learnable Gated Pooling: A New Approach
This approach combines the benefits of learnable weights (as discussed in the previous responses) with a gating mechanism. The gating mechanism allows the model to dynamically decide how much of each element in the input sequence should contribute to the final pooled vector. This adds another layer of flexibility and expressiveness compared to simple learnable weights.
Here's the breakdown:
Learnable Weights: Similar to the previous "Approach 1," we introduce learnable weights (w) for each dimension of the input vectors. These weights determine the importance of each feature.
Gating Mechanism: We introduce a separate set of learnable parameters to create a "gate" (g). This gate is also applied element-wise to the weighted input. The gate values are typically between 0 and 1, controlling how much of the weighted input is passed through to the pooling operation.
Pooling Operation: After applying both the weights and the gate, we perform a pooling operation (e.g., average pooling, sum pooling).
Mathematical Representation

Let x = [x_1, x_2, ..., x_n] be the input sequence, where x_i is the vector representation of the i-th element. Let w = [w_1, w_2, ..., w_d] be the learnable weights (where d is the dimension of x_i), and g = [g_1, g_2, ..., g_n] be the learnable gates (where n is the sequence length).

The Learnable Gated Pooling can be represented as:

weighted_x = x * w  (element-wise multiplication, broadcasting w across the sequence length)
gated_x = weighted_x * sigmoid(g) (element-wise multiplication)
pooled_vector = pooling_operation(gated_x)
where pooling_operation can be average pooling (mean), sum pooling, or another suitable operation. The sigmoid function ensures the gate values are between 0 and 1.
""",
        "new_method_code": """
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1) # Linear layer for gating

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)

        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2) # (batch_size, seq_len)
        gated_x = weighted_x * gate_values.unsqueeze(2)

        pooled_vector = torch.mean(gated_x, dim=1)  # Average pooling
        return pooled_vector

# Example usage
input_dim = 768  # Example: BERT embedding dimension
batch_size = 32
seq_len = 10
embeddings = torch.randn(batch_size, seq_len, input_dim)

learnable_gated_pooling = LearnableGatedPooling(input_dim, seq_len)
pooled_output = learnable_gated_pooling(embeddings)

print(pooled_output.shape)  # Output: torch.Size([32, 768])""",
    }
    graph.invoke(state, debug=True)
