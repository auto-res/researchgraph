executor_subgraph_input_data = {
    "github_owner": "auto-res2",
    "repository_name": "auto-research",
    "save_dir": "/workspaces/researchgraph/data",
    "fix_iteration_count": 1,
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
