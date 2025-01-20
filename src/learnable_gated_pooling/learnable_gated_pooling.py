import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim, seq_len):
        """
        Initialize the Learnable Gated Pooling module.
        
        Args:
            input_dim (int): Dimension of input vectors
            seq_len (int): Length of input sequence
        """
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)  # Linear layer for gating

    def forward(self, x):
        """
        Forward pass of the Learnable Gated Pooling module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Pooled output tensor of shape (batch_size, input_dim)
        """
        # Apply learnable weights to input
        weighted_x = x * self.weights  # Element-wise multiplication, broadcasting weights

        # Compute gate values
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)  # (batch_size, seq_len)
        
        # Apply gates to weighted input
        gated_x = weighted_x * gate_values.unsqueeze(2)  # Element-wise multiplication
        
        # Perform pooling operation (average pooling)
        pooled_vector = torch.mean(gated_x, dim=1)  # Average pooling across sequence length
        
        return pooled_vector
