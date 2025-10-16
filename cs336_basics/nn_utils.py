"""
Neural network utility modules for CS336 Assignment 1.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    """
    A linear transformation module that performs y = xW^T + b.
    
    This implementation follows PyTorch's nn.Linear interface but without bias.
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        
        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output  
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        
        # TODO: Store the input and output dimensions as instance variables
        # self.in_features = ?
        # self.out_features = ?
        
        # TODO: Create the weight parameter W with shape (out_features, in_features)
        # Use nn.Parameter to wrap the tensor so it's recognized as a parameter
        # Initialize using torch.nn.init.trunc_normal_ with default settings
        # self.W = ?
        
        # TODO: Move the parameter to the specified device and dtype if provided
        # if device is not None:
        #     self.W = self.W.to(device)
        # if dtype is not None:
        #     self.W = self.W.to(dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
        """
        # TODO: Implement the linear transformation y = xW^T
        # Hint: Use torch.matmul or @ operator
        # The weight W has shape (out_features, in_features)
        # So W^T has shape (in_features, out_features)
        # For batched input x of shape (batch_size, ..., in_features),
        # we want output of shape (batch_size, ..., out_features)
        
        # return ?
        raise NotImplementedError("TODO: Implement the forward pass")
    
    def extra_repr(self) -> str:
        """Return a string representation of the module parameters."""
        return f'in_features={self.in_features}, out_features={self.out_features}'


# Additional neural network modules can be added here in future assignments
