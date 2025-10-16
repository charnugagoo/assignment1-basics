"""
Neural network utility modules for CS336 Assignment 1.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    """
    A linear transformation module that performs y = xW^T.
    
    This implementation follows PyTorch's nn.Linear interface but without bias.
    """
    
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device=None,
            dtype=None
    ) -> None:
        """
        Construct a linear transformation module.
        
        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output  
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        
        # Store the input and output dimensions as instance variables
        self.in_features = in_features
        self.out_features = out_features
        
        # Create the weight parameter W with shape (out_features, in_features)
        # Use nn.Parameter to wrap the tensor so it's recognized as a parameter
        # Initialize using torch.nn.init.trunc_normal_ with default settings
        self.W = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.trunc_normal_(self.W, std=2.0 / (in_features + out_features))

        # Move the parameter to the specified device and dtype if provided
        if device is not None:
            self.W = self.W.to(device)
        if dtype is not None:
            self.W = self.W.to(dtype)
    
    def forward(self, x: Float[Tensor, " ... d_in"]) -> Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
        """
        # Implement the linear transformation y = xW^T
        # The weight W has shape (out_features, in_features)
        # So W^T has shape (in_features, out_features)
        # For batched input x of shape (batch_size, ..., in_features),
        # we want output of shape (batch_size, ..., out_features)
        
        return x @ self.W.T
        # Alternative einsum approach (equivalent but more complex):
        # return torch.einsum("...i,oi->...o", x, self.W)
    
    def extra_repr(self) -> str:
        """Return a string representation of the module parameters."""
        return f'in_features={self.in_features}, out_features={self.out_features}'

class Embedding(nn.Module):
    """
    A word embedding module that performs y = xE.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct a word embedding module.

        Don't use nn.Embedding or nn.functional.embedding here for this assignment for study purpose
        # self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        Args:
            num_embeddings (int): The size of the vocabulary
            embedding_dim (int): The dimensionality of the embedding vectors, i.e., d_model
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        
        # Store the vocabulary size and embedding dimensions as instance variables
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create the embedding parameter with shape (num_embeddings, embedding_dim)
        # Each row represents the embedding vector for a token ID
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        nn.init.trunc_normal_(self.embedding, std=1.0)

        # Move the parameter to the specified device and dtype if provided
        if device is not None:
            self.embedding = self.embedding.to(device)
        if dtype is not None:
            self.embedding = self.embedding.to(dtype)

    def forward(self, token_ids: Int[Tensor, " ... num_tokens"]) -> Float[Tensor, " ... d_model"]:
        """
        Apply the word embedding to the input.
        
        Args:
            token_ids (Int[Tensor, " ... num_tokens"]): Input tensor of token IDs with arbitrary shape
            
        Returns:
            Float[Tensor, " ... num_tokens embedding_dim"]: Output tensor with embedding_dim as the last dimension
        """
        # Lookup the embeddings for the token ids
        # This performs embedding lookup: for each token_id, return the corresponding row
        return self.embedding[token_ids]
    
    def extra_repr(self) -> str:
        """Return a string representation of the module parameters."""
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}'


# Additional neural network modules can be added here in future assignments
