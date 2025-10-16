# Assignment to implement neural networks

## Problem 1 - Linear - Implementing the liner module

Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear module, except for not having a bias argument or parameter. We recommend the following interface:

```
def __init__(self, in_features, out_features, device=None, dtype=None)
    """
    Construct a linear transformation module. This function should accept the following parameters:

    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters

def forward(self, x: torch.Tensor) -> torch.Tensor
    """
    Apply the linear transformation to the input.
    """
```
Make sure to:
• subclass nn.Module
• call the superclass constructor
• construct and store your parameter as W (not W⊤) for memory ordering reasons, putting it i an nn.Parameter
• of course, don’t use nn.Linear or nn.functional.linear

For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to initialize the weights.

To test your Linear module, implement the test adapter at [adapters.run_linear]. The adapter should load the given weights into your Linear module. You can use Module.load_state_dict for this purpose. Then, run uv run pytest -k test_linear.