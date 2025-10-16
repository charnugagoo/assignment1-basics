import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"

# Import neural network modules
from .nn_utils import Linear, Embedding

__all__ = ["Linear", "Embedding"]
