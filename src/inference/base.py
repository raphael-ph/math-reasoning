"""This script defines the base class to run a trained, .pt model"""

from abc import abstractmethod
from pathlib import Path

from torch.nn import Module
from pydantic import BaseModel, Field, ConfigDict

class InferenceEngine(BaseModel):
    # necessary because the tokenizers.Tokenizer is not a pydantic model
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_path: Path = Field(..., description="Path to model (.pt) that will be used for inference")
    model: Module = Field(..., description="Model classfor inference")
    weights_only: bool = Field(default=True, description="True for loading only trained weights (recommended), False to load whole model.")

    @abstractmethod
    def run(*args):
        raise NotImplementedError