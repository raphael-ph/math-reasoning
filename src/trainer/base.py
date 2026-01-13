# --- Base Trainer ---
# This is the base structure for training AI Models. Since I'll be training some models in this repo, 
# I thought it would be nice to design a common trainer that could be used to manage all my model training workflows.

from abc import abstractmethod

from pydantic import BaseModel, Field, ConfigDict

# torch imports
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Optimizer

class BaseTrainerConfig(BaseModel):
    # necessary because the torch.optim.Optimizer is not a pydantic model
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # these are the data hyperparameters
    vocab_size: int = Field(..., description="Size of the vocabulary")
    context_size: int = Field(..., description="Context window size")
    
    # training job hyperparameters
    max_iters: int = Field(..., description="Max number of training steps")
    eval_interval: int = Field(...,  description="Interval for making a run of evaluation")
    eval_iter: int = Field(..., description="Max number of eval steps")

    # model training hyperparameters
    n_embeddings: int = Field(..., description="Embedding size")
    n_heads: int = Field(..., description="Number of attention heads")
    n_layer: int = Field(..., description="Number of layers")
    dropout: float = Field(..., description="Amount of dropout on the net")
    learning_rate: float = Field(..., description="Optimizer learning rate")
    criterion: Optimizer = Field(..., description="Optimizer")

class BaseTrainer(BaseModel):
    """Base Trainer class, implements the abstract methods for the trainers"""
    # necessary because the torch.nn.Module and torch.utils.data.Datasert are not a pydantic models
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: nn.Module = Field(..., description="Model targeted to train.")
    dataset: Dataset = Field(..., description="Torch Dataset that will be used to train the model")
    config: BaseTrainerConfig = Field(..., description="Training run configuration hyperparamters")

    @abstractmethod
    def train(self):
        raise NotImplementedError
