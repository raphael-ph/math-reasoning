"""Inference engine for the Formalizer Model"""

import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

import torch

# internal imports
from .base import InferenceEngine
from ..models.transformer import Transformer
from ..utils.logger import get_logger
from tokenizers import Tokenizer

# --- CONFIGURATION ----
## Paths
VOCAB_METADATA_PATH = "./data/corpus/metadata.json"
TOKENIZER_PATH = "./data/vocab/tokenizer_vocab.json"
## Vars
with open(VOCAB_METADATA_PATH, "rb") as file:
    f = file.read()
    vocab_config = json.loads(f)
CONTEXT_SIZE = vocab_config["context_size"]
VOCAB_SIZE = vocab_config["vocab_size"]
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
_logger = get_logger("formalizer_inference", level="DEBUG")
EOS_TOKEN = "<|endoftext|>"
# ---------------------

class GenerationConfig(BaseModel):
    """Sampling hyperparameters for FormalizerInference.run()"""
    max_output_tokens: int = Field(default=CONTEXT_SIZE, description="Max number of tokens to generate")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling threshold; None disables top-p filtering")
    temperature: float = Field(default=1.0, description="Sampling temperature; 0 selects greedy decoding")

class FormalizerInference(InferenceEngine):
    tokenizer: Tokenizer = Field(default=tokenizer, description="Model Tokenizer")
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig, description="Generation sampling hyperparameters")
    eos_token_id: Optional[int] = Field(default=None, description="EOS token id, resolved from the tokenizer on init")

    def model_post_init(self, __context) -> None:
        """Loads model"""
        state_dict = torch.load(self.model_path, weights_only=self.weights_only)
        self.model.load_state_dict(state_dict)
        self.model.to("cuda")
        self.model.eval()
        self.eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)

    def run(self, text: str):
        """Run the inference engine to generate text"""
        # tokenize text
        _logger.debug(f"Input text: {text}")
        tokens = torch.tensor(self.tokenizer.encode(text).ids).unsqueeze(0).to("cuda")
        _logger.debug("Input text tokens:")
        _logger.debug(tokens)

        # Generate output tokens
        out, _ = self.model.generate(
            tokens,
            self.generation_config.max_output_tokens,
            top_p=self.generation_config.top_p,
            temperature=self.generation_config.temperature,
            eos_token_id=self.eos_token_id,
        )
        out = out.squeeze(0).tolist()
        _logger.debug("Output tokens:")
        _logger.debug(out)
        # get generated text
        generated_text = self.tokenizer.decode(out)

        return generated_text