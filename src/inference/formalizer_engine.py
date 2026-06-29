"""Inference engine for the Formalizer Model"""

import json
from pathlib import Path
from pydantic import Field

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
# ---------------------

class FormalizerInference(InferenceEngine):
    tokenizer: Tokenizer = Field(default=tokenizer, description="Model Tokenizer")

    def model_post_init(self, __context) -> None:
        """Loads model"""
        state_dict = torch.load(self.model_path, weights_only=self.weights_only)
        self.model.load_state_dict(state_dict)
        self.model.to("cuda")
        self.model.eval()

    def run(self, text: str, max_output_tokens: int = CONTEXT_SIZE):
        """Run the inference engine to generate text"""
        # tokenize text
        _logger.debug(f"Input text: {text}")
        tokens = torch.tensor(self.tokenizer.encode(text).ids).unsqueeze(0).to("cuda")
        _logger.debug("Input text tokens:")
        _logger.debug(tokens)

        # Generate output tokens
        out, _ = self.model.generate(tokens, max_output_tokens)
        out = out.squeeze(0).tolist()
        _logger.debug("Output tokens:")
        _logger.debug(out)
        # get generated text
        generated_text = self.tokenizer.decode(out)

        return generated_text