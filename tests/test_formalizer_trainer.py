import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import tempfile
import shutil

# --- 1. MOCKS (Simulating your real Tokenizer and Model) ---

class MockTokenizer:
    """Simulates a tokenizer that converts text to random integers."""
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text, return_tensors="pt"):
        # Returns a random tensor of integers simulating token IDs
        # Length is proportional to text length for realism
        length = len(text)
        return torch.randint(0, self.vocab_size, (1, length))

class MockGPTModel(nn.Module):
    """Simulates a Transformer that returns correct shapes for logits/loss."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Simple linear layer to ensure gradients can flow
        self.net = nn.Linear(config.n_embeddings, config.vocab_size)
        # Embedding to handle input integers
        self.embed = nn.Embedding(config.vocab_size, config.n_embeddings)

    def forward(self, idx, targets=None):
        # idx shape: (Batch, Context)
        b, t = idx.shape
        
        # Fake forward pass
        x = self.embed(idx) # (B, T, n_embeddings)
        logits = self.net(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Create a dummy scalar loss that requires grad
            # We use .sum() so it actually depends on input for backward() check
            loss = logits.sum() * 0.0 + torch.tensor(1.0, requires_grad=True)

        return logits, loss

# --- IMPORT TRAINER CODE ---
try:
    from src.trainer.base import BaseTrainerConfig
    from src.trainer.formalizer_trainer import FormalizerDataset, FormalizerTrainer
except ImportError:
    pass

# --- 3. PYTEST FIXTURES ---

@pytest.fixture
def toy_corpus():
    """Creates a temporary text file simulating the LEAN dataset."""
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / "toy_corpus.txt"
    
    # Write dummy "math" text
    content = "definition topological_space " * 500  # Enough for a few batches
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    yield file_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def trainer_config():
    """Returns a config optimized for CPU/Testing."""
    return BaseTrainerConfig(
        vocab_size=100,
        context_size=16,
        max_iters=5,        # Very short run
        eval_interval=2,
        eval_iters=2,
        batch_size=4,
        n_embeddings=32,    # Small dims for speed
        n_heads=2,
        n_layer=2,
        dropout=0.1,
        learning_rate=1e-3,
        device="cpu"        # Force CPU for CI/CD compatibility
    )

# --- 4. THE ACTUAL TESTS ---

def test_dataset_chunking(toy_corpus):
    """Test if FormalizerDataset chunks text correctly."""
    tokenizer = MockTokenizer(vocab_size=100)
    context_size = 10
    
    dataset = FormalizerDataset(toy_corpus, tokenizer, context_size)
    
    # Basic assertions
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Test shapes
    x, y = dataset[0]
    assert x.shape == (context_size,), f"Expected input shape ({context_size},), got {x.shape}"
    assert y.shape == (context_size,), f"Expected target shape ({context_size},), got {y.shape}"
    
    # Test Target Shift (y should be x shifted by 1)
    # Note: In our mock, data is random, but indices must align. 
    # Real test checks logic: dataset.data[0:10] vs dataset.data[1:11]
    
    raw_data = dataset.data
    expected_x = raw_data[0:context_size]
    expected_y = raw_data[1:context_size+1]
    
    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)

def test_trainer_initialization(toy_corpus, trainer_config):
    """Test if the Trainer initializes components correctly."""
    tokenizer = MockTokenizer(vocab_size=trainer_config.vocab_size)
    dataset = FormalizerDataset(toy_corpus, tokenizer, trainer_config.context_size)
    model = MockGPTModel(trainer_config)
    
    trainer = FormalizerTrainer(
        model=model,
        train_dataset=dataset,
        val_dataset=dataset, # Reuse for test
        config=trainer_config
    )
    
    assert trainer.model is not None
    assert trainer.config.device == "cpu"

def test_training_loop_runs(toy_corpus, trainer_config):
    """Smoke test: Does the train() method run without crashing?"""
    tokenizer = MockTokenizer(vocab_size=trainer_config.vocab_size)
    dataset = FormalizerDataset(toy_corpus, tokenizer, trainer_config.context_size)
    model = MockGPTModel(trainer_config)
    
    trainer = FormalizerTrainer(
        model=model,
        train_dataset=dataset,
        val_dataset=dataset,
        config=trainer_config
    )
    
    # Ensure no errors are raised during the loop
    try:
        trainer.train()
    except Exception as e:
        pytest.fail(f"Training loop crashed with error: {e}")

def test_model_params_update(toy_corpus, trainer_config):
    """Verification: Do model parameters actually change after training?"""
    tokenizer = MockTokenizer(vocab_size=trainer_config.vocab_size)
    dataset = FormalizerDataset(toy_corpus, tokenizer, trainer_config.context_size)
    model = MockGPTModel(trainer_config)
    
    trainer = FormalizerTrainer(
        model=model,
        train_dataset=dataset,
        val_dataset=dataset,
        config=trainer_config
    )
    
    # Take a snapshot of a parameter (e.g., first layer weight)
    # Clone it to ensure we have a deep copy
    param_before = list(model.parameters())[0].clone()
    
    trainer.train()
    
    param_after = list(model.parameters())[0]
    
    # Check if they are different (gradients applied)
    assert not torch.equal(param_before, param_after), "Model parameters did not update! Optimizer failed."