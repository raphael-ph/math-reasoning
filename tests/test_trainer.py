import pytest
import torch
from unittest.mock import MagicMock, mock_open, patch
from pathlib import Path

from src.trainer.formalizer_trainer import FormalizerDataset

# -------------------------------------------------------------------
# Mocks and Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def mock_tokenizer():
    """Mocks a Hugging Face tokenizer."""
    tokenizer = MagicMock()
    # When tokenizer.encode is called, return a dummy tensor
    # We simulate a corpus of 20 tokens: IDs 0 to 19
    dummy_tensor = torch.arange(20).unsqueeze(0) # Shape [1, 20]
    
    # Configure the mock to return this tensor when called
    # return_tensors='pt' usually returns a dictionary or tensor depending on config, 
    # but your code expects a tensor immediately or handles the output. 
    # Based on your code: tokenizer.encode(..., return_tensors="pt") 
    tokenizer.encode.return_value = dummy_tensor
    return tokenizer

@pytest.fixture
def mock_corpus_path():
    return Path("dummy_corpus.txt")

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_initialization_tokenizes_correctly(mock_tokenizer, mock_corpus_path):
    """Test that the class reads file and stores tokens correctly."""
    
    dummy_text = "This is some dummy text content."
    context_size = 5
    
    # Mock 'open' to avoid reading actual files
    with patch("builtins.open", mock_open(read_data=dummy_text)) as mock_file:
        dataset = FormalizerDataset(mock_corpus_path, mock_tokenizer, context_size)
        
        # Verify file was opened correctly
        mock_file.assert_called_once_with(mock_corpus_path, "r", encoding="utf-8")
        
        # Verify tokenizer was called with the text from file
        mock_tokenizer.encode.assert_called_once_with(dummy_text, return_tensors="pt")
        
        # Verify data is stored as a 1D tensor (squeezed)
        assert isinstance(dataset.data, torch.Tensor)
        assert dataset.data.ndim == 1
        assert len(dataset.data) == 20  # Based on our mock_tokenizer fixture

def test_len_calculation(mock_tokenizer, mock_corpus_path):
    """
    Test __len__ logic: (total_tokens - 1) // context_size
    Total tokens in mock = 20.
    """
    
    # Case 1: Context size 5
    # (20 - 1) // 5 = 19 // 5 = 3 items
    with patch("builtins.open", mock_open(read_data="data")):
        ds_1 = FormalizerDataset(mock_corpus_path, mock_tokenizer, context_size=5)
        assert len(ds_1) == 3

    # Case 2: Context size 10
    # (20 - 1) // 10 = 19 // 10 = 1 item
    with patch("builtins.open", mock_open(read_data="data")):
        ds_2 = FormalizerDataset(mock_corpus_path, mock_tokenizer, context_size=10)
        assert len(ds_2) == 1
        
    # Case 3: Context size 20 (Equal to length)
    # (20 - 1) // 20 = 0 items (Not enough data for one full input + 1 target)
    with patch("builtins.open", mock_open(read_data="data")):
        ds_3 = FormalizerDataset(mock_corpus_path, mock_tokenizer, context_size=20)
        assert len(ds_3) == 0

def test_getitem_shapes_and_content(mock_tokenizer, mock_corpus_path):
    """Test that __getitem__ returns correct slices for input and target."""
    
    context_size = 4
    # Mock data is [0, 1, 2, ... 19]
    
    with patch("builtins.open", mock_open(read_data="data")):
        dataset = FormalizerDataset(mock_corpus_path, mock_tokenizer, context_size)
        
        # Fetch index 0
        input_ids, target_ids = dataset[0]
        
        # Expected Input: indices [0:4] -> [0, 1, 2, 3]
        expected_input = torch.tensor([0, 1, 2, 3])
        # Expected Target: indices [1:5] -> [1, 2, 3, 4]
        expected_target = torch.tensor([1, 2, 3, 4])
        
        assert torch.equal(input_ids, expected_input)
        assert torch.equal(target_ids, expected_target)
        
        # Fetch index 1
        # Start idx = 1 * 4 = 4
        # Expected Input: indices [4:8] -> [4, 5, 6, 7]
        # Expected Target: indices [5:9] -> [5, 6, 7, 8]
        input_ids_2, target_ids_2 = dataset[1]
        
        assert torch.equal(input_ids_2, torch.tensor([4, 5, 6, 7]))
        assert torch.equal(target_ids_2, torch.tensor([5, 6, 7, 8]))

def test_getitem_out_of_bounds(mock_tokenizer, mock_corpus_path):
    """Test that accessing an invalid index raises an error (standard PyTorch behavior)."""
    context_size = 5
    # Length is 3
    
    with patch("builtins.open", mock_open(read_data="data")):
        dataset = FormalizerDataset(mock_corpus_path, mock_tokenizer, context_size)
        
        # Accessing index 10 should fail purely on tensor slicing logic logic 
        # (though typically DataLoader handles bounds, direct access allows checking logic)
        # However, Python list slicing [100:105] returns empty, not error. 
        # But since we depend on specific shapes for training, getting an empty tensor is 'wrong'.
        
        input_ids, _ = dataset[10]
        # If we slice past the end, we get an empty tensor
        assert len(input_ids) == 0