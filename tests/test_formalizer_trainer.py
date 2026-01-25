import sys
import pytest
import torch
import math
import itertools
from unittest.mock import MagicMock, patch

# --- 1. MOCK DEPENDENCIES ---
mock_base = MagicMock()
sys.modules['src.trainer.base'] = mock_base

from pydantic import BaseModel, ConfigDict

class MockBaseTrainer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: torch.nn.Module
    train_dataset: object
    val_dataset: object
    config: object
    def train(self): pass
    def model_post_init(self, __context): pass

mock_base.BaseTrainer = MockBaseTrainer
sys.modules['src.utils.logger'] = MagicMock()
sys.modules['src.preprocessing.hf_tokenizer'] = MagicMock()
sys.modules['src.preprocessing.fim'] = MagicMock()

# --- 2. IMPORT SUT ---
import src.trainer.formalizer_trainer as sut
from src.trainer.formalizer_trainer import FormalizerDataset, FormalizerTrainer

# --- FIXTURES ---

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    mock_encoding = MagicMock()
    mock_encoding.ids = list(range(11)) 
    tokenizer.encode.return_value = mock_encoding
    return tokenizer

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.device = "cpu"
    config.batch_size = 2
    config.learning_rate = 1e-4
    config.max_iters = 10
    config.eval_interval = 5
    config.checkpoint_interval = 5
    config.eval_iters = 2
    config.model_dump.return_value = {"param": "value"}
    return config

@pytest.fixture
def mock_model():
    model = MagicMock(spec=torch.nn.Module)
    # Return (logits, loss)
    model.return_value = (torch.randn(2, 10, 10), torch.tensor(2.0, requires_grad=True))
    model.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]
    return model

@pytest.fixture
def mock_dataset():
    ds = MagicMock() 
    ds.__len__.return_value = 10
    ds.__getitem__.return_value = (torch.zeros(10), torch.zeros(10))
    return ds

# --- DATASET TESTS (Unchanged) ---

def test_dataset_initialization_and_len(mock_tokenizer, tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    content = "Sample 1 <|endoftext|> Sample 2 <|endoftext|> \n <|endoftext|>" 
    corpus_file.write_text(content, encoding="utf-8")

    dataset = FormalizerDataset(corpus_file, mock_tokenizer, context_size=10)

    assert len(dataset) == 2
    assert dataset.samples == ["Sample 1", "Sample 2"]

def test_dataset_getitem_logic(mock_tokenizer, tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("Test Data<|endoftext|>", encoding="utf-8")
    
    with patch(f"{sut.__name__}.apply_line_level_fim") as mock_fim:
        mock_fim.return_value = "FIM_Applied_Text"
        dataset = FormalizerDataset(corpus_file, mock_tokenizer, context_size=10)
        input_ids, target_ids = dataset[0]

        mock_fim.assert_called_with("Test Data")
        expected_input = torch.tensor(list(range(10)), dtype=torch.long)
        expected_target = torch.tensor(list(range(1, 11)), dtype=torch.long)

        assert torch.equal(input_ids, expected_input)
        assert torch.equal(target_ids, expected_target)

# --- TRAINER TESTS ---

class TestFormalizerTrainer:

    def setup_trainer(self, config, model, dataset):
        return FormalizerTrainer(
            model=model,
            config=config,
            train_dataset=dataset,
            val_dataset=dataset
        )

    def test_model_post_init(self, mock_config, mock_model, mock_dataset):
        trainer = self.setup_trainer(mock_config, mock_model, mock_dataset)
        trainer.model_post_init(None)
        mock_model.to.assert_called_with(mock_config.device)

    def test_setup_dataloaders(self, mock_config, mock_model, mock_dataset):
        trainer = self.setup_trainer(mock_config, mock_model, mock_dataset)
        
        with patch(f"{sut.__name__}.DataLoader") as MockDL:
            trainer._setup_dataloaders()
            assert trainer._train_dataloader is not None
            assert trainer._val_dataloader is not None
            MockDL.assert_any_call(
                dataset=trainer.train_dataset,
                batch_size=mock_config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

    def test_train_loop_full_flow(self, mock_config, mock_model, mock_dataset):
        trainer = self.setup_trainer(mock_config, mock_model, mock_dataset)
        
        # --- TEST CONFIGURATION FIX ---
        # range(5) provides indices 0, 1, 2, 3, 4.
        # We set checkpoint_interval=2 so it triggers at i=2 and i=4.
        mock_config.max_iters = 5
        mock_config.checkpoint_interval = 2
        mock_config.eval_interval = 100 # Ensure we skip eval to isolate checkpoint logic
        
        dummy_x = torch.zeros((2, 10))
        dummy_y = torch.zeros((2, 10))
        mock_dl = MagicMock()
        mock_dl.__iter__.return_value = itertools.cycle([(dummy_x, dummy_y)])
        
        with patch(f"{sut.__name__}.DataLoader", return_value=mock_dl):
            with patch(f"{sut.__name__}.mlflow") as mock_mlflow:
                with patch(f"{sut.__name__}.time"):
                    trainer._estimate_loss = MagicMock(return_value={
                        'train_loss': 1.0, 'val_loss': 1.5, 'train_ppl': 2.7, 'val_ppl': 4.5
                    })
                    
                    trainer.train()
                    
                    mock_mlflow.set_tracking_uri.assert_called()
                    
                    # Now this will succeed because i=2 and i=4 triggered the save
                    mock_mlflow.pytorch.log_model.assert_called()

    def test_stop_iteration_handling(self, mock_config, mock_model, mock_dataset):
        """
        Verifies that if the iterator is exhausted (StopIteration),
        the trainer catches it, re-initializes the iterator, and continues.
        """
        trainer = self.setup_trainer(mock_config, mock_model, mock_dataset)
        trainer._setup_dataloaders = MagicMock() 

        mock_dl = MagicMock()
        trainer._train_dataloader = mock_dl
        
        dummy_x = torch.zeros((1, 1))
        dummy_y = torch.zeros((1, 1))
        
        # Generator 1: Yields one item, then crashes.
        def iter_crashes():
            yield (dummy_x, dummy_y)
            # Raise StopIteration implicitly by finishing
        
        # Generator 2: Infinite yielding
        def iter_infinite():
            while True:
                yield (dummy_x, dummy_y)

        # We set side_effect as a LIST of generators.
        # 1st call to iter(loader) gets iter_crashes()
        # 2nd call to iter(loader) gets iter_infinite()
        mock_dl.__iter__.side_effect = [iter_crashes(), iter_infinite()]

        # max_iters=2 means we need 2 batches.
        # Batch 0: From iter_crashes (OK)
        # Batch 1: iter_crashes raises StopIteration -> Catch -> Re-init -> Get from iter_infinite (OK)
        mock_config.max_iters = 2
        
        with patch(f"{sut.__name__}.mlflow"):
            with patch("torch.optim.AdamW"):
                trainer.train()

        # Should have called iter() twice (initial + recovery)
        assert mock_dl.__iter__.call_count == 2

    def test_estimate_loss_logic(self, mock_config, mock_model, mock_dataset):
        """
        Verifies _estimate_loss handles StopIteration across splits (train/val)
        """
        trainer = self.setup_trainer(mock_config, mock_model, mock_dataset)
        mock_config.eval_iters = 2
        
        mock_dl = MagicMock()
        # Use same mock for both to simplify verification
        trainer._train_dataloader = mock_dl
        trainer._val_dataloader = mock_dl
        
        dummy_x = torch.zeros((1, 1))
        dummy_y = torch.zeros((1, 1))
        
        mock_model.return_value = (None, torch.tensor(1.0))
        
        # We need to simulate the sequence of calls to iter(loader):
        # 1. 'train' split start -> Returns iter that yields 1 item then stops.
        # 2. 'train' split recovery -> Returns iter that yields items.
        # 3. 'val' split start -> Returns iter that yields items.
        
        def iter_crashes_after_one():
            yield (dummy_x, dummy_y)
            # Finish -> StopIteration
            
        def iter_infinite():
            while True:
                yield (dummy_x, dummy_y)

        mock_dl.__iter__.side_effect = [
            iter_crashes_after_one(), # Train Start
            iter_infinite(),          # Train Recovery
            iter_infinite()           # Val Start
        ]
        
        results = trainer._estimate_loss()
        
        # Verify calls
        assert results['train_loss'] == 1.0
        # Check perplexity math
        assert math.isclose(results['train_ppl'], math.exp(1.0), rel_tol=1e-4)
        # We expect 3 calls to iter(): Train(Crash), Train(Recover), Val
        assert mock_dl.__iter__.call_count == 3

    def test_checkpoint_saving_signature(self, mock_config, mock_model, mock_dataset):
        trainer = self.setup_trainer(mock_config, mock_model, mock_dataset)
        
        # Save every step
        mock_config.max_iters = 2
        mock_config.eval_interval = 10 
        mock_config.checkpoint_interval = 1 
        
        dummy_x = torch.zeros((2, 10))
        dummy_y = torch.zeros((2, 10))
        
        mock_dl = MagicMock()
        mock_dl.__iter__.return_value = itertools.cycle([(dummy_x, dummy_y)])
        
        trainer._setup_dataloaders = MagicMock()
        trainer._train_dataloader = mock_dl
        
        with patch(f"{sut.__name__}.mlflow") as mock_mlflow:
            with patch(f"{sut.__name__}.infer_signature") as mock_sig:
                # We skip eval loop via eval_interval=10, so estimate_loss isn't called
                trainer.train()
                
                # Check that infer_signature was called
                mock_sig.assert_called()
                # Check that log_model was called
                mock_mlflow.pytorch.log_model.assert_called()