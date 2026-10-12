import unittest

import torch

from src.models.transformer import Transformer
from src.trainer import grpo_math

VOCAB_SIZE, EMB_DIM, CONTEXT_SIZE = 50, 32, 16


def _logprobs(model, seqs):
    logits, _ = model(seqs[:, :-1], targets=None)
    return grpo_math.sequence_logprobs(logits, seqs[:, 1:])


class TestPolicyRatio(unittest.TestCase):
    """GRPO's ratio exp(new - old) must be exactly 1 before any weight update. It was
    ~14 once because dropout (incl. SDPA's dropout_p) stayed active in the log-prob passes."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = Transformer(VOCAB_SIZE, EMB_DIM, CONTEXT_SIZE, n_layers=2, n_heads=4)
        self.seqs = torch.randint(0, VOCAB_SIZE, (4, CONTEXT_SIZE))
        self.mask = torch.ones(4, CONTEXT_SIZE - 1, dtype=torch.bool)

    def test_ratio_is_one_in_eval_mode(self):
        self.model.eval()
        with torch.no_grad():
            old = _logprobs(self.model, self.seqs)
        new = _logprobs(self.model, self.seqs)
        adv = torch.randn(4)
        _, metrics = grpo_math.grpo_loss(old, new, old, adv, self.mask, clip_epsilon=0.2, kl_coef=0.04)
        self.assertAlmostEqual(metrics["mean_ratio"], 1.0, places=5)
        self.assertEqual(metrics["clip_fraction"], 0.0)
        self.assertAlmostEqual(metrics["mean_kl"], 0.0, places=5)

    def test_train_mode_is_noisy(self):
        """Guards the test above: dropout must actually perturb log-probs in train mode."""
        self.model.train()
        with torch.no_grad():
            a = _logprobs(self.model, self.seqs)
            b = _logprobs(self.model, self.seqs)
        self.assertFalse(torch.allclose(a, b, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
