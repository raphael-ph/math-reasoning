import torch
import unittest
import src.models.transformer as transformer
from src.models.transformer import *

transformer.DEVICE = "cpu"

# --- Test Hyperparameters ---
VOCAB_SIZE    = 100
EMB_DIM       = 32
CONTEXT_SIZE  = 16
N_HEADS       = 4
N_LAYERS      = 2
HEAD_SIZE     = EMB_DIM // N_HEADS  # 8
BATCH_SIZE    = 2

class TestAttentionHead(unittest.TestCase):

    def setUp(self):
        self.head = AttentionHead(EMB_DIM, HEAD_SIZE, CONTEXT_SIZE)
        self.x = torch.randn(BATCH_SIZE, CONTEXT_SIZE, EMB_DIM)

    def test_output_shape(self):
        out = self.head(self.x)
        self.assertEqual(out.shape, (BATCH_SIZE, CONTEXT_SIZE, HEAD_SIZE))

    def test_shorter_sequence(self):
        """T < context_size should work fine (tests the [:T, :T] tril slice)"""
        x_short = torch.randn(BATCH_SIZE, 5, EMB_DIM)
        out = self.head(x_short)
        self.assertEqual(out.shape, (BATCH_SIZE, 5, HEAD_SIZE))

    def test_rope_output_shape(self):
        cos, sin = self.head.precompute_freqs_cis(CONTEXT_SIZE)
        self.assertEqual(cos.shape, (CONTEXT_SIZE, HEAD_SIZE))
        self.assertEqual(sin.shape, (CONTEXT_SIZE, HEAD_SIZE))

    def test_rope_preserves_norm(self):
        """RoPE should not change the norm of Q and K"""
        Q = torch.randn(BATCH_SIZE, CONTEXT_SIZE, HEAD_SIZE)
        K = torch.randn(BATCH_SIZE, CONTEXT_SIZE, HEAD_SIZE)
        cos, sin = self.head.precompute_freqs_cis(CONTEXT_SIZE)
        Q_rot, K_rot = self.head.rope(Q, K, cos, sin)
        self.assertTrue(torch.allclose(Q.norm(dim=-1), Q_rot.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(K.norm(dim=-1), K_rot.norm(dim=-1), atol=1e-5))

    def test_causal_mask(self):
        """Future tokens must not influence past tokens — check attention is causal"""
        self.head.eval()
        x1 = torch.randn(1, CONTEXT_SIZE, EMB_DIM)
        x2 = x1.clone()
        x2[:, -1, :] = torch.randn(EMB_DIM)  # only change the last token
        out1 = self.head(x1)
        out2 = self.head(x2)
        # all positions except the last should be identical
        self.assertTrue(torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6))


class TestMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        self.mha = MultiHeadAttention(N_HEADS, EMB_DIM, HEAD_SIZE, CONTEXT_SIZE)
        self.x = torch.randn(BATCH_SIZE, CONTEXT_SIZE, EMB_DIM)

    def test_output_shape(self):
        out = self.mha(self.x)
        self.assertEqual(out.shape, (BATCH_SIZE, CONTEXT_SIZE, EMB_DIM))


class TestBlock(unittest.TestCase):

    def setUp(self):
        self.block = Block(N_HEADS, EMB_DIM, CONTEXT_SIZE)
        self.x = torch.randn(BATCH_SIZE, CONTEXT_SIZE, EMB_DIM)

    def test_output_shape(self):
        out = self.block(self.x)
        self.assertEqual(out.shape, (BATCH_SIZE, CONTEXT_SIZE, EMB_DIM))

    def test_residual_connection(self):
        """Output should not be identical to input — residual adds, not replaces"""
        out = self.block(self.x)
        self.assertFalse(torch.allclose(out, self.x))


class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.model = Transformer(VOCAB_SIZE, EMB_DIM, CONTEXT_SIZE, N_LAYERS, N_HEADS)
        self.idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_SIZE))
        self.targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_SIZE))

    def test_forward_no_targets(self):
        logits, loss = self.model(self.idx)
        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (BATCH_SIZE, CONTEXT_SIZE, VOCAB_SIZE))

    def test_forward_with_targets(self):
        logits, loss = self.model(self.idx, self.targets)
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)

    def test_loss_decreases(self):
        """A single gradient step should reduce the loss"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        _, loss_before = self.model(self.idx, self.targets)
        loss_before.backward()
        optimizer.step()
        optimizer.zero_grad()
        _, loss_after = self.model(self.idx, self.targets)
        self.assertLess(loss_after.item(), loss_before.item())

    def test_generate_shape(self):
        self.model.eval()
        idx_start = torch.randint(0, VOCAB_SIZE, (1, 1))
        out, _ = self.model.generate(idx_start, max_new_tokens=10)
        self.assertEqual(out.shape, (1, 11))  # 1 seed token + 10 generated

    def test_generate_stays_within_vocab(self):
        self.model.eval()
        idx_start = torch.randint(0, VOCAB_SIZE, (1, 1))
        out, _ = self.model.generate(idx_start, max_new_tokens=10)
        self.assertTrue((out >= 0).all() and (out < VOCAB_SIZE).all())

unittest.main(argv=[""], exit=False, verbosity=2)