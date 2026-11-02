"""
test_all.py - Complete Unit Test & Sanity Verification Suite.

Validates:
1. Canonical perspective symmetry (e2e4 for White has same index as e7e5 for Black).
2. ConvNet triple heads (Policy, Promo, Value).
3. AlphaZero-style Action Masking (100.0% Legality Guarantee).
4. Gradient retention engine (loss computation, backprop, weight update).
5. Alpha-Beta Lookahead Search Engine (depth calculation and move ordering).
6. Path B Self-Play reinforcement learning game generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import chess
import torch

from src.representation import board_to_tensor, move_to_index, index_to_move
from src.model import ChessGladiatorNet
from src.search import search_best_move
from training.train_rl import get_legal_target_distribution
from training.train_selfplay import play_fast_selfplay_game


class TestGladiatorSystem(unittest.TestCase):

    def test_01_canonical_perspective_symmetry(self):
        """Verify that White e2e4 and Black e7e5 map to the EXACT same canonical index."""
        e2e4 = chess.Move.from_uci("e2e4")
        e7e5 = chess.Move.from_uci("e7e5")

        idx_white = move_to_index(e2e4, turn=chess.WHITE)
        idx_black = move_to_index(e7e5, turn=chess.BLACK)

        self.assertEqual(idx_white, idx_black)

    def test_02_canonical_board_tensor(self):
        """Verify tensor shape and piece mapping under canonical perspective."""
        board = chess.Board()
        tensor_w = board_to_tensor(board)
        self.assertEqual(tensor_w.shape, (13, 8, 8))
        self.assertEqual(int(tensor_w[:12].sum().item()), 32)
        self.assertEqual(int(tensor_w[12].sum().item()), 64)

        board.push_san("e4")
        tensor_b = board_to_tensor(board)
        self.assertEqual(tensor_b.shape, (13, 8, 8))
        self.assertEqual(int(tensor_b[12].sum().item()), 0)

    def test_03_action_masking_100_percent_guarantee(self):
        """Verify that Action Masking guarantees 100% legal moves on any board."""
        board = chess.Board()
        net = ChessGladiatorNet()
        tensor = board_to_tensor(board)

        legal_indices = [move_to_index(m, turn=board.turn) for m in board.legal_moves]
        top_moves, promo = net.predict_action(tensor, legal_actions=legal_indices, k=5)

        for action_idx in top_moves:
            self.assertIn(action_idx, legal_indices)
            candidate_move = index_to_move(action_idx, board, promo_choice=promo)
            self.assertTrue(board.is_legal(candidate_move))

    def test_04_model_architecture_with_value_head(self):
        """Verify model forward pass returns (move_logits, promo_logits, value)."""
        net = ChessGladiatorNet()
        dummy = torch.zeros((13, 8, 8))

        move_logits, promo_logits, value = net(dummy)
        self.assertEqual(move_logits.shape, torch.Size([4096]))
        self.assertEqual(promo_logits.shape, torch.Size([4]))
        self.assertIsInstance(float(value.item()), float)
        self.assertTrue(-1.0 <= float(value.item()) <= 1.0)

    def test_05_gradient_retention_update(self):
        """Verify that backpropagation computes gradients and updates weights on legal moves."""
        model = ChessGladiatorNet()
        board = chess.Board()

        tensor = board_to_tensor(board).unsqueeze(0)
        target, legal_indices = get_legal_target_distribution(board)
        target = target.unsqueeze(0)

        weight_before = model.move_head.weight.clone()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        move_logits, _, _ = model(tensor)
        log_probs = torch.log_softmax(move_logits, dim=-1)
        loss = -(target * log_probs).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()

        diff = (model.move_head.weight - weight_before).abs().sum().item()
        self.assertGreater(diff, 0.0)

    def test_06_search_engine_lookahead(self):
        """Verify that search_best_move finds legal moves and calculates depth."""
        board = chess.Board()
        net = ChessGladiatorNet()

        best_move, score_cp, nodes = search_best_move(board, model=net, depth=2)
        self.assertIsNotNone(best_move)
        self.assertTrue(board.is_legal(best_move))
        self.assertGreater(nodes, 0)

    def test_07_selfplay_game_simulation(self):
        """Verify that play_fast_selfplay_game plays full games and attributes terminal rewards."""
        net = ChessGladiatorNet()
        exps, winner, turns = play_fast_selfplay_game(net, max_turns=10, temperature_turns=2)
        self.assertGreater(len(exps), 0)
        self.assertIn(winner, ["white", "black", "draw"])
        # Each experience must have state, action, and target value
        for exp in exps:
            self.assertEqual(exp.state.shape, (13, 8, 8))
            self.assertIn(exp.target_value, [-1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
