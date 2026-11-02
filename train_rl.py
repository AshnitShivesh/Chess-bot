"""
train_rl.py - Surgical Gradient Training & Policy Learning Engine.

Replaces blind random mutation with PyTorch Calculus (Backpropagation):
- Every position encountered in self-play is recorded on the PyTorch autograd tape.
- Calculates exact partial derivatives (dL/dW) for all 790,000 weights.
- Multi-Action Cross-Entropy: Simultaneously pushes UP the logits of all legal moves
  and pushes DOWN all illegal moves.
- Weights surgically retain which pieces can legally move.
"""

import os
import json
import time
import random
import argparse
from typing import List, Tuple
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ChessGladiatorNet
from representation import board_to_tensor, move_to_index, index_to_move


def get_legal_target_distribution(board: chess.Board) -> Tuple[torch.Tensor, List[int]]:
    """
    Creates a target probability distribution over all 4,096 canonical actions:
    - Uniform distribution across all legal moves (1 / num_legal)
    - 0.0 for all illegal moves
    """
    target = torch.zeros(4096, dtype=torch.float32)
    legal_indices = []

    for move in board.legal_moves:
        idx = move_to_index(move, turn=board.turn)
        legal_indices.append(idx)

    num_legal = len(legal_indices)
    if num_legal > 0:
        val = 1.0 / num_legal
        for idx in legal_indices:
            target[idx] = val

    return target, legal_indices


def generate_self_play_batch(
    model: ChessGladiatorNet,
    batch_size: int = 64,
    max_game_length: int = 60
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Plays self-play matches using the neural network to collect (state, target) training pairs.
    Mixes opening exploration so the network encounters diverse board states.
    """
    states: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        while len(states) < batch_size:
            board = chess.Board()

            # Random opening ply (0..6) to ensure wide coverage across the game tree
            random_opening_plies = random.randint(0, 6)
            for _ in range(random_opening_plies):
                if board.is_game_over():
                    break
                rand_move = random.choice(list(board.legal_moves))
                board.push(rand_move)

            # Play turns using the neural network
            turns = 0
            while not board.is_game_over() and turns < max_game_length and len(states) < batch_size:
                turns += 1
                tensor = board_to_tensor(board)
                target, legal_indices = get_legal_target_distribution(board)

                if len(legal_indices) == 0:
                    break

                states.append(tensor)
                targets.append(target)

                # Query network for next move
                move_logits, promo_logits = model(tensor)

                # Pick candidate move
                top_candidates = torch.topk(move_logits, k=20).indices.tolist()
                promo_choice = int(torch.argmax(promo_logits).item())

                chosen_move = None
                for c_idx in top_candidates:
                    cand = index_to_move(c_idx, board, promo_choice=promo_choice)
                    if board.is_legal(cand):
                        chosen_move = cand
                        break

                if chosen_move is None:
                    chosen_move = random.choice(list(board.legal_moves))

                board.push(chosen_move)

    states_tensor = torch.stack(states[:batch_size])
    targets_tensor = torch.stack(targets[:batch_size])
    return states_tensor, targets_tensor


def evaluate_legality(
    model: ChessGladiatorNet,
    test_positions: int = 250
) -> Tuple[float, float]:
    """
    Evaluates model on a battery of diverse game positions across depths 0 to 40.
    Returns (top1_legality_pct, top5_legality_pct).
    """
    model.eval()
    top1_legal = 0
    top5_legal = 0
    total = 0

    with torch.no_grad():
        while total < test_positions:
            board = chess.Board()
            depth = random.randint(0, 40)
            for _ in range(depth):
                if board.is_game_over():
                    break
                board.push(random.choice(list(board.legal_moves)))

            if board.is_game_over():
                continue

            total += 1
            tensor = board_to_tensor(board)
            top_indices, promo = model.predict_action(tensor, k=5)

            # Top 1 check
            first_move = index_to_move(top_indices[0], board, promo_choice=promo)
            if board.is_legal(first_move):
                top1_legal += 1

            # Top 5 check
            any_legal = False
            for idx in top_indices:
                cand = index_to_move(idx, board, promo_choice=promo)
                if board.is_legal(cand):
                    any_legal = True
                    break
            if any_legal:
                top5_legal += 1

    top1_pct = (top1_legal / total) * 100.0
    top5_pct = (top5_legal / total) * 100.0
    return top1_pct, top5_pct


def train_gradient_retention(
    target_top5: float = 99.5,
    target_top1: float = 90.0,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_steps: int = 600,
    eval_interval: int = 25,
    save_dir: str = "."
):
    """
    Runs the gradient retention training loop to push raw brain legality toward 100%.
    """
    print(f"\n{'='*75}")
    print(f"PUSHING THE RAW NEURAL NETWORK TO ITS LIMITS (NO MASKING)")
    print(f"Targets -> Top-5: {target_top5}% | Top-1: {target_top1}%")
    print(f"Max Steps: {max_steps} | Batch Size: {batch_size} | Initial LR: {learning_rate}")
    print(f"Device: CPU | Optimizer: AdamW + Cosine Annealing")
    print(f"{'='*75}\n")

    model = ChessGladiatorNet()

    # Load existing champion to continue training from where it left off
    champ_path = os.path.join(save_dir, "champion.pt")
    if os.path.exists(champ_path):
        try:
            model.load_state_dict(torch.load(champ_path, map_location="cpu"))
            print(f"Loaded existing weights from '{champ_path}' - continuing evolution!")
        except Exception:
            print("Starting fresh initialization.")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)

    history = []
    best_combined_score = 0.0
    start_time = time.time()

    for step in range(1, max_steps + 1):
        model.train()
        states, targets = generate_self_play_batch(model, batch_size=batch_size)

        optimizer.zero_grad()
        move_logits, _ = model(states)

        # Multi-Action Cross-Entropy Loss
        log_probs = F.log_softmax(move_logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == 1:
            top1_pct, top5_pct = evaluate_legality(model, test_positions=250)
            elapsed = time.time() - start_time

            stat = {
                "step": step,
                "loss": round(loss.item(), 4),
                "top1_legality": round(top1_pct, 2),
                "top5_legality": round(top5_pct, 2),
                "elapsed_sec": round(elapsed, 1)
            }
            history.append(stat)

            log_path = os.path.join(save_dir, "training_log.json")
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)

            print(
                f"Step {step:4d}/{max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Top-1: {top1_pct:5.1f}% | "
                f"Top-5: {top5_pct:5.1f}% | "
                f"Time: {elapsed:4.0f}s"
            )

            # Combined score to save best checkpoint
            combined = (top5_pct * 0.6) + (top1_pct * 0.4)
            if combined > best_combined_score:
                best_combined_score = combined
                torch.save(model.state_dict(), champ_path)

            # Check if both high bars are reached
            if top5_pct >= target_top5 and top1_pct >= target_top1:
                print(f"\n{'*'*75}")
                print(f"🎯 PHENOMENAL! Raw Brain reached Top-5: {top5_pct:.2f}% & Top-1: {top1_pct:.2f}% on Step {step}!")
                print(f"Total Time: {elapsed:.1f}s | Weights saved to: {champ_path}")
                print(f"{'*'*75}\n")
                break


def main():
    parser = argparse.ArgumentParser(description="Push Raw Chess Neural Net Legality")
    parser.add_argument("--top5-target", type=float, default=99.5, help="Target Top-5 legality percentage")
    parser.add_argument("--top1-target", type=float, default=90.0, help="Target Top-1 legality percentage")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=500, help="Maximum steps to run")
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()

    train_gradient_retention(
        target_top5=args.top5_target,
        target_top1=args.top1_target,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        eval_interval=args.eval_every
    )


if __name__ == "__main__":
    main()
