"""
play.py - Interactive Play & Match Spectator with Lookahead Search.

Allows you to:
1. Play a live match against your AI with configurable lookahead search depth (Depth 1..4).
2. Spectate two AI models dueling in real-time with visual board rendering.
3. 100% Guaranteed Legality via Action Masking and Alpha-Beta Search.
"""

import os
import time
import argparse
from typing import Tuple, Optional
import chess
import torch

from model import ChessGladiatorNet
from representation import board_to_tensor, index_to_move, move_to_index
from search import search_best_move


UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    ".": "·"
}


def render_board(board: chess.Board):
    """Prints a styled chessboard in the terminal with coordinate labels."""
    print("\n    a   b   c   d   e   f   g   h")
    print("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")
    for rank in range(7, -1, -1):
        row_str = f"{rank + 1} │"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            symbol = UNICODE_PIECES[piece.symbol()] if piece else UNICODE_PIECES["."]
            row_str += f" {symbol} │"
        print(row_str)
        if rank > 0:
            print("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
    print("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    print("    a   b   c   d   e   f   g   h\n")


def load_model(weights_path: str = "champion.pt") -> ChessGladiatorNet:
    """Loads a ChessGladiatorNet from file if it exists, otherwise returns a fresh model."""
    net = ChessGladiatorNet()
    if os.path.exists(weights_path):
        try:
            net.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")), strict=False)
            print(f"Loaded trained champion weights from '{weights_path}'")
        except Exception as e:
            print(f"Loaded weights with note: {e}")
    else:
        print(f"No checkpoint found at '{weights_path}' - using baseline model.")
    net.eval()
    return net


def get_ai_move_with_search(
    net: ChessGladiatorNet,
    board: chess.Board,
    depth: int = 3
) -> Tuple[chess.Move, int, int]:
    """
    Finds the best move using lookahead Alpha-Beta search guided by the neural net.
    Returns (move, eval_cp, nodes_searched).
    """
    if depth <= 0:
        # Depth 0: Direct neural net inference
        tensor = board_to_tensor(board)
        legal_indices = [move_to_index(m, turn=board.turn) for m in board.legal_moves]
        top_moves, promo_choice = net.predict_action(tensor, legal_actions=legal_indices, k=1)
        move = index_to_move(top_moves[0], board, promo_choice=promo_choice)
        return move, 0, 1

    move, eval_cp, nodes = search_best_move(board, model=net, depth=depth)
    return move, eval_cp, nodes


def play_human_vs_ai(model_path: str, human_color: chess.Color, depth: int = 3):
    """Interactive console game: Human vs Neural Engine."""
    ai = load_model(model_path)
    board = chess.Board()

    print(f"\nGame Started! AI Search Depth: {depth} ply | 100% Guaranteed Legal")
    print("Type moves in UCI format (e.g. 'e2e4', 'g1f3', 'e7e8q').")
    print("Type 'quit' to exit.\n")

    while not board.is_game_over():
        render_board(board)
        is_human = (board.turn == human_color)

        if is_human:
            user_input = input(f"Your move ({'White' if human_color == chess.WHITE else 'Black'}): ").strip()
            if user_input.lower() in ("quit", "exit"):
                print("Game exited.")
                return

            try:
                move = chess.Move.from_uci(user_input)
                if board.is_legal(move):
                    board.push(move)
                else:
                    legal_samples = [m.uci() for m in board.legal_moves][:8]
                    print(f"Illegal move. Examples of legal moves right now: {legal_samples}")
            except Exception:
                print("Invalid format. Please enter a valid UCI move like 'e2e4'.")
        else:
            print(f"AI is thinking (Depth {depth})...")
            start = time.time()
            ai_move, score_cp, nodes = get_ai_move_with_search(ai, board, depth=depth)
            calc_time = time.time() - start

            if ai_move is None:
                print("AI has no legal moves!")
                break

            score_str = f"{score_cp / 100.0:+.2f}"
            print(f"AI plays: {ai_move.uci()} | Eval: {score_str} | Nodes: {nodes} ({calc_time:.2f}s)")
            board.push(ai_move)

    render_board(board)
    print("\nGame Over! Result:", board.result())


def play_spectator_match(model_path: str, depth: int = 2, delay_sec: float = 0.5):
    """Spectate two AI instances dueling with lookahead search."""
    ai_white = load_model(model_path)
    ai_black = load_model(model_path)
    board = chess.Board()

    turn = 0
    max_turns = 100
    print(f"Starting Spectator Match (Depth {depth})!")

    while not board.is_game_over() and turn < max_turns:
        turn += 1
        render_board(board)
        color_str = "White" if board.turn == chess.WHITE else "Black"
        net = ai_white if board.turn == chess.WHITE else ai_black

        move, score_cp, nodes = get_ai_move_with_search(net, board, depth=depth)
        if move is None:
            print(f"{color_str} has no legal moves on turn {turn}!")
            return

        score_str = f"{score_cp / 100.0:+.2f}"
        print(f"Turn {turn:2d} | {color_str} plays: {move.uci()} | Eval: {score_str} (Nodes: {nodes})")
        board.push(move)
        time.sleep(delay_sec)

    render_board(board)
    print(f"\nMatch concluded on Turn {turn}! Result: {board.result()}")


def main():
    parser = argparse.ArgumentParser(description="Play against or watch the Search-Guided Chess AI")
    parser.add_argument("--mode", choices=["human-white", "human-black", "spectate"], default="human-white")
    parser.add_argument("--weights", type=str, default="champion.pt")
    parser.add_argument("--depth", type=int, default=3, help="Lookahead search depth (default: 3)")
    parser.add_argument("--speed", type=float, default=0.4, help="Seconds between moves in spectate mode")
    args = parser.parse_args()

    if args.mode == "human-white":
        play_human_vs_ai(args.weights, chess.WHITE, depth=args.depth)
    elif args.mode == "human-black":
        play_human_vs_ai(args.weights, chess.BLACK, depth=args.depth)
    else:
        play_spectator_match(args.weights, depth=args.depth, delay_sec=args.speed)


if __name__ == "__main__":
    main()
