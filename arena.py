"""
arena.py - Gladiator Duel Arena & Soft-Fallback Survival Buffer.

Features:
- Canonical Perspective: Active player always sees the board from its own perspective.
- Action Masking (use_mask=True): 100.00% Guaranteed Legal Play.
- Soft Fallback (if use_mask=False): 3-strike tolerance for raw brain experiments.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import chess
import torch

from representation import board_to_tensor, index_to_move, move_to_index
from model import ChessGladiatorNet


PIECE_VALUES: Dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass
class MatchResult:
    winner: str                    # 'white', 'black', or 'draw'
    end_reason: str                # 'checkmate', 'disqualification', 'move_limit', 'stalemate', 'repetition'
    white_fitness: float
    black_fitness: float
    turns_played: int
    white_legal_moves: int
    white_attempted_moves: int
    black_legal_moves: int
    black_attempted_moves: int
    disqualification_note: Optional[str] = None


def compute_material(board: chess.Board, color: chess.Color) -> float:
    """Computes total piece value for a given color."""
    total = 0.0
    for square, piece in board.piece_map().items():
        if piece.color == color:
            total += PIECE_VALUES.get(piece.piece_type, 0.0)
    return total


def play_match(
    white_net: ChessGladiatorNet,
    black_net: ChessGladiatorNet,
    max_turns: int = 80,
    top_k: int = 5,
    use_mask: bool = True,
    max_strikes: int = 3
) -> MatchResult:
    """
    Executes a chess game between white_net and black_net.
    If use_mask=True (default), all moves are 100.0% guaranteed legal.
    """
    board = chess.Board()

    white_legal_moves = 0
    black_legal_moves = 0
    white_attempts = 0
    black_attempts = 0

    white_strikes = 0
    black_strikes = 0

    white_fitness = 0.0
    black_fitness = 0.0

    position_history: Dict[str, int] = {}

    turns = 0
    winner = "draw"
    end_reason = "move_limit"
    disqualification_note = None

    while not board.is_game_over() and turns < max_turns:
        turns += 1
        is_white_turn = (board.turn == chess.WHITE)
        current_net = white_net if is_white_turn else black_net

        if is_white_turn:
            white_attempts += 1
        else:
            black_attempts += 1

        # 1. Query neural net
        tensor = board_to_tensor(board)
        legal_indices = None
        if use_mask:
            legal_indices = [move_to_index(m, turn=board.turn) for m in board.legal_moves]

        top_move_indices, promo_choice = current_net.predict_action(
            tensor, legal_actions=legal_indices, k=top_k
        )

        # 2. Check candidate moves for legality
        legal_move_found = None
        for action_idx in top_move_indices:
            candidate_move = index_to_move(action_idx, board, promo_choice=promo_choice)
            if board.is_legal(candidate_move):
                legal_move_found = candidate_move
                break

        # 3. Handle move execution or strike penalty
        if legal_move_found is not None:
            board.push(legal_move_found)
            if is_white_turn:
                white_legal_moves += 1
                white_fitness += 10.0
            else:
                black_legal_moves += 1
                black_fitness += 10.0
        else:
            # Only reached if use_mask=False and raw network misses
            if is_white_turn:
                white_strikes += 1
                white_fitness -= 15.0
                if white_strikes >= max_strikes:
                    end_reason = "disqualification"
                    winner = "black"
                    white_fitness -= 50.0
                    black_fitness += 50.0
                    disqualification_note = f"White disqualified on turn {turns}"
                    break
            else:
                black_strikes += 1
                black_fitness -= 15.0
                if black_strikes >= max_strikes:
                    end_reason = "disqualification"
                    winner = "white"
                    black_fitness -= 50.0
                    white_fitness += 50.0
                    disqualification_note = f"Black disqualified on turn {turns}"
                    break

            fallback_move = next(iter(board.legal_moves))
            board.push(fallback_move)

        # 4. Anti-stalling repetition check
        pos_key = board.fen().split(" ")[0]
        count = position_history.get(pos_key, 0) + 1
        position_history[pos_key] = count
        if count >= 3:
            end_reason = "repetition"
            winner = "draw"
            if is_white_turn:
                white_fitness -= 30.0
            else:
                black_fitness -= 30.0
            break

    # Determine game outcomes if game finished naturally
    if end_reason not in ("disqualification", "repetition"):
        if board.is_checkmate():
            end_reason = "checkmate"
            if board.turn == chess.BLACK:
                winner = "white"
                white_fitness += 50.0
            else:
                winner = "black"
                black_fitness += 50.0
        elif board.is_stalemate() or board.is_insufficient_material():
            end_reason = "stalemate"
            winner = "draw"
            white_fitness += 15.0
            black_fitness += 15.0
        else:
            end_reason = "move_limit"
            winner = "draw"
            white_fitness += 10.0
            black_fitness += 10.0

    # Material differential bonus
    white_mat = compute_material(board, chess.WHITE)
    black_mat = compute_material(board, chess.BLACK)
    mat_diff = (white_mat - black_mat) * 2.0
    white_fitness += mat_diff
    black_fitness -= mat_diff

    return MatchResult(
        winner=winner,
        end_reason=end_reason,
        white_fitness=round(white_fitness, 2),
        black_fitness=round(black_fitness, 2),
        turns_played=turns,
        white_legal_moves=white_legal_moves,
        white_attempted_moves=white_attempts,
        black_legal_moves=black_legal_moves,
        black_attempted_moves=black_attempts,
        disqualification_note=disqualification_note
    )


if __name__ == "__main__":
    g1 = ChessGladiatorNet()
    g2 = ChessGladiatorNet()
    res = play_match(g1, g2, max_turns=40, use_mask=True)
    print(f"Match Winner: {res.winner.upper()} | Turns: {res.turns_played} | Reason: {res.end_reason}")
    print(f"White: {res.white_legal_moves}/{res.white_attempted_moves} legal | Black: {res.black_legal_moves}/{res.black_attempted_moves} legal")
