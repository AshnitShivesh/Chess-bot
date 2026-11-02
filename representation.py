"""
representation.py - Canonical Perspective & Move Action Encoding.

Canonical Perspective (The AlphaZero Standard):
The neural network ALWAYS sees the board from the active player's perspective:
- Channels 0..5:  OUR pieces [P, N, B, R, Q, K] (White if White's turn, Black if Black's turn)
- Channels 6..11: OPPONENT pieces [P, N, B, R, Q, K]
- Channel 12:     Perspective flag (1.0 for White, 0.0 for Black)

When it is Black's turn, the board is vertically mirrored (rank 1 <-> rank 8).
This means opening moves (like pushing the king's pawn two squares) have the EXACT
same action index (796) whether playing as White (e2e4) or Black (e7e5).
The model learns chess ONCE and plays equally well as both colors!
"""

from typing import List
import chess
import torch


PIECE_OFFSET = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

PROMO_PIECES = [
    chess.QUEEN,   # 0
    chess.ROOK,    # 1
    chess.BISHOP,  # 2
    chess.KNIGHT   # 3
]


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts board to a (13, 8, 8) tensor from the ACTIVE PLAYER'S CANONICAL PERSPECTIVE.
    Channels 0..5:  Active player's pieces
    Channels 6..11: Opponent's pieces
    Channel 12:     1.0 if White, 0.0 if Black
    """
    tensor = torch.zeros((13, 8, 8), dtype=torch.float32)
    is_white = (board.turn == chess.WHITE)

    for square, piece in board.piece_map().items():
        # Canonical square: vertically mirrored for Black (square ^ 56)
        canon_sq = square if is_white else (square ^ 56)
        rank = canon_sq >> 3
        file = canon_sq & 7

        # Channels 0..5 are always active player's pieces
        channel = PIECE_OFFSET[piece.piece_type]
        if piece.color != board.turn:
            channel += 6

        tensor[channel, rank, file] = 1.0

    # Channel 12: Color flag (1.0 for White, 0.0 for Black)
    if is_white:
        tensor[12].fill_(1.0)
    return tensor


def move_to_index(move: chess.Move, turn: chess.Color = chess.WHITE) -> int:
    """
    Encodes a physical chess.Move into a canonical action index [0 .. 4095].
    Mirrors squares vertically if turn == chess.BLACK.
    """
    is_white = (turn == chess.WHITE)
    canon_from = move.from_square if is_white else chess.square_mirror(move.from_square)
    canon_to = move.to_square if is_white else chess.square_mirror(move.to_square)
    return (canon_from * 64) + canon_to


def index_to_move(index: int, board: chess.Board, promo_choice: int = 0) -> chess.Move:
    """
    Decodes a canonical action index [0 .. 4095] into a physical chess.Move on the board.
    Translates canonical coordinates back to physical board coordinates.
    """
    canon_from = index // 64
    canon_to = index % 64

    is_white = (board.turn == chess.WHITE)
    real_from = canon_from if is_white else chess.square_mirror(canon_from)
    real_to = canon_to if is_white else chess.square_mirror(canon_to)

    # In canonical coordinates, promotion is moving from rank 6 to rank 7
    canon_from_rank = chess.square_rank(canon_from)
    canon_to_rank = chess.square_rank(canon_to)
    piece = board.piece_at(real_from)

    if piece and piece.piece_type == chess.PAWN and canon_to_rank == 7 and canon_from_rank == 6:
        p_idx = max(0, min(3, promo_choice))
        return chess.Move(real_from, real_to, promotion=PROMO_PIECES[p_idx])

    return chess.Move(real_from, real_to)


def get_top_k_actions(move_logits: torch.Tensor, k: int = 5) -> List[int]:
    """Extracts top-k highest scoring action indices."""
    return torch.topk(move_logits, k=k).indices.tolist()


if __name__ == "__main__":
    # Test canonical symmetry
    board = chess.Board()

    # White e2e4
    e2e4 = chess.Move.from_uci("e2e4")
    idx_white = move_to_index(e2e4, turn=chess.WHITE)
    print(f"White e2e4 canonical index: {idx_white}")

    # Push e4
    board.push(e2e4)

    # Black e7e5
    e7e5 = chess.Move.from_uci("e7e5")
    idx_black = move_to_index(e7e5, turn=chess.BLACK)
    print(f"Black e7e5 canonical index: {idx_black}")

    assert idx_white == idx_black, "Canonical perspective symmetry failed!"
    print(f"[SUCCESS] Perfect Canonical Symmetry! Both e2e4 and e7e5 map to index {idx_white}!")

    # Test decoding for Black
    decoded_black = index_to_move(idx_black, board)
    print(f"Index {idx_black} decoded for Black: {decoded_black.uci()} (Expected: e7e5)")
    assert decoded_black == e7e5
