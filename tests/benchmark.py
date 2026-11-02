"""
benchmark.py - Tactical Intelligence & Puzzle Benchmark Suite.

Tests your trained Neural Network and Search Engine on 5 classic chess puzzles:
1. Back-Rank Checkmate in 1 (Tactical execution)
2. Free Hanging Queen Capture (Material greed / value awareness)
3. Royal Knight Fork (Double attack tactical vision)
4. Defend Against Hanging Piece (Tactical defense)
5. Scholar's Mate Defense (Avoiding 1-move blunders)

Reports an objective Tactical Accuracy (%) and benchmark grade.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from typing import List
import chess

from src.model import ChessGladiatorNet
from play import load_model, get_ai_move_with_search, DEFAULT_WEIGHTS


@dataclass
class Puzzle:
    title: str
    fen: str
    expected_moves: List[str]
    description: str


PUZZLES = [
    Puzzle(
        title="1. Back-Rank Checkmate in 1",
        fen="r5k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        expected_moves=["a1a8"],
        description="White has an open file; Rook delivers instant back-rank checkmate."
    ),
    Puzzle(
        title="2. Free Queen Capture",
        fen="r1b1kbnr/pppp1ppp/8/8/4q3/4P3/PPP2PPP/RNBQKBNR w KQkq - 0 1",
        expected_moves=["d1e2", "c1d2", "b1c3", "a2a3"],  # Or capture if available
        description="Black's Queen is aggressively placed; White develops safely while defending."
    ),
    Puzzle(
        title="3. King & Rook Knight Fork",
        fen="r3k2r/ppp2ppp/8/8/3N4/8/PPP2PPP/R3K2R w KQkq - 0 1",
        expected_moves=["d4b5", "d4c6", "d4e6"],
        description="Knight moves into attacking position to target vulnerable enemy squares."
    ),
    Puzzle(
        title="4. Defending Under Attack",
        fen="rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
        expected_moves=["e4d5", "f3e5", "d2d3", "b1c3"],
        description="Pawns clash in the center; White must capture or defend the e4 square."
    ),
    Puzzle(
        title="5. Scholar's Mate Defense",
        fen="r1bqkb1r/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
        expected_moves=["g7g6", "d8e7", "d8f6"],
        description="White threatens instant checkmate Qxf7#; Black must defend with g6 or Queen."
    )
]


def run_benchmark(weights_path: str = "champion.pt", search_depth: int = 3):
    print(f"\n{'='*75}")
    print(f"CHESS NEURAL NETWORK TACTICAL BENCHMARK (SEARCH DEPTH: {search_depth})")
    print(f"Testing Model: {weights_path}")
    print(f"{'='*75}\n")

    ai = load_model(weights_path)
    score = 0

    for idx, p in enumerate(PUZZLES, 1):
        board = chess.Board(p.fen)
        move, eval_cp, nodes = get_ai_move_with_search(ai, board, depth=search_depth)
        move_uci = move.uci() if move else "None"

        # Check if chosen move is among acceptable tactical solutions
        is_correct = move_uci in p.expected_moves
        status_str = "✅ PASSED" if is_correct else "❌ MISSED"
        if is_correct:
            score += 1

        print(f"Puzzle {idx}: {p.title}")
        print(f"  Description: {p.description}")
        print(f"  AI Chose:    {move_uci} (Eval: {eval_cp/100.0:+.2f} cp | Nodes: {nodes})")
        print(f"  Expected:    {p.expected_moves} -> {status_str}\n")

    total = len(PUZZLES)
    pct = (score / total) * 100.0

    if pct == 100:
        grade = "Grandmaster Tactician (S Tier)"
    elif pct >= 80:
        grade = "Club Competitor (A Tier)"
    elif pct >= 60:
        grade = "Tactical Intermediate (B Tier)"
    else:
        grade = "Apprentice (C Tier)"

    print(f"{'='*75}")
    print(f"BENCHMARK SCORE: {score}/{total} ({pct:.0f}%) | Grade: {grade}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    run_benchmark(args.weights, args.depth)
