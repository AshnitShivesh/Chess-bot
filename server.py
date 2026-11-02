"""
server.py - Lightweight Local HTTP Server for the Pixel Chess Web GUI.

Uses Python's standard library http.server (ZERO extra pip dependencies needed).
Connects your trained champion.pt neural network and search.py engine to the browser.

Endpoints:
- GET  /                  -> Serves web/index.html
- GET  /web/*             -> Serves static files (CSS, JS, fonts)
- POST /api/move          -> Receives FEN and search depth, returns best move and evaluation
- GET  /api/status        -> Health check & model status
- POST /api/train/start   -> Starts live background training session (self-play RL)
- POST /api/train/stop    -> Stops background training session
- GET  /api/train/status  -> Polls live training progress, loss curve, and board FEN
"""

import os
import json
import time
import urllib.parse
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import chess
import torch

from src.model import ChessGladiatorNet
from src.search import search_best_move
from src.representation import board_to_tensor, move_to_index, index_to_move


PORT = 8000
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Load model once at startup
MODEL = ChessGladiatorNet()
CHAMP_PATH = os.path.join(os.path.dirname(__file__), "weights", "champion.pt")
if not os.path.exists(CHAMP_PATH):
    CHAMP_PATH = os.path.join(os.path.dirname(__file__), "champion.pt")
MODEL_LOADED = False

if os.path.exists(CHAMP_PATH):
    try:
        MODEL.load_state_dict(torch.load(CHAMP_PATH, map_location="cpu"), strict=False)
        MODEL.eval()
        MODEL_LOADED = True
        print(f"[SERVER] Successfully loaded trained weights from '{CHAMP_PATH}'")
    except Exception as e:
        print(f"[SERVER] Model load note: {e}")
else:
    print(f"[SERVER] Warning: '{CHAMP_PATH}' not found, using initialized baseline model.")


# ==============================================================================
# LIVE BACKGROUND TRAINING ENGINE
# ==============================================================================
TRAINING_STATE = {
    "is_training": False,
    "step": 0,
    "total_steps": 5,
    "loss": 0.0,
    "loss_history": [],
    "current_fen": chess.STARTING_FEN,
    "last_move": None,
    "logs": ["[IDLE] Ready to train. Click START TRAINING to begin."]
}
TRAINING_STOP_FLAG = False
TRAIN_LOCK = threading.Lock()


def run_training_worker(total_steps=5, mode="selfplay"):
    global TRAINING_STATE, TRAINING_STOP_FLAG, MODEL

    with TRAIN_LOCK:
        TRAINING_STATE["is_training"] = True
        TRAINING_STATE["step"] = 0
        TRAINING_STATE["total_steps"] = total_steps
        TRAINING_STATE["loss_history"] = []
        TRAINING_STATE["logs"] = [f"[INIT] Started {mode} training session ({total_steps} iterations)..."]
        TRAINING_STOP_FLAG = False

    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=5e-4, weight_decay=1e-4)

    for step in range(1, total_steps + 1):
        if TRAINING_STOP_FLAG:
            with TRAIN_LOCK:
                TRAINING_STATE["logs"].append("[ABORT] Training stopped by user.")
            break

        with TRAIN_LOCK:
            TRAINING_STATE["step"] = step
            TRAINING_STATE["logs"].append(f"[STEP {step}/{total_steps}] Simulating self-play game...")

        # Play a self-play game and stream moves
        board = chess.Board()
        moves_count = 0
        experiences = []

        while not board.is_game_over() and moves_count < 60:
            if TRAINING_STOP_FLAG:
                break
            moves_count += 1
            tensor = board_to_tensor(board)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            legal_indices = [move_to_index(m, turn=board.turn) for m in legal_moves]

            with torch.no_grad():
                move_logits, promo_logits, _ = MODEL(tensor)
                mask = torch.full_like(move_logits, float('-inf'))
                mask[legal_indices] = 0.0
                masked_logits = move_logits + mask

                # 1-Ply Tactical Check
                mate_move = None
                for m in legal_moves:
                    board.push(m)
                    if board.is_checkmate():
                        mate_move = m
                        board.pop()
                        break
                    board.pop()

                if mate_move is not None:
                    chosen_idx = move_to_index(mate_move, turn=board.turn)
                    chosen_move = mate_move
                elif moves_count <= 8:
                    # Opening exploration with temperature
                    probs = torch.softmax(masked_logits / 1.0, dim=-1)
                    legal_probs = probs[legal_indices]
                    legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                    sampled_idx = torch.multinomial(legal_probs, 1).item()
                    chosen_idx = legal_indices[sampled_idx]
                    promo_choice = int(torch.argmax(promo_logits).item())
                    chosen_move = index_to_move(chosen_idx, board, promo_choice=promo_choice)
                else:
                    chosen_idx = int(torch.argmax(masked_logits).item())
                    promo_choice = int(torch.argmax(promo_logits).item())
                    chosen_move = index_to_move(chosen_idx, board, promo_choice=promo_choice)

            experiences.append((tensor, chosen_idx, board.turn))
            board.push(chosen_move)

            with TRAIN_LOCK:
                TRAINING_STATE["current_fen"] = board.fen()
                TRAINING_STATE["last_move"] = chosen_move.uci()

            time.sleep(0.04)  # 40ms delay per move for smooth visual playback

        if TRAINING_STOP_FLAG:
            break

        # Compute winner and reward
        if board.is_checkmate():
            winner_str = "White win" if board.turn == chess.BLACK else "Black win"
            winner_val = 1.0 if board.turn == chess.BLACK else -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or moves_count >= 60:
            winner_str = "Draw"
            winner_val = 0.0
        else:
            winner_str = "Terminated"
            winner_val = 0.0

        with TRAIN_LOCK:
            TRAINING_STATE["logs"].append(f"[GAME {step}] {winner_str} in {moves_count} moves.")

        # Train neural net with backprop
        if experiences:
            MODEL.train()
            optimizer.zero_grad()

            batch_states = torch.stack([exp[0] for exp in experiences])
            batch_actions = torch.tensor([exp[1] for exp in experiences], dtype=torch.long)

            val_targets = []
            for exp in experiences:
                if winner_val == 0.0:
                    val_targets.append(0.0)
                else:
                    val_targets.append(winner_val if exp[2] == chess.WHITE else -winner_val)
            val_targets = torch.tensor(val_targets, dtype=torch.float32).unsqueeze(1)

            pred_moves, _, pred_vals = MODEL(batch_states)
            policy_loss = torch.nn.functional.cross_entropy(pred_moves, batch_actions)
            value_loss = torch.nn.functional.mse_loss(pred_vals, val_targets)
            loss = policy_loss + 1.2 * value_loss

            loss.backward()
            optimizer.step()
            MODEL.eval()

            current_loss = round(float(loss.item()), 4)
            with TRAIN_LOCK:
                TRAINING_STATE["loss"] = current_loss
                TRAINING_STATE["loss_history"].append(current_loss)
                TRAINING_STATE["logs"].append(
                    f"[LOSS {step}] Loss: {current_loss:.4f} (Policy: {policy_loss.item():.3f}, Value: {value_loss.item():.3f})"
                )

    # Save weights after training completes
    try:
        torch.save(MODEL.state_dict(), CHAMP_PATH)
        with TRAIN_LOCK:
            TRAINING_STATE["logs"].append(f"[SAVED] Updated champion weights saved to '{CHAMP_PATH}'")
    except Exception as e:
        with TRAIN_LOCK:
            TRAINING_STATE["logs"].append(f"[SAVE NOTE] {e}")

    with TRAIN_LOCK:
        TRAINING_STATE["is_training"] = False
        TRAINING_STATE["logs"].append("[COMPLETE] Training session finished!")


# ==============================================================================
# HTTP HANDLER
# ==============================================================================
class ChessAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ["/", "/index.html"]:
            self.path = "/index.html"
            return super().do_GET()

        if parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            status = {
                "status": "online",
                "model_loaded": MODEL_LOADED,
                "weights_path": CHAMP_PATH
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return

        if parsed.path == "/api/train/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with TRAIN_LOCK:
                state_copy = dict(TRAINING_STATE)
            self.wfile.write(json.dumps(state_copy).encode("utf-8"))
            return

        # Serve static web files
        return super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        if self.path == "/api/move":
            try:
                data = json.loads(body)
                fen = data.get("fen", chess.STARTING_FEN)
                depth = int(data.get("depth", 3))

                board = chess.Board(fen)

                if board.is_game_over():
                    response = {
                        "error": "Game over",
                        "game_over": True,
                        "result": board.result()
                    }
                else:
                    move, score_cp, nodes = search_best_move(board, model=MODEL, depth=depth)
                    move_uci = move.uci() if move else None

                    new_fen = None
                    if move:
                        board.push(move)
                        new_fen = board.fen()

                    response = {
                        "move": move_uci,
                        "score_cp": score_cp,
                        "score_display": f"{score_cp / 100.0:+.2f}",
                        "nodes": nodes,
                        "new_fen": new_fen,
                        "game_over": board.is_game_over(),
                        "result": board.result() if board.is_game_over() else None
                    }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if self.path == "/api/train/start":
            global TRAINING_STOP_FLAG
            try:
                data = json.loads(body)
                steps = int(data.get("steps", 5))
                mode = data.get("mode", "selfplay")

                with TRAIN_LOCK:
                    if TRAINING_STATE["is_training"]:
                        response = {"status": "busy", "message": "Training already in progress."}
                    else:
                        TRAINING_STOP_FLAG = False
                        t = threading.Thread(target=run_training_worker, args=(steps, mode), daemon=True)
                        t.start()
                        response = {"status": "started", "steps": steps, "mode": mode}

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if self.path == "/api/train/stop":
            TRAINING_STOP_FLAG = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopping"}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server():
    server = HTTPServer(("127.0.0.1", PORT), ChessAPIHandler)
    print(f"\n{'='*70}")
    print(f"PIXEL CHESS GUI SERVER RUNNING")
    print(f"Open your browser at: http://localhost:{PORT}")
    print(f"Press Ctrl + C to stop the server")
    print(f"{'='*70}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Server stopped.")


if __name__ == "__main__":
    run_server()
