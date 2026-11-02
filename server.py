"""
server.py - Lightweight Local HTTP Server for the Pixel Chess Web GUI.

Uses Python's standard library http.server (ZERO extra pip dependencies needed).
Connects your trained champion.pt neural network and search.py engine to the browser.

Endpoints:
- GET  /              -> Serves web/index.html
- GET  /web/*         -> Serves static files (CSS, JS, fonts)
- POST /api/move      -> Receives FEN and search depth, returns best move and evaluation
- GET  /api/status    -> Health check & model status
"""

import os
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import chess
import torch

from src.model import ChessGladiatorNet
from src.search import search_best_move


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


class ChessAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
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

        # Serve static web files
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/move":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

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

                    # If move is valid, compute new board FEN
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
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        # Handle CORS preflight
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
