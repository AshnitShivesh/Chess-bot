# Chess Neural Network

A deep neural network and chess engine built from scratch in PyTorch to explore how neural networks learn spatial rules and game strategy from a blank slate.

The primary motivation behind this project was to understand neural network mechanics and reinforcement learning firsthand—specifically, whether an autonomous network could deduce the legal rules and geometry of chess without hardcoded rule masks, and then learn strategy through self-play.

---

## Overview

The engine utilizes a spatial Convolutional Neural Network (ConvNet) with a dual-headed architecture (Policy and Value), coupled with Alpha-Beta lookahead search and invalid action masking.

- **State Representation**: `(13, 8, 8)` float tensor representing piece placement from a canonical perspective.
- **Action Space**: Flat 4,096 discrete transition space (`from_square * 64 + to_square`) with pawn promotion selection.
- **Inference**: AlphaZero-style invalid action masking guarantees 100% legal moves during play.
- **Search**: Alpha-Beta search with root policy caching, MVV-LVA move ordering, and quiescence search.

---

## Development Log & Technical Challenges

### 1. Initial Approach: Neuroevolution & The 50% Plateau
Development began with an evolutionary approach: a population of 32 randomly initialized networks dueling head-to-head, where illegal move attempts incurred penalties and survivors reproduced with Gaussian weight mutations.

While legality initially climbed from 3% to 35%, training stalled at an exact **50.0% legality ceiling**. 

**Root Cause**: In standard chess coordinates, White moves up the board while Black moves down. The network learned opening pawn pushes as White, but when playing as Black, the same learned weights attempted to move White's pieces, resulting in immediate disqualification on move 2. Because every duel terminated on turn 2 (one legal White move, one failed Black attempt), the metric locked at 1/2 (50.0%).

**Solution**: Implemented **Canonical Perspective**. Whenever Black is to move, the board is vertically mirrored (`square ^ 56`) and the piece channels are swapped so that the network always views the board from the active player's perspective. A learned move like `e2e4` for White automatically functions as `e7e5` for Black.

### 2. High-Dimensional Optimization Limits (The 65% Plateau)
With the perspective fix and strike tolerance, games extended to ~13 turns, but legality plateaued between **61% and 66%**.

**Root Cause**: The ConvNet contains roughly 790,000 parameters. In evolutionary algorithms, scalar fitness values provide no credit assignment—the network receives no signal indicating *which* weights caused an illegal move on turn 9 versus a legal move on turn 2. In a 790,000-dimensional continuous space, random mutations to later moves inevitably degrade weights responsible for earlier turns (catastrophic interference).

### 3. Transition to Backpropagation
To resolve the credit assignment bottleneck, training shifted from blind mutation to gradient-based policy learning (`train_rl.py`):
- For each board state, a uniform target distribution is constructed over all legal moves.
- Multi-action cross-entropy loss is applied:
  $$\mathcal{L} = -\sum_{m \in \text{legal}} \frac{1}{|\text{legal}|} \log P(m)$$
- PyTorch backpropagation calculates exact gradients for all 790,000 weights, boosting legal move logits while suppressing all invalid transitions and empty squares.

**Result**: Legality rose from 54% to **98.0% in 14.4 seconds**, eventually reaching **99.6% Top-5 legality** and **90.4% Top-1 legality** on raw unmasked predictions.

### 4. Search and Self-Play (Phase 2)
To transition from rule legality to strategic play:
1. **Value Head**: Added a $\tanh$ output head in $[-1.0, +1.0]$ predicting win probability.
2. **Lookahead Search (`search.py`)**: Implemented an Alpha-Beta search engine thinking 3 moves ahead, using root policy caching and quiescence search to avoid tactical blunders.
3. **Self-Play Training (`train_selfplay.py`)**: The model plays games against itself, attributing game outcomes ($+1.0$ win, $-1.0$ loss) to train the Value Head directly.

---

## Architecture

```
Board State -> Canonical (13, 8, 8) Float Tensor
                      │
                      ▼
     Conv2D (13 -> 32, 3x3, pad=1) + ReLU
     Conv2D (32 -> 32, 3x3, pad=1) + ReLU
     Flatten -> Linear (2048 -> 128) + ReLU
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  Policy Head    Promo Head    Value Head
 (4,096 logits)   (4 logits)   (1 scalar, [-1, 1])
        │             │             │
        └─────────────┼─────────────┘
                      ▼
         Action Masking (-inf on illegal)
                      │
                      ▼
         Alpha-Beta Search (Depth 3)
```

---

## File Structure

```
chess-neural-network/
├── src/                         # Core chess neural network & search package
│   ├── __init__.py
│   ├── model.py                 # Neural network architecture (Policy + Promo + Value heads)
│   ├── representation.py        # 13x8x8 board encoding with canonical perspective
│   └── search.py                # Alpha-Beta search, quiescence, root caching
├── training/                    # RL and evolution training pipelines
│   ├── __init__.py
│   ├── train_rl.py              # Policy gradient backpropagation trainer
│   ├── train_selfplay.py        # AlphaZero-style self-play RL loop
│   ├── evolution.py             # Phase 1 genetic algorithm & mutation engine
│   └── arena.py                 # Tournament matchmaking & survival selection
├── weights/                     # Pre-trained model weights
│   └── champion.pt              # 99.6% trained neural network weights
├── web/                         # Interactive retro-arcade frontend
│   ├── index.html               # Semantic arcade layout & CRT scanline overlay
│   ├── style.css                # Pixel-art CSS styling
│   └── app.js                   # Interactive board, 8-bit Web Audio synth, offline AI
├── tests/                       # Verification & benchmarks
│   ├── __init__.py
│   ├── test_all.py              # Full unit test suite (7 tests)
│   └── benchmark.py             # Tactical puzzle benchmark suite (checkmates, forks, defense)
├── play.py                      # CLI game entrypoint (Human vs AI, AI vs AI)
├── server.py                    # Web GUI server entrypoint (Zero-dependency HTTP server)
├── requirements.txt             # Minimal pip requirements (torch, python-chess, numpy)
└── README.md                    # Project documentation & architecture guide
```

---

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Interactive Pixel Web GUI (Browser)
Run the local lightweight server:
```bash
python server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in any browser.

**Web Features:**
- **Retro Arcade Aesthetic**: CRT scanline filter, responsive pixel-art vector pieces (no external image downloads needed), and retro typography.
- **8-Bit Web Audio SFX**: Real-time sound effects generated via Web Audio API oscillators (move blip, capture crunch, check alert, victory fanfare).
- **Live Evaluation Gauge**: Vertical real-time advantage bar and score calculated by the neural network's Value Head and Alpha-Beta search.
- **Multiple Modes**: Play against AI (as White or Black with automatic board orientation), spectate AI Auto-play, or analyze in Free Play mode.
- **Offline Fallback**: Can also be hosted directly on GitHub Pages—if the Python backend is offline, an integrated browser-based minimax engine activates automatically.

### 2. Play in the Terminal (CLI)
```bash
python play.py --mode human-white --depth 3
```

### 3. Run Tactical Benchmarks
```bash
python tests/benchmark.py --depth 3
```

### 4. Run Self-Play Training
```bash
python training/train_selfplay.py --iters 30 --games 4
```

### 5. Run Unit Tests
```bash
python tests/test_all.py
```
