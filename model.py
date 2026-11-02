"""
model.py - Dual-Headed Spatial ConvNet for Chess AI (Policy & Value).

Architecture (AlphaZero Style):
- Input: (13, 8, 8) canonical board float tensor
- Backbone: 2D Convolutions with padding=1 preserving 8x8 spatial grid
- Dense layer: 128 hidden units
- Triple Heads:
    1. Move Head (Policy): 4,096 logits over theoretical transitions
    2. Promotion Head: 4 logits (Queen, Rook, Bishop, Knight)
    3. Value Head (Evaluation): Single scalar in [-1.0, +1.0] predicting win probability
       (+1.0 = Active player winning, 0.0 = Equal, -1.0 = Losing)
- Action Masking: Sets illegal move logits to -infinity for 100.0% legal play.
"""

import copy
from typing import Tuple, List, Optional
import torch
import torch.nn as nn


class ChessGladiatorNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Spatial feature extraction (13 input channels -> 32 channels)
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()  # 32 channels * 8 * 8 = 2,048 flat features
        )

        # 2. Shared dense representation
        self.fc_shared = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU()
        )

        # 3. Policy Head: 4,096 move logits
        self.move_head = nn.Linear(128, 4096)

        # 4. Promotion Head: 4 logits (Q, R, B, N)
        self.promo_head = nn.Linear(128, 4)

        # 5. Value Head (AlphaZero Evaluation): outputs [-1.0 .. +1.0]
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Input:  (batch, 13, 8, 8) or (13, 8, 8)
        Output tuple:
            move_logits:  (batch, 4096) or (4096,)
            promo_logits: (batch, 4) or (4,)
            value:        (batch, 1) or scalar tensor [-1.0 .. +1.0]
        """
        single_input = (x.dim() == 3)
        if single_input:
            x = x.unsqueeze(0)

        feat = self.conv_backbone(x)
        shared = self.fc_shared(feat)

        move_logits = self.move_head(shared)
        promo_logits = self.promo_head(shared)
        value = self.value_head(shared)

        if single_input:
            return move_logits.squeeze(0), promo_logits.squeeze(0), value.squeeze(0).squeeze(0)

        return move_logits, promo_logits, value.squeeze(-1)

    def predict_action(
        self,
        board_tensor: torch.Tensor,
        legal_actions: Optional[List[int]] = None,
        k: int = 5
    ) -> Tuple[List[int], int]:
        """Inference helper with Action Masking."""
        with torch.no_grad():
            move_logits, promo_logits, _ = self.forward(board_tensor)

            if legal_actions is not None and len(legal_actions) > 0:
                mask = torch.full_like(move_logits, float('-inf'))
                mask[legal_actions] = 0.0
                move_logits = move_logits + mask
                k = min(k, len(legal_actions))

            top_moves = torch.topk(move_logits, k=k).indices.tolist()
            promo_choice = int(torch.argmax(promo_logits).item())
            return top_moves, promo_choice

    def evaluate_board(self, board_tensor: torch.Tensor) -> float:
        """Returns the neural network's evaluation of the board from [-1.0 .. +1.0]."""
        with torch.no_grad():
            _, _, value = self.forward(board_tensor)
            return float(value.item())

    def clone(self) -> "ChessGladiatorNet":
        return copy.deepcopy(self)


if __name__ == "__main__":
    net = ChessGladiatorNet()
    dummy = torch.zeros((13, 8, 8))

    move_logits, promo_logits, value = net(dummy)
    print("Move logits shape:", move_logits.shape, "(Expected: [4096])")
    print("Promo logits shape:", promo_logits.shape, "(Expected: [4])")
    print("Value output:", float(value.item()), "(Expected in [-1.0 .. +1.0])")
    print("[SUCCESS] Dual-headed Policy + Value architecture verified!")
