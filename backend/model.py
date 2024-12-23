from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor


class LSTMNet(nn.Module):
    """
    LSTMを用いたジャンケンゲームのモデル
    """

    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int, tagset_size: int) -> None:
        super(LSTMNet, self).__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(
        self, x: Tensor, hc: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor] | None]:
        x = self.linear(x)
        x, hc = self.lstm(x, hc) if hc is not None else self.lstm(x)
        x = self.hidden2tag(x)
        return x, hc
