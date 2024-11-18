import torch
import torch.nn as nn
from torch import Tensor

MODEL_PATH = "data/lstm3.pth"

class LSTMModel(nn.Module):
    """
    ジャンケンの手を予測する LSTM モデル。
    """
    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, hc=None) -> tuple[Tensor, tuple]:
        x = self.linear(x)
        x, hc = self.lstm(x, hc) if hc else self.lstm(x)
        x = self.output_layer(x)
        return x, hc

def load_model(device: torch.device) -> LSTMModel:
    """
    モデルをロードして評価モードに設定する。
    """
    model = LSTMModel(63, 50, 100, 3).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("モデルが正常に読み込まれました。")
        return model
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        raise
