import torch
import torch.nn as nn
MODEL_PATH = "backend/data/lstm.pth"


class LSTMModel(nn.Module):
    """
    ジャンケン用モデル
    """
    def __init__(self, input_dim, emb_dim, hidden_dim, tagset_size):
        super(LSTMModel, self).__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, hc=None):
        x = self.linear(x)
        if hc==None:
            x, hc = self.lstm(x)
        else:
            x, hc = self.lstm(x, hc)
        x = self.hidden2tag(x)
        return x, hc


def load_model(device: torch.device) -> LSTMModel:
    """
    モデルをロードして評価モードに設定する。
    """
    model = LSTMModel(63, 50, 100, 3).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))
        model.eval()
        print("モデルが正常に読み込まれました。")
        return model
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        raise
