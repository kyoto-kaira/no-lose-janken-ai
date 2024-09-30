import torch.nn as nn
import torch.nn.functional as F

class LSTM_Net0(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Net0, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2tag1 = nn.Linear(hidden_dim, 258)
        self.hidden2tag2 = nn.Linear(hidden_dim, 258)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x1 = self.hidden2tag1(x)
        x2 = self.hidden2tag2(x)
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)
        return x1, x2