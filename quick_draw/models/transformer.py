import math

import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)'''
    
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Positional Encoding行列の初期化
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        
        # 偶数インデックスにsin、奇数インデックスにcosを適用
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数番目
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数番目
        
        # `pe` のサイズを `(1, max_len, model_dim)` にしてバッチ全体に同じエンコーディングを適用
        pe = pe.unsqueeze(0)  # (1, max_len, model_dim) の形にする
        
        # バッファに登録することで、学習されないパラメータとして保持
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        引数 x の形は (batch_size, seq_len, model_dim)
        """
        # 入力テンソルのシーケンス長に応じてPositional Encodingを追加
        seq_len = x.size(1)
        
        # 位置エンコーディングの適用
        x = x + self.pe[:, :seq_len, :].to(x.device)  # (1, seq_len, model_dim) の形をバッチ全体にブロードキャスト
        return x

class TransformerDecoderNet(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super(TransformerDecoderNet, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_decoder = TransformerEncoder(decoder_layers, nlayers)
        self.encoder = nn.Linear(2, d_model)
        self.d_model = d_model
        self.gen = nn.Linear(d_model, 2)
        self.cla = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, src_mask):
        """
        Args:
            src: Transformerへの入力データ
            src_mask: 入力データにかけるマスク
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_decoder(src, src, tgt_mask = src_mask)
        gen_out = self.sigmoid(self.gen(output))*255
        cla_out = self.sigmoid(self.cla(output))
        return gen_out, cla_out
    
class GPTNet(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super(GPTNet, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(decoder_layers, nlayers)
        self.embedding = nn.Linear(2, d_model)
        self.d_model = d_model
        self.gen = nn.Linear(d_model, 2)
        self.cla = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, src_mask):
        """
        Args:
            src: Transformerへの入力データ
            src_mask: 入力データにかけるマスク
        """
        src = self.embedding(src)# * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_mask)
        gen_out = self.sigmoid(self.gen(src))*255
        cla_out = self.sigmoid(self.cla(src))
        return gen_out, cla_out