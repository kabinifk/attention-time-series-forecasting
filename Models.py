import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, 128)
        self.linear2 = nn.Linear(128, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.attn_weights = None

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            need_weights=True,
            average_attn_weights=False
        )
        self.attn_weights = attn_weights
        src = self.norm1(src + self.dropout(attn_output))
        ff = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + self.dropout(ff))
        return src


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([CustomEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x[:, -1, :])

    def get_attention_weights(self):
        return [layer.attn_weights for layer in self.layers]


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, input_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
