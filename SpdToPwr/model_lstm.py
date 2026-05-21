# model_lstm.py
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    双层 LSTM 预测器（无 Attention，纯 LSTM + FC）
    - 适用于能耗预测（output_dim=1）和温度预测（output_dim=4）
    - 取最后一时间步的隐藏状态送入全连接层
    """
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.3):
        super(LSTMPredictor, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # LSTM Layer 1
        x, _ = self.lstm1(x)
        x = self.norm1(x)

        # LSTM Layer 2
        x, _ = self.lstm2(x)

        # 取最后时间步
        x = x[:, -1, :]          # (batch, hidden_dim)

        x = self.dropout(x)
        return self.fc(x)
