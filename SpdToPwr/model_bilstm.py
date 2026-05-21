# model_bilstm.py
import torch
import torch.nn as nn


class BiLSTMPredictor(nn.Module):
    """
    双层 BiLSTM 预测器（无 Attention，纯 BiLSTM + FC）
    - 适用于能耗预测（output_dim=1）和温度预测（output_dim=4）
    - 最后一时间步的拼接隐藏状态（forward + backward）送入全连接层
    """
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.3):
        super(BiLSTMPredictor, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)

        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1,
                             batch_first=True, bidirectional=True)

        lstm_out_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # BiLSTM Layer 1
        x, _ = self.lstm1(x)

        # BatchNorm 需要 (batch, channel, seq_len) 格式
        x = x.transpose(1, 2)
        x = self.bn1(x)
        x = x.transpose(1, 2)

        # BiLSTM Layer 2
        x, _ = self.lstm2(x)

        # 取最后时间步：前向+后向拼接 → (batch, hidden_dim * 2)
        x = x[:, -1, :]

        x = self.dropout(x)
        return self.fc(x)
