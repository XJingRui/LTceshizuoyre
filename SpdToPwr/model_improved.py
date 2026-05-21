# model_improved.py
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        score = self.v(torch.tanh(self.W(lstm_out)))
        attention_weights = torch.softmax(score, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return context, attention_weights


class EVPowerImprovedModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, seq_len=20, output_dim=1):
        super(EVPowerImprovedModel, self).__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)

        lstm_out_dim = hidden_dim * 2
        self.attention = BahdanauAttention(lstm_out_dim)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.input_projection(x)
        lstm_out, _ = self.lstm1(x)

        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = self.bn1(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)

        lstm_out, _ = self.lstm2(lstm_out)
        context, attn_weights = self.attention(lstm_out)

        context = self.dropout(context)
        predictions = self.fc(context)
        return predictions