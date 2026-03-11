# models.py (improved for better Webshell detection)

import torch
import torch.nn as nn


class WebshellDetector(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, hidden_dim=100):
        super(WebshellDetector, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32).clone().detach(), freeze=False
        )
        self.gru = nn.GRU(100, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)  # Added dropout for better generalization
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)
        return self.sigmoid(output)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, out_channels=128):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32).clone().detach(), freeze=False
        )
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, (k, 100)) for k in [3, 4, 5]])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(out_channels * 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max(pool, dim=2)[0] for pool in x]
        x = self.dropout(torch.cat(x, dim=1))
        x = self.fc(x)
        return self.sigmoid(x)


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, hidden_dim=128):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32).clone().detach(), freeze=False
        )
        self.lstm = nn.LSTM(100, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        out = self.dropout(output[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, hidden_dim=128, n_heads=2, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32).clone().detach(), freeze=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=100, nhead=n_heads, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_out = self.transformer(embedded)
        pooled = transformer_out.mean(dim=1)
        out = self.fc(pooled)
        return self.sigmoid(out)
