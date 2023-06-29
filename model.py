import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_size=256, embedding_size=128, n_layers=4, device=None):
        super(Model, self).__init__()

        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.main = nn.GRU(self.embedding_size, self.embedding_size, self.n_layers, dropout=0.1)
        self.fc = nn.Linear(self.embedding_size, self.input_size, bias=False)

    def forward(self, x, hidden=None):
        x = self.encoder(x).squeeze(2)

        if hidden is None:
            out, hidden = self.main(x)
        else:
            out, hidden = self.main(x, hidden)

        x = self.fc(out)
        return x, hidden
