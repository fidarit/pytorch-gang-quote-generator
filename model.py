import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_size, embedding_size, n_layers=4, device=None):
        super(Model, self).__init__()

        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.main = nn.GRU(self.embedding_size, self.embedding_size, self.n_layers, dropout=0.4)
        self.decoder = nn.Linear(self.embedding_size, self.input_size)

    def forward(self, x, hidden=None):
        x = self.encoder(x).squeeze(2)

        if hidden is None:
            x, hidden = self.main(x)
        else:
            x, hidden = self.main(x, hidden)

        x = self.decoder(x)
        return x, hidden
