import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class EmbedHandler(nn.Module):

    def __init__(self, actions_count):
        super(EmbedHandler, self).__init__()
        self.theta = nn.Parameter(torch.zeros(actions_count, requires_grad=True))
        self.mu = nn.Parameter(torch.zeros(actions_count, requires_grad=True))

    def initWeights(self):
        self.theta.weight.data.uniform_(-0.5, 0.5)
        self.mu.weight.data.uniform_(-0.5, 0.5)

    def forward(self, tau, inputs):
        if len(inputs.shape) == 1:
            ix = inputs[0].item()
        else:
            ix = inputs.item()
        return torch.sigmoid(self.theta[ix] + self.mu[ix] * tau)


class RNNTasa(nn.Module):
    def calc_proba(self, tau, inputs):
        #print(self.theta)
        #print(self.mu)
        #print()
        #print(inputs[0].item())
        #assert 1 == 2
        #ix = actions_data.action2index(inputs)
        if len(inputs.shape) == 1:
            ix = inputs[0].item()
        else:
            ix = inputs.item()
        #print(self.mu[ix], tau)
        #print(F.sigmoid(self.theta[ix] + self.mu[ix] * tau))
        return torch.sigmoid(self.theta[ix] + self.mu[ix] * tau)

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderTASA(RNNTasa):
    def __init__(self, actions_count, hidden_size):
        super(EncoderTASA, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(actions_count, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        self.initWeights()

    def initWeights(self):
        self.embedding.weight.data.uniform_(-0.05, 0.05)

    def forward(self, inputs, hidden, tau, eh):
        embedded = self.embedding(inputs).view(1, 1, -1)
        output = embedded
        #prob = self.calc_proba(tau, inputs)
        prob = eh(tau, inputs)
        output *= prob
        output, hidden = self.gru(output, hidden)
        return output, hidden


class DecoderTASA(RNNTasa):
    def __init__(self, hidden_size, actions_count):
        super(DecoderTASA, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(actions_count, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.lin = nn.Linear(hidden_size, actions_count)
        self.softmax = nn.LogSoftmax(dim=1)

        self.initWeights()

    def initWeights(self):
        self.embedding.weight.data.uniform_(-0.05, 0.05)

    def forward(self, inputs, hidden, tau, eh):
        output = self.embedding(inputs).view(1, 1, -1)
        #output = F.relu(output)
        #prob = self.calc_proba(tau, inputs)
        prob = eh(tau, inputs)
        output *= prob
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.lin(output[0]))
        return output, hidden