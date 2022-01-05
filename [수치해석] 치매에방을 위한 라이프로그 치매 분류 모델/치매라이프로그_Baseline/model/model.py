"""
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable



class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, device, n_layers=1):
        """
        """
        super(LSTM, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.randn(2 * self.n_layers, batch_size, self.hidden_dim)).to(self.device)
        c_0 = Variable(torch.randn(2 * self.n_layers, batch_size, self.hidden_dim)).to(self.device)
        return (h_0, c_0)

    def forward(self, x):
        """
        """
        batch_size = x.shape[0]
        self.h_c = self.init_hidden(batch_size)
        lstm_out, self.h_c = self.lstm(x, self.h_c)
        output = self.fc(lstm_out[:, -1, :])
        return F.softmax(output, dim=1)


