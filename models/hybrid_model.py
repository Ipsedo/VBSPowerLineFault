import torch.nn as nn
import torch as th


class SpacedConvModel(nn.Module):
    def __init__(self, batch_size):
        super(SpacedConvModel, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(50, 50)
        )
        # size = 128, 2666

        self.lstm = nn.LSTM(64, 128, num_layers=1, batch_first=True, bidirectional=False)

        self.lin = nn.Sequential(
            nn.Linear(128, 128 * 4),
            nn.ReLU(),
            nn.Linear(128 * 4, 3),
            nn.Sigmoid()
        )

        self.c = th.randn(1, batch_size, 128)
        self.h = th.randn(1, batch_size, 128)

    def forward(self, x):
        out_conv = self.seq1(x)
        out_conv = out_conv.transpose(1, 2)

        c, h = self.c[:, :out_conv.size(0), :], self.h[:, :out_conv.size(0), :]

        if out_conv.is_cuda:
            c, h = c.cuda(), h.cuda()

        out_lstm, _ = self.lstm(out_conv, (h, c))
        out_lstm = out_lstm[:, -1, :]

        out_lin = self.lin(out_lstm)

        return out_lin
