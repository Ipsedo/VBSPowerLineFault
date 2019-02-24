import torch as th
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()

        self.seq1 = nn.Sequential(nn.Conv1d(3, 3 * 2, kernel_size=3),
                                  nn.MaxPool1d(3, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(3 * 2, 3 * 4, kernel_size=11, stride=5),
                                  nn.MaxPool1d(11, 11),
                                  nn.ReLU(),
                                  nn.Conv1d(3 * 4, 3 * 6, kernel_size=21, stride=9),
                                  nn.MaxPool1d(21, 21),
                                  nn.ReLU())

        self.seq2 = nn.Sequential(nn.Linear(3 * 6 * 25, 3),
                                  nn.Sigmoid())

    def forward(self, x):
        out = self.seq1(x)
        out = out.view(-1, 3 * 6 * 25)
        out = self.seq2(out)
        return out
