import torch as th
import torch.nn as nn


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()

        self.seq1 = nn.Sequential(nn.Conv1d(3, 18, kernel_size=3),
                                  nn.MaxPool1d(3, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(18, 24, kernel_size=11, stride=5),
                                  nn.MaxPool1d(11, 11),
                                  nn.ReLU(),
                                  nn.Conv1d(24, 56, kernel_size=21, stride=9),
                                  nn.MaxPool1d(21, 21),
                                  nn.ReLU())

        self.seq2 = nn.Sequential(nn.Linear(56 * 25, 56 * 25 * 4),
                                  #nn.BatchNorm1d(3 ** 4 * 25 * 3),
                                  nn.ReLU(),
                                  nn.Linear(56 * 25 * 4, 3),
                                  nn.Sigmoid())

    def forward(self, x):
        out = self.seq1(x)
        out = out.view(-1, 56 * 25)
        out = self.seq2(out)
        return out


class ConvModel2(nn.Module):
    def __init__(self):
        super(ConvModel2, self).__init__()

        self.seq1 = nn.Sequential(nn.Conv1d(3, 3 ** 2, kernel_size=3),
                                  nn.MaxPool1d(3, stride=3),
                                  nn.ReLU(),
                                  )

        self.seq2_c1 = nn.Sequential(                                     nn.Conv1d(3 ** 2, 3 ** 3, kernel_size=11, stride=5),
                                     nn.MaxPool1d(11, 11),
                                     nn.ReLU(),
                                     nn.Conv1d(3 ** 3, 3 ** 4, kernel_size=21, stride=9),
                                     nn.MaxPool1d(21, 21),
                                     nn.ReLU())

        self.seq2_c2 = nn.Sequential(
                                     nn.Conv1d(3 ** 2, 3 ** 3, kernel_size=11, stride=5),
                                     nn.MaxPool1d(11, 11),
                                     nn.ReLU(),
                                     nn.Conv1d(3 ** 3, 3 ** 4, kernel_size=21, stride=9),
                                     nn.MaxPool1d(21, 21),
                                     nn.ReLU())

        self.seq2_c3 = nn.Sequential(
                                     nn.Conv1d(3 ** 2, 3 ** 3, kernel_size=11, stride=5),
                                     nn.MaxPool1d(11, 11),
                                     nn.ReLU(),
                                     nn.Conv1d(3 ** 3, 3 ** 4, kernel_size=21, stride=9),
                                     nn.MaxPool1d(21, 21),
                                     nn.ReLU())

        self.seq3_c1 = nn.Sequential(nn.Linear(3 ** 4 * 25, 1),
                                     nn.Sigmoid())

        self.seq3_c2 = nn.Sequential(nn.Linear(3 ** 4 * 25, 1),
                                     nn.Sigmoid())

        self.seq3_c3 = nn.Sequential(nn.Linear(3 ** 4 * 25, 1),
                                     nn.Sigmoid())

    def forward(self, x):
        out = self.seq1(x)

        out_c1 = self.seq2_c1(out).view(-1, 3 ** 4 * 25)
        out_c2 = self.seq2_c2(out).view(-1, 3 ** 4 * 25)
        out_c3 = self.seq2_c3(out).view(-1, 3 ** 4 * 25)

        out_c1 = self.seq3_c1(out_c1)
        out_c2 = self.seq3_c2(out_c2)
        out_c3 = self.seq3_c3(out_c3)

        out = th.cat((out_c1, out_c2, out_c3), dim=1)

        return out


class ConvModel3SubModule(nn.Module):
    def __init__(self):
        super(ConvModel3SubModule, self).__init__()

        self.seq1 = nn.Sequential(nn.Conv1d(3, 3 ** 2, kernel_size=3),
                                  nn.MaxPool1d(3, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(3 ** 2, 3 ** 3, kernel_size=11, stride=5),
                                  nn.MaxPool1d(11, 11),
                                  nn.ReLU(),
                                  nn.Conv1d(3 ** 3, 3 ** 4, kernel_size=21, stride=9),
                                  nn.MaxPool1d(21, 21),
                                  nn.ReLU())

        self.seq2 = nn.Sequential(nn.Linear(3 ** 4 * 25, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        out = self.seq1(x)
        out = out.view(-1, 3 ** 4 * 25)
        out = self.seq2(out)
        return out


class ConvModel3(nn.Module):
    def __init__(self):
        super(ConvModel3, self).__init__()
        self.c1 = ConvModel3SubModule()
        self.c2 = ConvModel3SubModule()
        self.c3 = ConvModel3SubModule()

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(x)
        c3 = self.c3(x)
        return th.cat((c1, c2, c3), dim=1)


class LargeConvModel(nn.Module):
    def __init__(self):
        super(LargeConvModel, self).__init__()

        self.seq1 = nn.Sequential(nn.Conv1d(3, 16, kernel_size=64, stride=10),
                                  nn.ReLU(),
                                  nn.Conv1d(16, 32, kernel_size=128, stride=20),
                                  nn.ReLU(),
                                  nn.Conv1d(32, 64, kernel_size=256, stride=30),
                                  nn.ReLU())

        self.seq2 = nn.Sequential(nn.Linear(125 * 64, 3),
                                  nn.Sigmoid())

    def forward(self, x):
        out1 = self.seq1(x).view(-1, 125 * 64)
        out2 = self.seq2(out1)
        return out2


class SmallConvModel(nn.Module):
    def __init__(self):
        super(SmallConvModel, self).__init__()

        self.seq1 = nn.Sequential(nn.Conv1d(3, 50, kernel_size=5),
                                  nn.MaxPool1d(3, 3),
                                  nn.ReLU(),
                                  nn.Conv1d(50, 100, kernel_size=17),
                                  nn.MaxPool1d(100, 100),
                                  nn.ReLU(),
                                  nn.Conv1d(100, 200, kernel_size=35),
                                  nn.MaxPool1d(1000, 1000),
                                  nn.ReLU())

        self.seq2 = nn.Sequential(nn.Linear(200 * 2, 3),
                                  nn.Sigmoid())

    def forward(self, x):
        out = self.seq1(x).view(-1, 200 * 2)
        out = self.seq2(out)
        return out
