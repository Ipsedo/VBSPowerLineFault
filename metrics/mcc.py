from math import sqrt
import torch as th


class MCCMeter:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def add(self, output, target):
        self.TP += th.sum((output > 0.5) & (target == 1)).item()
        self.FP += th.sum((output > 0.5) & (target == 0)).item()
        self.TN += th.sum((output < 0.5) & (target == 0)).item()
        self.FN += th.sum((output < 0.5) & (target == 1)).item()

    def reset(self):
        self.FP = 0
        self.FP = 0
        self.TN = 0
        self.TP = 0

    def value(self):
        return (self.TP * self.TN - self.FP * self.FN) \
               / sqrt((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))
