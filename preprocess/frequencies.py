from numpy.fft import *
import numpy as np


def filter_signal(signal, threshold=1e8):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)


def applat(signal, seuil=1e8):
    return signal - filter_signal(signal, threshold=seuil)


def normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std
