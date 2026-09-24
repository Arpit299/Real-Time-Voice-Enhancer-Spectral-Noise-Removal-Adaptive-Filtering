import argparse
import math
import os
import struct
import sys
import wave
from collections import deque
import numpy as np
try:
    import sounddevice as sd
except Exception:
    sd = None
class RingBuffer:
    def __init__(self, size):
        self.data = deque([0.0] * size, maxlen=size)
    def extend(self, values):
        self.data.extend(float(v) for v in values)
    def values(self):
        return np.fromiter(self.data, dtype=np.float32)
class AdaptiveNLMS:
    def __init__(self, order=12, step=0.2, leakage=0.999):
        self.order = int(order)
        self.step = float(step)
        self.leakage = float(leakage)
        self.weights = np.zeros(self.order, dtype=np.float32)
        self.history = deque([0.0] * self.order, maxlen=self.order)
    def reset(self):
        self.weights.fill(0.0)
        self.history = deque([0.0] * self.order, maxlen=self.order)
    def process(self, samples, amount=0.18):
        x = np.asarray(samples, dtype=np.float32)
        y = np.empty_like(x)
        eps = 1e-7
        for i in range(x.size):
            ref = np.fromiter(reversed(self.history), dtype=np.float32, count=self.order)
            predicted = float(np.dot(self.weights, ref))
            error = float(x[i] - predicted)
            power = float(np.dot(ref, ref)) + eps
            mu = self.step / power
            self.weights = (self.leakage * self.weights + mu * error * ref).astype(np.float32)
            y[i] = float(x[i] - amount * predicted)
            self.history.append(float(x[i]))
        return y
class SpectralSuppressor:
    def __init__(self, blocksize=1024, sample_rate=48000, floor=0.08, strength=1.15, attack=0.75, release=0.985):
        self.blocksize = int(blocksize)
        self.sample_rate = int(sample_rate)
        self.floor = float(floor)
        self.strength = float(strength)
        self.attack = float(attack)
        self.release = float(release)
        self.window = np.hanning(self.blocksize).astype(np.float32)
        self.noise_power = np.full(self.blocksize // 2 + 1, 1e-6, dtype=np.float32)
        self.initialized = False
        self.frames = 0
    def update_noise(self, power, alpha=None):
        a = self.attack if alpha is None else float(alpha)
        if not self.initialized:
            self.noise_power = np.maximum(power.astype(np.float32), 1e-8)
            self.initialized = True
        else:
            self.noise_power = a * self.noise_power + (1.0 - a) * power.astype(np.float32)
    def calibrate(self, block):
        x = np.asarray(block, dtype=np.float32)
        if x.size != self.blocksize:
            x = np.pad(x, (0, max(0, self.blocksize - x.size)))[:self.blocksize]
        z = x * self.window
        spec = np.fft.rfft(z)
        power = np.abs(spec).astype(np.float32) ** 2
        self.update_noise(power, alpha=0.35 if not self.initialized else 0.85)
        self.frames += 1
    def process(self, block):
        x = np.asarray(block, dtype=np.float32)
        original_size = x.size
        if original_size < self.blocksize:
            x = np.pad(x, (0, self.blocksize - original_size))
        elif original_size > self.blocksize:
            x = x[:self.blocksize]
            original_size = self.blocksize
        energy = float(np.mean(x * x)) + 1e-10
        z = x * self.window
        spec = np.fft.rfft(z)
        power = np.abs(spec).astype(np.float32) ** 2
        noise_mean = float(np.mean(self.noise_power)) + 1e-10
        noise_ratio = float(np.mean(power) / noise_mean)
        if not self.initialized:
            self.update_noise(power, alpha=0.25)
        elif noise_ratio < 2.4:
            self.update_noise(power, alpha=self.release)
        snr = np.maximum(power / (self.noise_power + 1e-10) - 1.0, 0.0)
        gain = (snr / (snr + 1.0 + 1e-10)) ** self.strength
        gain = np.maximum(gain, self.floor)
        gain[0] *= 0.35
        out = np.fft.irfft(spec * gain, n=self.blocksize).astype(np.float32)
        out *= 1.0 / max(0.12, float(np.mean(self.window)) * 2.0)
        peak = float(np.max(np.abs(out))) + 1e-8