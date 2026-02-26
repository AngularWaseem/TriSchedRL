from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class ReplayBufferConfig:
    capacity: int = 100_000

class ReplayBuffer:
    def __init__(self, state_dim: int, cfg: ReplayBufferConfig):
        self.state_dim = int(state_dim)
        self.capacity = int(cfg.capacity)
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.sp = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)

    def add(self, s, a, r, sp, done: bool):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.sp[i] = sp
        self.done[i] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {"s": self.s[idx], "a": self.a[idx], "r": self.r[idx], "sp": self.sp[idx], "done": self.done[idx]}
