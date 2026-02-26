from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn

def mlp(sizes: List[int], activation=nn.ReLU, out_activation=None) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)

@dataclass
class ActorCriticConfig:
    state_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (256, 256)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([state_dim, *hidden_sizes, action_dim])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.v = mlp([state_dim, *hidden_sizes, 1])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.v(s).squeeze(-1)
