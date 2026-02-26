from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import Actor, Critic

@dataclass
class ACTrainConfig:
    gamma: float = 0.99
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    batch_size: int = 128
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    device: str = "cpu"

class ActorCriticAgent:
    def __init__(self, state_dim: int, action_dim: int, train_cfg: ACTrainConfig):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=train_cfg.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=train_cfg.lr_critic)

    @torch.no_grad()
    def act(self, s: np.ndarray) -> Tuple[int, np.ndarray]:
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(s_t).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        a = int(dist.sample().item())
        probs = dist.probs.detach().cpu().numpy()
        return a, probs

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        s = torch.tensor(batch["s"], dtype=torch.float32, device=self.device)
        a = torch.tensor(batch["a"], dtype=torch.int64, device=self.device)
        r = torch.tensor(batch["r"], dtype=torch.float32, device=self.device)
        sp = torch.tensor(batch["sp"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            v_next = self.critic(sp)
            target = r + self.cfg.gamma * (1.0 - done) * v_next

        v = self.critic(s)
        td = target - v
        critic_loss = (td ** 2).mean()

        self.opt_critic.zero_grad()
        critic_loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.opt_critic.step()

        logits = self.actor(s)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        entropy = dist.entropy().mean()

        adv = td.detach()
        actor_loss = -(logp * adv).mean() - self.cfg.entropy_coef * entropy

        self.opt_actor.zero_grad()
        actor_loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.opt_actor.step()

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()), "entropy": float(entropy.item())}
