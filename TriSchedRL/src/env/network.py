from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Link:
    bandwidth_mbps: float
    rtt_ms: float
    loss: float
    overhead_ms: float = 1.0

class NetworkModel:
    def __init__(self):
        self.links: Dict[Tuple[str, str], Link] = {}

    def set_link(self, src: str, dst: str, link: Link) -> None:
        self.links[(src, dst)] = link

    def get_link(self, src: str, dst: str) -> Link:
        return self.links[(src, dst)]
