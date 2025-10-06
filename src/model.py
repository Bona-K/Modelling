# src/model.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Agent:
    id: int
    dwell_left: int
    position: Optional[int] = None
    prev_position: Optional[int] = None
