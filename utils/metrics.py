# utils/metrics.py
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from typing import Dict, List

def compute_hhi(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    shares_sq = [(v/total)**2 for v in counts.values()]
    return sum(shares_sq)

def compute_gini(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

def hop_dist_from_anchor(G: nx.Graph, anchor_id: int) -> Dict[int, int]:
    lengths = nx.single_source_shortest_path_length(G, anchor_id)
    return lengths

def spillover_curve(visits: Dict[int,int], hop_dist: Dict[int,int]) -> Dict[int, int]:
    byhop = defaultdict(int)
    for node, cnt in visits.items():
        h = hop_dist.get(node, None)
        if h is not None:
            byhop[h] += cnt
    return dict(sorted(byhop.items()))
