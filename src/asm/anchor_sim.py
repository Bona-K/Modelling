from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
import os
from utils import ensure_dir

@dataclass
class Params:
    n_nodes: int = 40
    p_edge: float = 0.2
    categories: tuple = ("similar", "different")

    theta_same: float = 0.6
    p_move_below: float = 0.4
    p_stay_origin: float = 0.3
    anti_backtrack: bool = True
    backtrack_penalty: float = 0.5

# 1 tick = 15 mins
    min_ticks: int = 4
    max_ticks: int = 8
    purchase_prob = {"origin": 0.20, "similar": 0.25, "different": 0.25}

    n_agents: int = 200
    steps: int = 12
    seed: int = 42

    output_dir: str = "data/outputs"
    save_csv: bool = True
    csv_name: str = "asm_results.csv"

class AnchorSim:
    def __init__(self, params: Params):
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        self.mall_graph, self.start_node = self._build_graph()
        self.customer_positions = None
        self.previous_positions = None
        self.time_left = None
        self.visit_counts = None
        self.sales_counts = None
        self.exit_nodes = None
        self.traces = []
        self.t = 0

    def _build_graph(self):
        G = nx.erdos_renyi_graph(self.p.n_nodes, self.p.p_edge, seed=self.p.seed)
        origin = int(self.rng.integers(0, self.p.n_nodes))
        for n in G.nodes:
            is_origin = (n == origin)
            G.nodes[n]["role"] = "origin" if is_origin else "store"
            G.nodes[n]["category"] = "origin" if is_origin else self.rng.choice(self.p.categories)
            G.nodes[n]["A"] = 3 if is_origin else 1
            cat = G.nodes[n]["category"]
            G.nodes[n]["purchase_prob"] = float(self.p.purchase_prob.get(cat, 0.2))
        
        for n in list(G.nodes):
            if G.degree[n] == 0:
                G.add_edge(n, n)
        return G, origin

    def _neighbors(self, node: int):
        neigh = list(self.mall_graph.neighbors(node))
        return neigh if len(neigh) > 0 else [node]

    def reset(self):
        N, V = self.p.n_agents, self.p.n_nodes
        self.customer_positions = np.full(N, self.start_node, dtype=int)
        self.previous_positions = np.full(N, -1, dtype=int)
        self.time_left = self.rng.integers(self.p.min_ticks, self.p.max_ticks+1, size=N)
        self.visit_counts = np.zeros(V, dtype=int)
        self.sales_counts = np.zeros(V, dtype=int)
        self.exit_nodes = np.full(N, -1, dtype=int)
        self.traces = []
        for pos in self.customer_positions:
            self.visit_counts[pos] += 1
            self._maybe_purchase(pos)
        self.traces.append(self.customer_positions.copy())
        self.t = 0

    def _maybe_purchase(self, node: int):
        if self.rng.random() < self.mall_graph.nodes[node]["purchase_prob"]:
            self.sales_counts[node] += 1
    
    def _complement_neighbors(self, node: int, neighbors):
        curr = self.mall_graph.nodes[node]["category"]
        if curr == "origin":
            target = {"similar", "different"}
        else:
            traget = {different} if curr == "similar" else {"similar"}
        return [ v for v in neighbors if self.mall_graph.nodes[v]["category"] in target]
    
    def step(self):
        N = len(self.customer_positions)
        for i in range(N):
            if self.time_left[i] <= 0:
                continue
            current = int(self.customer_positions[i])
            neighbors = self._neighbors(current)

            curr_cat = self.mall_graph.nodes[current]["category"]
            same_neighbors = [v for v in neighbors if self.mall_graph.nodes[v]["category"] == curr_cat]
            same_ratio = len(same_neighbors) / len(neighbors) if neighbors else 0.0

            
