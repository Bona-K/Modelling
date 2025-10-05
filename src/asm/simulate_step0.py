import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from utils import ensure_dir

@dataclass
class Params:
    n_nodes = 20
    n_anchors = 1
    p_edge = 0.2     
    n_agents = 100      
    steps = 10
    seed = 1        

def build_graph(params):
    G = nx.erdos_renyi_graph(params.n_nodes, params.p_edge, seed=params.seed)
    anchors = set(np.random.choice(params.n_nodes, params.n_anchors, replace=False))
    for n in G.nodes:
        if n in anchors:
            G.nodes[n]["role"] = "anchor"
            G.nodes[n]["category"] = "similar"
            G.nodes[n]["A"] = 3
        else:
            G.nodes[n]["role"] = "tenant"
            G.nodes[n]["category"] = np.random.choice(["similar","different"])
            G.nodes[n]["A"] = 1
    return G, anchors

def simulate(params):
    G, anchors = build_graph(params)
    A0 = next(iter(anchors))
    positions = np.full(params.n_agents, A0, dtype = int)
    footfall = np.zeros(params.n_nodes, dtype=int)

    for c in positions:
        footfall[c] += 1

    for _ in range(params.steps):
        for i in range(params.n_agents):
            here = positions[i]
            neighbors = list(G.neighbors(here))
            if not neighbors:
                neighbors = [here]
            nxt = np.random.choice(neighbors)
            positions[i] = nxt
            footfall[nxt] += 1

    df = pd.DataFrame({
        "node": np.arange(params.n_nodes),
        "role": [G.nodes[n]["role"] for n in G.nodes],
        "category": [G.nodes[n]["category"] for n in G.nodes],
        "footfall": footfall
    })
    ensure_dir("data/outputs")
    df.to_csv("data/outputs/step0.csv", index= False)
    return df

if __name__ == "__main__":
    df = simulate(Params())
    print(df.head())