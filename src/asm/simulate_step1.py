import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from utils.io import ensure_dir 

@dataclass
class Params:
    n_nodes: int = 20
    n_anchors: int = 1
    steps: int = 10
    theta_same: float = 0.6 #threshold rule: min. ratio of same-category neighbors required to move toward them
    p_edge: float = 0.2
    n_agents: int = 100
    p_stay_anchor: float = 0.3 #probability of staying longer at anchor (anchor quality factor)
    seed: int = 1
    count_t0 = True #whether to count t=0 (initial anchor positions) in footfall
  

def build_mall(params):
    rng = np.random.default_rng(params.seed)
    G = nx.erdos_renyi_graph(params.n_nodes, params.p_edge, seed=params.seed)
    anchors = set(rng.choice(params.n_nodes, params.n_anchors, replace=False))
    for n in G.nodes:
        if n in anchors:
            G.nodes[n]["role"] = "anchor"
            G.nodes[n]["category"] = "similar"
            G.nodes[n]["A"] = 3
        else:
            G.nodes[n]["role"] = "tenant"
            G.nodes[n]["category"] = rng.choice(["similar", "different"])
            G.nodes[n]["A"] = 1
    return G, anchors


def simulate(params):
    # create local random generator (same result every run with same seed)
    rng = np.random.default_rng(params.seed)
    G, anchors = build_mall(params)   
    A0 = next(iter(anchors))          
    customers = np.full(20, A0, dtype=int) 
    footfall = np.zeros(params.n_nodes, dtype=int)
    p_stay_anchor = 0.3 

    if params.count_t0:
        for c in customers:
            footfall[c] += 1

    for _ in range(params.steps):
        for i in range(len(customers)):
            here = customers[i]
            neighbors = list(G.neighbors(here))
            # prevent isolation
            if not neighbors:
                neighbors = [here]
           
           #checking current store category
            curr_cat = G.nodes[here]["category"]

           #finding same neighbors
            same_neighbors = [v for v in neighbors if G.nodes[v]["category"] == curr_cat]

           #calculate same category ratio
            ratio_same = len(same_neighbors) / len(neighbors) if neighbors else 0.0

           #threshold rule
            if G.nodes[here]["role"] == "anchor" and np.random.random() < p_stay_anchor:
                nxt = here
            elif len(same_neighbors) > 0 and ratio_same >= params.theta_same:
                nxt = int(rng.choice(same_neighbors))
            else:
                pool = neighbors + [here]
                weights = np.array([G.nodes[v]["A"] for v in pool], dtype=float)
                probs = weights / weights.sum()
                nxt = int(rng.choice(pool, p=probs))

            customers[i] = nxt
            footfall[nxt] += 1

    df = pd.DataFrame({
        "node": np.arange(params.n_nodes),
        "role": [G.nodes[n]["role"] for n in G.nodes],
        "category": [G.nodes[n]["category"] for n in G.nodes],
        "footfall": footfall
    })
    ensure_dir("data/outputs")
    df.to_csv("data/outputs/step1.csv", index=False)
    print("[Saved] data/outputs/step1.csv")
    return df

print(simulate(Params()))