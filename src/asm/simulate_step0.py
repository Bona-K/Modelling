import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass

class Params:
    n_nodes = 20     
    n_anchors = 2      
    steps = 10        

def build_mall(params):
    G = nx.erdos_renyi_graph(params.n_nodes, 0.2, seed=1) 
    anchors = set(np.random.choice(params.n_nodes, params.n_anchors, replace=False))
    for n in G.nodes:
        G.nodes[n]["type"] = "anchor" if n in anchors else "tenant"
        G.nodes[n]["A"] = 3 if n in anchors else 1
    return G

def simulate(params):
    G = build_mall(params)
    customers = np.random.randint(0, params.n_nodes, size=20) 
    footfall = np.zeros(params.n_nodes, dtype=int)

    for c in customers:
        footfall[c] += 1

    for _ in range(params.steps):
        for i in range(len(customers)):
            here = customers[i]
            neigh = list(G.neighbors(here)) or [here]
           
            weights = [G.nodes[v]["A"] for v in neigh]
            probs = np.array(weights) / sum(weights)
            nxt = np.random.choice(neigh, p=probs)
            customers[i] = nxt
            footfall[nxt] += 1

    df = pd.DataFrame({
        "node": np.arange(params.n_nodes),
        "type": [G.nodes[n]["type"] for n in G.nodes],
        "footfall": footfall
    })
    return df

print(simulate(Params()))
