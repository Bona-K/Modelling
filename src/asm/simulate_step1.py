import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass

class Params:
    n_nodes = 20     
    n_anchors = 2      
    steps = 10
    # adding threshold rule
    theta_same = 0.6        

def build_mall(params):
    G = nx.erdos_renyi_graph(params.n_nodes, 0.2, seed=1) 
    anchors = set(np.random.choice(params.n_nodes, params.n_anchors, replace=False))
    for n in G.nodes:
        G.nodes[n]["type"] = "anchor" if n in anchors else "tenant"
        G.nodes[n]["A"] = 3 if n in anchors else 1
    return G, anchors

def simulate(params):
    G, anchors = build_mall(params)   
    A0 = next(iter(anchors))          
    customers = np.full(20, A0, dtype=int) 
    footfall = np.zeros(params.n_nodes, dtype=int)
    p_stay_anchor = 0.3 

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
            curr_cat = G.nodes[here]["type"]

           #finding same neighbors
            same_neighbors = [v for v in neighbors if G.nodes[v]["type"] == curr_cat]

           #calculate same category ratio
            ratio_same = len(same_neighbors) / len(neighbors) if neighbors else 0.0

           #threshold rule
            if G.nodes[here]["type"] == "anchor" and np.random.random() < p_stay_anchor:
                nxt = here
            elif len(same_neighbors) > 0 and ratio_same >= params.theta_same:
                nxt = int(np.random.choice(same_neighbors))
            else:
                pool = neighbors + [here]
                weights = np.array([G.nodes[v]["A"] for v in pool], dtype=float)
                probs = weights / weights.sum()
                nxt = int(np.random.choice(pool, p=probs))

            customers[i] = nxt
            footfall[nxt] += 1

    df = pd.DataFrame({
        "node": np.arange(params.n_nodes),
        "type": [G.nodes[n]["type"] for n in G.nodes],
        "footfall": footfall
    })
    return df

print(simulate(Params()))


