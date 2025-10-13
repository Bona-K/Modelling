# Step3 = Step2 + (NEW) 3-way categories + (NEW) purchase logic

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from utils.io import ensure_dir

@dataclass
class Params:
    n_nodes: int = 20
    n_anchors: int = 1
    steps: int = 24
    theta_same: float = 0.5
    p_edge: float = 0.2
    n_agents: int = 120
    p_stay_anchor: float = 0.2
    p_move_below: float = 0.7
    anti_backtrack: bool = True
    seed: int = 1
    count_t0: bool = True

def build_mall(p: Params):
    rng = np.random.default_rng(p.seed)
    G = nx.erdos_renyi_graph(p.n_nodes, p.p_edge, seed=p.seed)
    anchors = {0}  # fixed anchor (same as step2)

    for n in G.nodes:
        if n in anchors:
            G.nodes[n]["role"] = "anchor"
            G.nodes[n]["category"] = "similar"
            G.nodes[n]["A"] = 3
            G.nodes[n]["purchase_prob"] = 0.20     # (NEW) anchor purchase prob
        else:
            G.nodes[n]["role"] = "tenant"
            G.nodes[n]["category"] = rng.choice(["similar", "complementary", "different"])  # (CHANGED) 3 categories
            G.nodes[n]["A"] = 1
            cat = G.nodes[n]["category"]
            purchase_prob = {"similar": 0.18, "complementary": 0.35, "different": 0.12}
            G.nodes[n]["purchase_prob"] = purchase_prob.get(cat, 0.12)  # (NEW) add purchase prob

    return G, anchors

def simulate(p: Params):
    rng = np.random.default_rng(p.seed)
    G, anchors = build_mall(p)
    A0 = next(iter(anchors))
    customers = np.full(p.n_agents, A0, dtype=int)
    prev_pos = np.full(p.n_agents, A0, dtype=int)
    footfall = np.zeros(p.n_nodes, dtype=int)
    sales = np.zeros(p.n_nodes, dtype=int)   # (NEW) sales counter

    if p.count_t0:
        footfall[A0] += p.n_agents

    for _ in range(p.steps):
        for i in range(p.n_agents):
            here = customers[i]
            neighbors = list(G.neighbors(here)) or [here]

            curr_cat = G.nodes[here]["category"]
            same_neighbors = [v for v in neighbors if G.nodes[v]["category"] == curr_cat]
            comp_neighbors = [v for v in neighbors if G.nodes[v]["category"] == "complementary"]  # (NEW)
            ratio_same = len(same_neighbors) / len(neighbors) if neighbors else 0.0

            if G.nodes[here]["role"] == "anchor" and rng.random() < p.p_stay_anchor:
                nxt = here
            elif same_neighbors and ratio_same >= p.theta_same:
                nxt = int(rng.choice(same_neighbors))
            else:
                if rng.random() < p.p_move_below:
                    weights = np.array(
                        [1.2 if G.nodes[v]["category"] == "complementary" else 1.0
                         for v in neighbors],
                         dtype=float
                         )
                    probs = weights / weights.sum()
                    nxt = int(rng.choice(neighbors, p=probs))
                else:
                    nxt = here

            if p.anti_backtrack and nxt == prev_pos[i] and rng.random() < 0.5:
                nxt = here

            prev_pos[i] = here
            customers[i] = nxt
            footfall[nxt] += 1

            if rng.random() < G.nodes[nxt]["purchase_prob"]:  # (NEW) record sales
                sales[nxt] += 1

    df = pd.DataFrame({
        "node": np.arange(p.n_nodes),
        "role": [G.nodes[n]["role"] for n in G.nodes],
        "category": [G.nodes[n]["category"] for n in G.nodes],
        "footfall": footfall,
        "sales": sales,   # (NEW)
    })

    ensure_dir("data/outputs")
    df.to_csv("data/outputs/step3.csv", index=False)
    print("[Saved] data/outputs/step3.csv")

    return df, A0

if __name__ == "__main__":
    df, A0 = simulate(Params())
    print("Anchor fixed at node:", A0)
    print(df.sort_values(["sales", "footfall"], ascending=False).head())